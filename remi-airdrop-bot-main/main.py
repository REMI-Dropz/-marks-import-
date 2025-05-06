from xumm import XummSdk
import asyncio
import logging
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, Header
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
from xrpl.asyncio.clients import AsyncWebsocketClient
from xrpl.models.requests import AccountInfo, Ledger, AccountLines, ServerInfo, GatewayBalances
from xrpl.models.requests import GenericRequest
from xrpl.utils import xrp_to_drops
import uvicorn
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Serve index.html
@app.get("/")
async def serve_index():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
XAMAN_API_KEY = os.getenv("XAMAN_API_KEY")
XAMAN_API_SECRET = os.getenv("XAMAN_API_SECRET")
if not XAMAN_API_KEY or not XAMAN_API_SECRET:
    logger.error("XAMAN_API_KEY or XAMAN_API_SECRET not set in .env file")
    raise ValueError("XAMAN_API_KEY and XAMAN_API_SECRET must be set in .env file")
FEE_WALLET_ADDRESS = "rNtwcwRkSJE7kAE3pEzt93txNf3WzdxeZy"
FLUX_ISSUER = "rhbmVVzvDme96hHsb2DxKKKfxqnMexB2mz"
FLUX_CURRENCY = "464C555800000000000000000000000000000000"
TRUSTLINE_RESERVE_XRP = 2.0
NFT_CACHE_FILE = "nft_cache.json"
CLIO_URL = "https://s2.ripple.com:51234"
DEFAULT_NFT_ISSUER = "rnmmDJh7heit6rad15Fa8AxXtyWBKmucFH"

# Pydantic models
class Wallet(BaseModel):
    address: str
    amount: Optional[float] = 0
    nft_count: Optional[int] = 0  # Added for NFT holders

class AirdropRequest(BaseModel):
    token_type: str
    total_amount: float
    issuer: Optional[str] = None
    currency: Optional[str] = None
    wallets: List[Wallet]
    account: str
    use_nft_holders: Optional[bool] = False  # Flag for NFT-based airdrop
    nft_issuer: Optional[str] = None  # NFT issuer address

class Token(BaseModel):
    name: str
    issuer: str
    currency: str

# Load or initialize NFT cache
if os.path.exists(NFT_CACHE_FILE):
    with open(NFT_CACHE_FILE, 'r') as f:
        nft_cache = json.load(f)
else:
    nft_cache = {"last_updated": None, "data": {}}
    with open(NFT_CACHE_FILE, 'w') as f:
        json.dump(nft_cache, f)

# Dependency for access token
async def get_access_token(authorization: str = Header(...), account: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    logger.info(f"Validating access token: {token} for account: {account}")
    return {"token": token, "account": account}

# Get XRPL client
async def get_xrpl_client():
    primary_node = "wss://s1.ripple.com"
    fallback_node = "wss://s2.ripple.com"
    logger.info(f"Connecting to primary Mainnet node: {primary_node}")
    client = None
    try:
        client = AsyncWebsocketClient(primary_node)
        await client.open()
        if client.is_open():
            response = await client.request(ServerInfo())
            if response.is_successful():
                network_id = response.result.get("info", {}).get("network_id", 0)
                if network_id != 0:
                    logger.error(f"Connected to non-Mainnet network (network_id: {network_id})")
                    await client.close()
                    raise HTTPException(status_code=500, detail="Connected to incorrect network.")
                logger.info(f"Connected to primary Mainnet node: {response.result}")
                return client
            else:
                logger.warning("Primary node response invalid, trying fallback...")
                await client.close()
    except Exception as e:
        logger.warning(f"Failed to connect to primary node: {str(e)}, trying fallback...")
        if client and client.is_open():
            await client.close()

    logger.info(f"Connecting to fallback Mainnet node: {fallback_node}")
    try:
        client = AsyncWebsocketClient(fallback_node)
        await client.open()
        if client.is_open():
            response = await client.request(ServerInfo())
            if response.is_successful():
                network_id = response.result.get("info", {}).get("network_id", 0)
                if network_id != 0:
                    logger.error(f"Connected to non-Mainnet network (network_id: {network_id})")
                    await client.close()
                    raise HTTPException(status_code=500, detail="Connected to incorrect network.")
                logger.info(f"Connected to fallback Mainnet node: {response.result}")
                return client
            else:
                logger.error("Fallback node response invalid.")
                await client.close()
    except Exception as e:
        logger.error(f"Failed to connect to fallback node: {str(e)}")
        if client and client.is_open():
            await client.close()
    raise HTTPException(status_code=500, detail="Failed to connect to XRP Ledger Mainnet nodes.")

# Get current network fee
async def get_current_fee(client: AsyncWebsocketClient) -> int:
    try:
        response = await asyncio.wait_for(client.request(Ledger(ledger_index="validated")), timeout=30)
        if not response.is_successful():
            raise Exception("Failed to fetch ledger data")
        base_fee = int(response.result["ledger"]["base_fee"])
        logger.info(f"Fetched base fee: {base_fee} drops")
        return base_fee
    except Exception as e:
        logger.error(f"Fee fetch error: {str(e)}")
        return 12

# Decode hex currency
def decode_hex_currency(hex_currency: str) -> str:
    if not hex_currency or len(hex_currency) != 40 or not hex_currency.isalnum():
        return hex_currency
    try:
        result = ''
        for i in range(0, len(hex_currency), 2):
            byte = int(hex_currency[i:i+2], 16)
            if byte == 0:
                break
            result += chr(byte)
        return result or hex_currency
    except Exception as e:
        logger.warning(f"Failed to decode hex currency {hex_currency}: {str(e)}")
        return hex_currency

# Fetch NFT holders from Clio
async def fetch_nft_holders(nft_issuer: str, force_refresh: bool = False):
    global nft_cache
    try:
        # Check cache if not forcing refresh
        if not force_refresh:
            last_updated = nft_cache.get("last_updated")
            if last_updated:
                last_updated_time = datetime.fromisoformat(last_updated)
                if datetime.now() - last_updated_time < timedelta(hours=1):
                    logger.info("Using cached NFT data")
                    return nft_cache["data"]

        # Fetch NFTs by issuer
        all_nfts = []
        marker = None
        while True:
            payload = {
                "method": "nfts_by_issuer",
                "params": [{"issuer": nft_issuer, "limit": 400}]
            }
            if marker:
                payload["params"][0]["marker"] = marker
            response = requests.post(CLIO_URL, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "result" not in data or "nfts" not in data["result"]:
                logger.error("No NFTs found in response")
                return {}
            all_nfts.extend(data["result"]["nfts"])
            marker = data["result"].get("marker")
            logger.info(f"Fetched {len(all_nfts)} NFTs so far...")
            if not marker:
                break

        # Fetch NFT owners
        holders = {}
        for nft in all_nfts:
            nft_id = nft["nft_id"]
            try:
                payload = {"method": "nft_info", "params": [{"nft_id": nft_id}]}
                response = requests.post(CLIO_URL, json=payload, timeout=5)
                response.raise_for_status()
                data = response.json()
                if "result" in data and "owner" in data["result"]:
                    owner = data["result"]["owner"]
                    holders[owner] = holders.get(owner, 0) + 1
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching owner for NFT {nft_id}: {e}")

        # Update cache
        nft_cache = {
            "last_updated": datetime.now().isoformat(),
            "data": holders
        }
        with open(NFT_CACHE_FILE, 'w') as f:
            json.dump(nft_cache, f)
        logger.info("NFT cache updated")
        return holders
    except Exception as e:
        logger.error(f"Error fetching NFT holders: {e}")
        return {}

# New endpoint to get NFT holders
@app.get("/nft-holders")
async def get_nft_holders(
    nft_issuer: str = Query(DEFAULT_NFT_ISSUER),
    force_refresh: bool = Query(False),
    token_data: dict = Depends(get_access_token)
):
    logger.info(f"Fetching NFT holders for issuer: {nft_issuer}, force_refresh: {force_refresh}")
    holders = await fetch_nft_holders(nft_issuer, force_refresh)
    return [
        {"address": address, "nft_count": count}
        for address, count in holders.items()
        if count > 0
    ]

# Modified /check-balances endpoint
@app.post("/check-balances")
async def check_balances(
    wallets: List[Wallet],
    token_type: str = Query(...),
    issuer: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    token_data: dict = Depends(get_access_token)
):
    logger.info(f"Checking balances for wallets: {wallets}, token_type: {token_type}, issuer: {issuer}, currency: {currency}")
    decoded_token_type = decode_hex_currency(token_type)
    decoded_currency = decode_hex_currency(currency) if currency else decode_hex_currency(token_type)
    if decoded_token_type != "XRP" and (not issuer or not decoded_currency):
        raise HTTPException(status_code=400, detail="Issuer and currency required for token balance checks")
    client = None
    try:
        client = await get_xrpl_client()
        results = []
        for wallet in wallets:
            wallet.address = wallet.address.strip()
            try:
                account_info_request = AccountInfo(account=wallet.address, ledger_index="validated")
                account_response = await asyncio.wait_for(client.request(account_info_request), timeout=30)
                if not account_response.is_successful():
                    logger.warning(f"Account {wallet.address} does not exist or is not funded")
                    results.append({
                        "address": wallet.address,
                        "has_balance": False,
                        "error": "Account not found or not funded"
                    })
                    continue
                xrp_balance = float(account_response.result["account_data"]["Balance"]) / 1_000_000
                if xrp_balance < TRUSTLINE_RESERVE_XRP:
                    logger.warning(f"Insufficient XRP balance for {wallet.address}: {xrp_balance} XRP")
                    results.append({
                        "address": wallet.address,
                        "has_balance": False,
                        "error": f"Insufficient XRP balance ({xrp_balance} XRP)"
                    })
                    continue
                if decoded_token_type == "XRP":
                    results.append({
                        "address": wallet.address,
                        "has_balance": True,
                        "error": None
                    })
                else:
                    trustline_request = GenericRequest(
                        command="account_lines",
                        account=wallet.address,
                        ledger_index="validated"
                    )
                    trustline_response = await asyncio.wait_for(client.request(trustline_request), timeout=30)
                    if not trustline_response.is_successful():
                        logger.error(f"Trustline request failed for {wallet.address}")
                        results.append({
                            "address": wallet.address,
                            "has_balance": False,
                            "error": "Failed to fetch trustlines"
                        })
                        continue
                    trustlines = trustline_response.result.get("lines", [])
                    has_balance = any(
                        line["account"] == issuer and decode_hex_currency(line["currency"]) == decoded_currency
                        and float(line["balance"]) > 0
                        for line in trustlines
                    )
                    results.append({
                        "address": wallet.address,
                        "has_balance": has_balance,
                        "error": None if has_balance else "No token balance found"
                    })
            except Exception as e:
                logger.error(f"Error checking balance for {wallet.address}: {str(e)}")
                results.append({
                    "address": wallet.address,
                    "has_balance": False,
                    "error": str(e)
                })
        return results
    except Exception as e:
        logger.error(f"Balance check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check balances")
    finally:
        if client and client.is_open():
            await client.close()

# Modified /check-trustlines endpoint
@app.post("/check-trustlines")
async def check_trustlines(
    wallets: List[Wallet],
    token_type: str = Query(...),
    issuer: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    token_data: dict = Depends(get_access_token)
):
    logger.info(f"Checking trustlines for wallets: {wallets}, token_type: {token_type}, issuer: {issuer}, currency: {currency}")
    decoded_token_type = decode_hex_currency(token_type)
    decoded_currency = decode_hex_currency(currency) if currency else decode_hex_currency(token_type)
    if decoded_token_type != "XRP" and (not issuer or not decoded_currency):
        raise HTTPException(status_code=400, detail="Issuer and currency required")
    client = None
    try:
        client = await get_xrpl_client()
        results = []
        for wallet in wallets:
            wallet.address = wallet.address.strip()
            try:
                if decoded_token_type == "XRP":
                    results.append({
                        "address": wallet.address,
                        "has_trustline": True
                    })
                else:
                    account_info_request = AccountInfo(account=wallet.address, ledger_index="validated")
                    account_response = await asyncio.wait_for(client.request(account_info_request), timeout=30)
                    if not account_response.is_successful():
                        logger.warning(f"Account {wallet.address} does not exist or is not funded")
                        results.append({
                            "address": wallet.address,
                            "has_trustline": False
                        })
                        continue
                    trustline_request = GenericRequest(
                        command="account_lines",
                        account=wallet.address,
                        ledger_index="validated"
                    )
                    response = await asyncio.wait_for(client.request(trustline_request), timeout=30)
                    if not response.is_successful():
                        logger.error(f"Trustline request failed for {wallet.address}")
                        results.append({
                            "address": wallet.address,
                            "has_trustline": False
                        })
                        continue
                    trustlines = response.result.get("lines", [])
                    trustline_exists = any(
                        line["account"] == issuer and decode_hex_currency(line["currency"]) == decoded_currency
                        for line in trustlines
                    )
                    results.append({
                        "address": wallet.address,
                        "has_trustline": trustline_exists
                    })
            except Exception as e:
                logger.error(f"Error checking trustline for {wallet.address}: {str(e)}")
                results.append({
                    "address": wallet.address,
                    "has_trustline": False
                })
        return results
    except Exception as e:
        logger.error(f"Trustline check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check trustlines")
    finally:
        if client and client.is_open():
            await client.close()

# Initiate OAuth
xumm = XummSdk(XAMAN_API_KEY, XAMAN_API_SECRET)

@app.post("/initiate-oauth")
async def initiate_oauth():
    logger.info("Initiating OAuth")
    try:
        payload = xumm.payload.create({"TransactionType": "SignIn", "options": {"push": True}})
        return {
            "payload_uuid": payload.uuid,
            "qr_code_url": f"https://xumm.app/sign/{payload.uuid}_q.png",
            "authorize_url": payload.next.always,
            "websocket_url": payload.refs.websocket_status,
            "mobile_url": payload.refs.deeplink if hasattr(payload.refs, "deeplink") else payload.next.always,
            "pushed": payload.pushed
        }
    except Exception as e:
        logger.error(f"Error in initiate_oauth: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate OAuth: {str(e)}")

# Callback for OAuth
@app.get("/callback")
async def callback(payload_uuid: str):
    headers = {
        "X-API-Key": XAMAN_API_KEY,
        "X-API-Secret": XAMAN_API_SECRET
    }
    response = requests.get(f"https://xumm.app/api/v1/platform/payload/{payload_uuid}", headers=headers)
    data = response.json()
    logger.info(f"Callback response for payload {payload_uuid}: {data}")
    
    if (
        response.status_code == 200 
        and data.get("meta", {}).get("exists", False)
        and data.get("meta", {}).get("signed", False)
        and data.get("response", {}).get("account")
        and data.get("application", {}).get("issued_user_token")
    ):
        account = data["response"]["account"]
        issued_user_token = data["application"]["issued_user_token"]
        token_response = requests.post(
            "https://xumm.app/api/v1/platform/user-token",
            headers=headers,
            json={"user_token": data["response"]["hex"]}
        )
        if token_response.status_code == 200:
            token_data = token_response.json()
            user_token = token_data.get("token", issued_user_token)
        else:
            user_token = issued_user_token
        return {
            "meta": {"signed": True},
            "application": {"issued_user_token": user_token},
            "response": {"account": account}
        }
    elif data.get("meta", {}).get("cancelled", False):
        raise HTTPException(status_code=400, detail="Payload was cancelled")
    elif data.get("meta", {}).get("expired", False):
        raise HTTPException(status_code=400, detail="Payload has expired")
    else:
        return JSONResponse(status_code=202, content={"status": "pending"})

# Get token holdings
@app.get("/get-tokens")
async def get_tokens(token_data: dict = Depends(get_access_token)):
    client = None
    try:
        client = await get_xrpl_client()
        account_lines_request = AccountLines(account=token_data["account"], ledger_index="validated")
        response = await asyncio.wait_for(client.request(account_lines_request), timeout=30)
        if not response.is_successful():
            raise Exception(response.result.get("error_message", "Failed to fetch account lines"))
        tokens = []
        for line in response.result.get("lines", []):
            if float(line["limit"]) > 0:
                tokens.append({
                    "name": decode_hex_currency(line["currency"]),
                    "issuer": line["account"],
                    "currency": line["currency"]
                })
        logger.info(f"Token holdings for {token_data['account']}: {tokens}")
        return {"tokens": tokens}
    except Exception as e:
        logger.error(f"Token fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch token holdings")
    finally:
        if client and client.is_open():
            await client.close()

# Check balance
@app.get("/balance")
async def balance(
    issuer: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    token_data: dict = Depends(get_access_token)
):
    logger.info(f"Checking balance for account: {token_data['account']}, issuer: {issuer}, currency: {currency}")
    client = None
    try:
        client = await get_xrpl_client()
        account_info_request = AccountInfo(account=token_data["account"], ledger_index="validated")
        response = await asyncio.wait_for(client.request(account_info_request), timeout=30)
        if not response.is_successful():
            raise Exception(response.result.get("error_message", "Account not found"))
        balance_xrp = float(response.result["account_data"]["Balance"]) / 1_000_000
        
        if issuer and currency:
            account_lines_request = AccountLines(account=token_data["account"], ledger_index="validated")
            lines_response = await asyncio.wait_for(client.request(account_lines_request), timeout=30)
            if not lines_response.is_successful():
                raise Exception(lines_response.result.get("error_message", "Failed to fetch account lines"))
            balance_token = 0
            for line in lines_response.result.get("lines", []):
                if line["account"] == issuer and line["currency"] == currency:
                    balance_token = float(line["balance"])
                    break
            return {
                "account": token_data["account"],
                "balance_token": balance_token,
                "currency": currency
            }
        return {
            "account": token_data["account"],
            "balance_xrp": balance_xrp
        }
    except Exception as e:
        logger.error(f"Balance error: {str(e)}")
        raise HTTPException(status_code=404, detail="Failed to fetch balance")
    finally:
        if client and client.is_open():
            await client.close()

# Validate wallets
@app.post("/validate-wallets")
async def validate_wallets(
    wallets: List[Wallet],
    token_type: str = Query(...),
    issuer: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    token_data: dict = Depends(get_access_token)
):
    logger.info(f"Validating wallets: {wallets}, token_type: {token_type}, issuer: {issuer}, currency: {currency}")
    client = None
    try:
        client = await get_xrpl_client()
        results = []
        for wallet in wallets:
            wallet.address = wallet.address.strip()
            try:
                account_info_request = AccountInfo(account=wallet.address, ledger_index="validated")
                response = await asyncio.wait_for(client.request(account_info_request), timeout=30)
                if response.is_successful():
                    results.append({
                        "address": wallet.address,
                        "status": "Valid",
                        "error": None
                    })
                else:
                    error_message = response.result.get("error_message", "Unknown error")
                    results.append({
                        "address": wallet.address,
                        "status": "Invalid",
                        "error": error_message
                    })
            except Exception as e:
                results.append({
                    "address": wallet.address,
                    "status": "Invalid",
                    "error": str(e)
                })
        return results
    except Exception as e:
        logger.error(f"Wallet validation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to validate wallets")
    finally:
        if client and client.is_open():
            await client.close()

# Modified airdrop endpoint
@app.post("/airdrop")
async def airdrop(
    request: AirdropRequest,
    token_data: dict = Depends(get_access_token),
    xpmarket_mode: bool = Query(False)
):
    logger.info(f"Initiating airdrop with payload: {request.model_dump()}, xpmarket_mode: {xpmarket_mode}")
    account = request.account
    if not account:
        raise HTTPException(status_code=400, detail="Account is required")
    
    # If using NFT holders, fetch them
    wallets = request.wallets
    if request.use_nft_holders:
        nft_issuer = request.nft_issuer or DEFAULT_NFT_ISSUER
        holders = await fetch_nft_holders(nft_issuer)
        total_nfts = sum(holders.values())
        if total_nfts == 0:
            raise HTTPException(status_code=400, detail="No NFT holders found for the specified issuer")
        
        # Distribute total_amount proportionally based on NFT count
        wallets = []
        for address, nft_count in holders.items():
            if nft_count > 0:
                amount = (nft_count / total_nfts) * request.total_amount
                wallets.append(Wallet(address=address, amount=round(amount, 6), nft_count=nft_count))
    
    try:
        total_wallet_amount = round(sum(float(wallet.amount or 0) for wallet in wallets), 6)
        request_total_amount = round(float(request.total_amount or 0), 6)
        if abs(total_wallet_amount - request_total_amount) > 0.000001:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Total amount does not match sum of wallet amounts",
                    "request_total_amount": request_total_amount,
                    "calculated_wallet_amount": total_wallet_amount
                }
            )
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid amount data: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid amount data: {str(e)}")

    if not wallets:
        raise HTTPException(status_code=422, detail="At least one wallet is required")

    if request.token_type != "XRP" and (not request.issuer or not request.currency):
        raise HTTPException(status_code=400, detail="Issuer and currency required for token airdrops")

    client = None
    try:
        client = await get_xrpl_client()
        sequence_response = await asyncio.wait_for(client.request(AccountInfo(account=account)), timeout=30)
        if not sequence_response.is_successful():
            raise Exception(f"Failed to fetch account info: {sequence_response.result.get('error_message')}")
        sequence_int = int(sequence_response.result["account_data"]["Sequence"])

        last_ledger_response = await asyncio.wait_for(client.request(Ledger(ledger_index="validated")), timeout=30)
        if not last_ledger_response.is_successful():
            raise Exception(f"Failed to fetch ledger: {last_ledger_response.result.get('error_message')}")
        last_ledger = int(last_ledger_response.result["ledger"]["ledger_index"])
        last_ledger_sequence = last_ledger + 100

        fee = await get_current_fee(client)
        service_fee = min(len(wallets) * 0.05, 5.0)
        total_network_fee = 0
        transactions = []
        fee_transaction = None
        headers = {
            "X-API-Key": XAMAN_API_KEY,
            "X-API-Secret": XAMAN_API_SECRET,
            "Authorization": f"Bearer {token_data['token']}"
        }

        # Create fee transaction
        if service_fee > 0:
            fee_amount = xrp_to_drops(service_fee)
            fee_tx = {
                "TransactionType": "Payment",
                "Account": account,
                "Destination": FEE_WALLET_ADDRESS,
                "Amount": str(fee_amount),
                "Fee": str(fee),
                "Sequence": int(sequence_int),
                "LastLedgerSequence": int(last_ledger_sequence)
            }
            payload = {"txjson": fee_tx}
            response = requests.post("https://xumm.app/api/v1/platform/payload", headers=headers, json=payload)
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Failed to create fee payment payload: {response.text}")
            payload_data = response.json()
            fee_transaction = {
                "payload_uuid": payload_data["uuid"],
                "sign_url": payload_data["next"]["always"]
            }
            total_network_fee += float(fee) / 1_000_000
            sequence_int += 1

        # Create airdrop transactions
        for wallet in wallets:
            wallet.address = wallet.address.strip()
            if float(wallet.amount or 0) <= 0:
                transactions.append({
                    "status": {
                        "address": wallet.address,
                        "status": "Skipped",
                        "error": "Amount is zero"
                    }
                })
                continue
            if request.token_type != "XRP":
                if not xpmarket_mode:
                    trustline_request = GenericRequest(
                        command="account_lines",
                        account=wallet.address,
                        ledger_index="validated"
                    )
                    trustline_response = await asyncio.wait_for(client.request(trustline_request), timeout=30)
                    trustlines = trustline_response.result.get("lines", [])
                    trustline_exists = any(
                        line["account"] == request.issuer and line["currency"] == request.currency
                        for line in trustlines
                    )
                    if not trustline_exists:
                        transactions.append({
                            "status": {
                                "address": wallet.address,
                                "status": "Failed",
                                "error": "No trustline exists for this token"
                            }
                        })
                        continue
                else:
                    account_info_request = AccountInfo(account=wallet.address, ledger_index="validated")
                    account_response = await asyncio.wait_for(client.request(account_info_request), timeout=30)
                    if not account_response.is_successful():
                        transactions.append({
                            "status": {
                                "address": wallet.address,
                                "status": "Failed",
                                "error": "Account not found or not funded"
                            }
                        })
                        continue
                    xrp_balance = float(account_response.result["account_data"]["Balance"]) / 1_000_000
                    if xrp_balance < TRUSTLINE_RESERVE_XRP:
                        transactions.append({
                            "status": {
                                "address": wallet.address,
                                "status": "Failed",
                                "error": f"Insufficient XRP balance ({xrp_balance} XRP)"
                            }
                        })
                        continue
                payment_tx = {
                    "TransactionType": "Payment",
                    "Account": account,
                    "Destination": wallet.address,
                    "Amount": {
                        "currency": request.currency,
                        "value": str(float(wallet.amount)),
                        "issuer": request.issuer
                    },
                    "Fee": str(fee),
                    "Sequence": int(sequence_int),
                    "LastLedgerSequence": int(last_ledger_sequence)
                }
                payload = {"txjson": payment_tx}
                response = requests.post("https://xumm.app/api/v1/platform/payload", headers=headers, json=payload)
                if response.status_code != 200:
                    transactions.append({
                        "status": {
                            "address": wallet.address,
                            "status": "Failed",
                            "error": f"Failed to create payment payload: {response.text}"
                        }
                    })
                    continue
                payload_data = response.json()
                transactions.append({
                    "payload_uuid": payload_data["uuid"],
                    "sign_url": payload_data["next"]["always"],
                    "status": {
                        "address": wallet.address,
                        "status": "Pending Payment"
                    }
                })
                total_network_fee += float(fee) / 1_000_000
                sequence_int += 1
            else:
                amount = xrp_to_drops(float(wallet.amount or 0))
                payment_tx = {
                    "TransactionType": "Payment",
                    "Account": account,
                    "Destination": wallet.address,
                    "Amount": str(amount),
                    "Fee": str(fee),
                    "Sequence": int(sequence_int),
                    "LastLedgerSequence": int(last_ledger_sequence)
                }
                payload = {"txjson": payment_tx}
                response = requests.post("https://xumm.app/api/v1/platform/payload", headers=headers, json=payload)
                if response.status_code != 200:
                    transactions.append({
                        "status": {
                            "address": wallet.address,
                            "status": "Failed",
                            "error": f"Failed to create payment payload: {response.text}"
                        }
                    })
                    continue
                payload_data = response.json()
                transactions.append({
                    "payload_uuid": payload_data["uuid"],
                    "sign_url": payload_data["next"]["always"],
                    "status": {
                        "address": wallet.address,
                        "status": "Pending Payment"
                    }
                })
                total_network_fee += float(fee) / 1_000_000
                sequence_int += 1
    except Exception as e:
        logger.error(f"Airdrop error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process airdrop")
    finally:
        if client and client.is_open():
            await client.close()
    response_content = {
        "transactions": transactions,
        "total_fee": total_network_fee,
        "service_fee": service_fee
    }
    if fee_transaction:
        response_content["fee_transaction"] = fee_transaction
    return JSONResponse(content=response_content)

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
