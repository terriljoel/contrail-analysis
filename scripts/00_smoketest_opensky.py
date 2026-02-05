import os
import requests
from dotenv import load_dotenv

TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
API_URL = "https://opensky-network.org/api/states/all"

# Germany bounding box
BBOX = {"lamin": 47.2, "lomin": 5.9, "lamax": 55.1, "lomax": 15.3}

def get_token(client_id: str, client_secret: str) -> str:
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in token response: {r.text[:300]}")
    return token

def main():
    load_dotenv() 
    '''
    load_dotenv() is a function from the python-dotenv package that reads a file named .env (usually in your project root) and loads the key/value pairs into your process environment, so os.getenv() can read them
    '''
    cid = os.getenv("OPENSKY_CLIENT_ID")
    csec = os.getenv("OPENSKY_CLIENT_SECRET")
    if not cid or not csec:
        raise RuntimeError("Missing OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET in .env")

    token = get_token(cid, csec)
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.get(API_URL, params={**BBOX, "extended": 1}, headers=headers, timeout=30)
    r.raise_for_status()
    payload = r.json()

    states = payload.get("states") or []
    print("Snapshot time (unix):", payload.get("time"))
    print("Number of aircraft in bbox:", len(states))

    if states:
        # Print one example row (raw)
        print("Example state vector:", states[0])

if __name__ == "__main__":
    main()
