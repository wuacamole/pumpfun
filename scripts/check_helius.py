"""Quick connectivity check against the configured Helius endpoint.

Run: uv run python scripts/check_helius.py
Verifies: env loading, RPC reachability, getHealth, getAccountInfo decoding path.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from helius import HeliusClient  # noqa: E402


async def main() -> int:
    key = os.environ.get("HELIUS_API_KEY")
    if not key:
        print("FAIL: HELIUS_API_KEY not set")
        return 1
    print(f"HELIUS_API_KEY loaded (len={len(key)})")

    async with HeliusClient() as client:
        print("endpoint: https://mainnet.helius-rpc.com/?api-key=<redacted>")
        # 1. Health check — cheapest call.
        try:
            res = await client._rpc("getHealth", [])
            print(f"getHealth -> {res}")
        except Exception as e:
            print(f"FAIL: getHealth error: {e}")
            return 2

        # 2. Slot — sanity that the RPC returns numbers.
        try:
            slot = await client._rpc("getSlot", [])
            print(f"getSlot  -> {slot}")
        except Exception as e:
            print(f"FAIL: getSlot error: {e}")
            return 3

        # 3. A parsed-transaction call on a dummy signature should return None
        #    (not raise), so we know classify_via_frontend degrades gracefully.
        try:
            v = await client.classify_via_frontend("1" * 88)
            print(f"classify_via_frontend(dummy) -> {v}  (None = expected: not found)")
        except Exception as e:
            print(f"FAIL: classify_via_frontend raised on missing tx: {e}")
            return 4

    print("\nHelius connection OK.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))