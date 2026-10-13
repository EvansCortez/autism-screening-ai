"""Bounded CI readiness gate: require both the app and saved model, not just a TCP port."""

import argparse
import json
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    last = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=2) as response:
                health = json.load(response)
            if health.get("model_loaded") is True:
                print("FastAPI app and saved model are ready.")
                return
            last = "app responded but model_loaded is not true"
        except (OSError, ValueError, urllib.error.URLError) as error:
            last = str(error)
        time.sleep(0.5)
    raise SystemExit(f"App/model readiness timed out after {args.timeout}s: {last}. Inspect server logs and /health.")


if __name__ == "__main__":
    main()
