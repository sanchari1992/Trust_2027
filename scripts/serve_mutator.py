"""Starts the mutation web page and API on your own computer.

    python scripts/serve_mutator.py                 # then open http://127.0.0.1:8000
    python scripts/serve_mutator.py --port 8080
    python scripts/serve_mutator.py --profile pilot   # a config.yaml profile, like the other scripts

Every request uses your DeepSeek key (about 1-7 LLM calls per mutation), so by default only this computer can
reach it. To let others on your network use it, add --host 0.0.0.0 AND --token SOMETHING; people then enter that
token on the page. API documentation (try it in the browser): http://127.0.0.1:8000/docs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.service.web import create_app  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--token", default=None, help="require this token (header X-Token) for /api/mutate")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost") and not args.token:
        sys.exit("For safety, a --token is required when --host is not this computer (your API key pays for every request).")
    import uvicorn

    from src.config import load_config

    app = create_app(load_config(profile=args.profile), token=args.token)
    print(f"Open http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}   (stop with Ctrl+C)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
