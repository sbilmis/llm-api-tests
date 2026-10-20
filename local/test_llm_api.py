#!/usr/bin/env python3
"""
Quick test script for a local LLM API (OpenAI-compatible endpoint).
Usage from the repository root: python3 local/test_llm_api.py
"""

import os
import json
import urllib.request
import urllib.error

# Load .env if present
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
API_KEY  = os.getenv("LLM_API_KEY",  "your-api-key-here")
MODEL    = os.getenv("LLM_MODEL",    "")   # leave empty to auto-detect
PROMPT   = "Say hello and tell me your model name in one sentence."
# ─────────────────────────────────────────────────────────────────────────────


def probe(url: str, key: str):
    """GET a URL and return (status_code, body_snippet)."""
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode(errors="replace")[:300]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]
    except urllib.error.URLError as e:
        return None, str(e.reason)
    except Exception as e:
        return None, str(e)


def list_models(base: str, key: str) -> list[str]:
    """Try several common model-list endpoints; print debug for each."""
    root = base.rstrip("/")
    root_no_v1 = root[:-3] if root.endswith("/v1") else root

    candidates = [
        f"{root}/models",
        f"{root_no_v1}/models",
        f"{root_no_v1}/api/models",
        f"{root_no_v1}/api/tags",
        f"{root_no_v1}/",            # root index — sometimes reveals routes
    ]

    print("  Probing endpoints:")
    for url in candidates:
        status, body = probe(url, key)
        print(f"    {url}")
        print(f"      → status={status}  body={body!r}")

        if status == 200:
            try:
                # re-fetch full body (probe only returns a 300-char snippet)
                req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())

                if "data" in data:
                    ids = [m["id"] for m in data["data"] if m.get("id")]
                elif "models" in data:
                    ids = [m.get("name") or m.get("id", "") for m in data["models"]]
                else:
                    ids = list(data.keys())   # show top-level keys as hint

                if ids:
                    return ids
            except Exception as ex:
                print(f"      parse error: {ex}")

    return []


def chat(base: str, key: str, model: str, prompt: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
    }).encode()

    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def main():
    print(f"\n{'─'*60}")
    print(f"  Local LLM API Test")
    print(f"{'─'*60}")
    print(f"  Base URL : {API_BASE}")
    key_status = "configured" if API_KEY and API_KEY != "your-api-key-here" else "not configured"
    print(f"  API Key  : {key_status}")

    # ── Step 1: discover model ────────────────────────────────────────────────
    print("\n[1] Fetching available models …")
    models = list_models(API_BASE, API_KEY)

    model = MODEL
    if models:
        print(f"\n    Found   : {models}")
        if not model:
            model = models[0]
            print(f"    Selected: {model}")
    else:
        print("\n    None of the probe URLs returned a model list.")
        if not model:
            model = input("    Enter model name manually: ").strip()

    if not model:
        print("ERROR: No model specified.")
        return

    # ── Step 2: send a chat message ───────────────────────────────────────────
    print(f"\n[2] Sending test prompt …")
    print(f"    Model : {model}")
    print(f"    Prompt: {PROMPT}")

    try:
        reply = chat(API_BASE, API_KEY, model, PROMPT)
        print(f"\n{'─'*60}")
        print(f"  ✓  API is working!")
        print(f"{'─'*60}")
        print(f"  Reply:\n\n  {reply}\n")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"\n  ✗  HTTP {e.code}: {e.reason}")
        print(f"     {body[:400]}")
    except urllib.error.URLError as e:
        print(f"\n  ✗  Connection failed: {e.reason}")
    except (KeyError, IndexError) as e:
        print(f"\n  ✗  Unexpected response shape: {e}")


if __name__ == "__main__":
    main()
