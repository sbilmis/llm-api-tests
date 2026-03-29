#!/usr/bin/env python3
"""
LLM API Tester
==============
Tests Anthropic, Gemini, and OpenRouter APIs independently.
Reads keys from .env in the same directory.

Usage:
    python test_llm_apis.py              # test all configured providers
    python test_llm_apis.py anthropic    # test one provider
    python test_llm_apis.py gemini
    python test_llm_apis.py openrouter
"""

import os
import sys
import time
from pathlib import Path

# Load .env from same directory as this script
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

PROMPT = "Say hello in one sentence."

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def test_anthropic():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    model   = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")
    if not api_key:
        return None, "ANTHROPIC_API_KEY not set — skipped"

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    t0 = time.time()
    msg = client.messages.create(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": PROMPT}],
    )
    elapsed = time.time() - t0
    text = msg.content[0].text.strip()
    return elapsed, text


def test_gemini():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    if not api_key:
        return None, "GEMINI_API_KEY not set — skipped"

    from google import genai
    from google.genai import types
    gc = genai.Client(api_key=api_key)
    t0 = time.time()
    response = gc.models.generate_content(
        model=model,
        contents=PROMPT,
        config=types.GenerateContentConfig(max_output_tokens=64),
    )
    elapsed = time.time() - t0
    return elapsed, response.text.strip()


def test_openrouter():
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model   = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    if not api_key:
        return None, "OPENROUTER_API_KEY not set — skipped"

    import httpx
    t0 = time.time()
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://llm-api-tests.local",
            "X-Title": "LLM API Tester",
        },
        json={
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": PROMPT}],
        },
        timeout=60,
    )
    elapsed = time.time() - t0
    if not resp.is_success:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    if "error" in data:
        raise RuntimeError(str(data["error"])[:300])
    return elapsed, data["choices"][0]["message"]["content"].strip()


PROVIDERS = {
    "anthropic":  (test_anthropic,  lambda: os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")),
    "gemini":     (test_gemini,     lambda: os.environ.get("GEMINI_MODEL",    "gemini-2.0-flash")),
    "openrouter": (test_openrouter, lambda: os.environ.get("OPENROUTER_MODEL","meta-llama/llama-3.3-70b-instruct:free")),
}


def run(name: str, fn, model_fn):
    model = model_fn()
    label = f"{name} ({model})"
    print(f"Testing {BOLD}{label}{RESET}...", end=" ", flush=True)
    try:
        elapsed, result = fn()
        if elapsed is None:
            # skipped (no key)
            print(f"{YELLOW}⚠ SKIP{RESET}  {result}")
            return "skip"
        snippet = result[:70].replace("\n", " ")
        print(f'{GREEN}✅ PASS{RESET}  ({elapsed:.1f}s)  "{snippet}"')
        return "pass"
    except Exception as e:
        print(f"{RED}❌ FAIL{RESET}  {type(e).__name__}: {str(e)[:200]}")
        return "fail"


def main():
    filter_names = [a.lower() for a in sys.argv[1:]]
    to_run = {k: v for k, v in PROVIDERS.items() if not filter_names or k in filter_names}

    if not to_run:
        print(f"Unknown provider(s): {sys.argv[1:]}. Choose from: {list(PROVIDERS)}")
        sys.exit(1)

    print(f"\n{BOLD}LLM API Test{RESET}  (prompt: \"{PROMPT}\")\n")
    results = {name: run(name, fn, mfn) for name, (fn, mfn) in to_run.items()}
    print()

    passed  = sum(1 for r in results.values() if r == "pass")
    failed  = sum(1 for r in results.values() if r == "fail")
    skipped = sum(1 for r in results.values() if r == "skip")
    total   = len(results)

    parts = []
    if passed:  parts.append(f"{GREEN}{passed} passed{RESET}")
    if failed:  parts.append(f"{RED}{failed} failed{RESET}")
    if skipped: parts.append(f"{YELLOW}{skipped} skipped{RESET}")
    print("  ".join(parts) + f"  ({total} providers)")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
