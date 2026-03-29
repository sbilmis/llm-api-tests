#!/usr/bin/env python3
"""
LLM API Tester
==============
Tests Anthropic, Gemini, and OpenRouter APIs independently.
Reads keys from .env in the same directory.

Usage:
    python test_llm_apis.py                          # test all configured providers
    python test_llm_apis.py anthropic                # test one provider
    python test_llm_apis.py openrouter               # test with model from .env
    python test_llm_apis.py --model deepseek/deepseek-r1:free   # test specific OR model
    python test_llm_apis.py --list                   # list all available OpenRouter models
    python test_llm_apis.py --list --free            # list only free OpenRouter models
    python test_llm_apis.py --list llama             # list OpenRouter models matching "llama"
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
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

ANTHROPIC_MODELS = [
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
]

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


# ── List OpenRouter models ────────────────────────────────────────────────────

def list_openrouter_models(search: str = "", free_only: bool = False):
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print(f"{RED}OPENROUTER_API_KEY not set in .env{RESET}")
        sys.exit(1)

    import httpx
    print(f"{BOLD}Fetching OpenRouter model list...{RESET}", flush=True)
    resp = httpx.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    if not resp.is_success:
        print(f"{RED}Failed: HTTP {resp.status_code}: {resp.text[:200]}{RESET}")
        sys.exit(1)

    models = resp.json().get("data", [])

    # Filter
    if free_only:
        models = [m for m in models if m["id"].endswith(":free")]
    if search:
        s = search.lower()
        models = [m for m in models if s in m["id"].lower() or s in m.get("name", "").lower()]

    # Sort: free first, then alphabetically
    models.sort(key=lambda m: (not m["id"].endswith(":free"), m["id"]))

    print(f"\n{BOLD}{'MODEL ID':<55} {'NAME':<30} {'CTX':>7}  PRICE/1M in{RESET}")
    print("─" * 110)

    for m in models:
        mid     = m["id"]
        name    = m.get("name", "")[:29]
        ctx     = m.get("context_length", 0)
        price   = m.get("pricing", {}).get("prompt", "0")
        is_free = mid.endswith(":free")

        try:
            price_per_1m = float(price) * 1_000_000
            price_str = f"{GREEN}FREE{RESET}" if is_free else f"${price_per_1m:.3f}"
        except (ValueError, TypeError):
            price_str = DIM + "n/a" + RESET

        ctx_str = f"{ctx // 1000}k" if ctx >= 1000 else str(ctx)
        free_tag = f" {GREEN}[FREE]{RESET}" if is_free else ""
        print(f"  {CYAN}{mid:<53}{RESET}{free_tag:<6}  {DIM}{name:<30}{RESET}  {ctx_str:>6}   {price_str}")

    print(f"\n{len(models)} models shown.")
    if not free_only and not search:
        print(f"  Filter: {BOLD}--free{RESET}  or  {BOLD}--list llama{RESET}  to narrow results")
    print(f"\nTo test a model:")
    print(f"  {BOLD}python test_llm_apis.py --model <model-id>{RESET}")


# ── Provider tests ────────────────────────────────────────────────────────────

def test_anthropic(model_override=None):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    model   = model_override or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")
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
    return elapsed, msg.content[0].text.strip(), model


def test_gemini(model_override=None):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model   = model_override or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    if not api_key:
        return None, "GEMINI_API_KEY not set — skipped", model

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
    return elapsed, response.text.strip(), model


def test_openrouter(model_override=None):
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model   = model_override or os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    if not api_key:
        return None, "OPENROUTER_API_KEY not set — skipped", model

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
    return elapsed, data["choices"][0]["message"]["content"].strip(), model


PROVIDERS = {
    "anthropic":  test_anthropic,
    "gemini":     test_gemini,
    "openrouter": test_openrouter,
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run(name: str, fn, model_override=None):
    print(f"Testing {BOLD}{name}{RESET}...", end=" ", flush=True)
    try:
        elapsed, result, model = fn(model_override)
        if elapsed is None:
            print(f"{YELLOW}⚠ SKIP{RESET}  {result}")
            return "skip"
        snippet = result[:70].replace("\n", " ")
        print(f'{GREEN}✅ PASS{RESET}  ({elapsed:.1f}s)  [{DIM}{model}{RESET}]  "{snippet}"')
        return "pass"
    except Exception as e:
        print(f"{RED}❌ FAIL{RESET}  {type(e).__name__}: {str(e)[:200]}")
        return "fail"


def main():
    args = sys.argv[1:]

    # ── --list mode ──────────────────────────────────────────────────────────
    if "--list" in args:
        args = [a for a in args if a != "--list"]
        free_only = "--free" in args
        args = [a for a in args if a != "--free"]
        search = args[0] if args else ""
        list_openrouter_models(search=search, free_only=free_only)
        return

    # ── --model <id> ─────────────────────────────────────────────────────────
    model_override = None
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 >= len(args):
            print(f"{RED}--model requires a model id{RESET}")
            sys.exit(1)
        model_override = args[idx + 1]
        args = args[:idx] + args[idx + 2:]
        # When --model is given, run only openrouter unless provider specified
        if not args:
            args = ["openrouter"]

    # ── normal test mode ─────────────────────────────────────────────────────
    filter_names = [a.lower() for a in args]
    to_run = {k: v for k, v in PROVIDERS.items() if not filter_names or k in filter_names}

    if not to_run:
        print(f"Unknown provider(s): {args}. Choose from: {list(PROVIDERS)}")
        sys.exit(1)

    print(f"\n{BOLD}LLM API Test{RESET}  (prompt: \"{PROMPT}\")\n")
    results = {name: run(name, fn, model_override) for name, fn in to_run.items()}
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
