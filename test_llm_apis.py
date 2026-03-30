#!/usr/bin/env python3
"""
LLM API Tester
==============
Tests Anthropic, Gemini, and OpenRouter APIs independently.
Reads keys from .env in the same directory.

Usage:
    python test_llm_apis.py                            # test all configured providers
    python test_llm_apis.py anthropic                  # test one provider
    python test_llm_apis.py --model deepseek/...       # test specific OpenRouter model
    python test_llm_apis.py --list                     # list all providers' models
    python test_llm_apis.py --list anthropic           # list Anthropic models only
    python test_llm_apis.py --list gemini              # list Gemini models only
    python test_llm_apis.py --list openrouter          # list OpenRouter models
    python test_llm_apis.py --list openrouter --free   # free OpenRouter models only
    python test_llm_apis.py --list llama               # search OpenRouter by keyword
    python test_llm_apis.py --pick                     # interactive: list → type number → test
    python test_llm_apis.py --pick llama --free        # pick from filtered list
    python test_llm_apis.py --scan anthropic           # test all known Anthropic model names
    python test_llm_apis.py --scan gemini              # test all known Gemini model names
    python test_llm_apis.py --prompt "Explain MPI."    # custom test prompt
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

DEFAULT_PROMPT = "Say hello in one sentence."

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
    "gemini-2.5-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


# ── List helpers (all providers) ─────────────────────────────────────────────

def list_anthropic_models():
    """Fetch and print available Anthropic models from the API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(f"{RED}ANTHROPIC_API_KEY not set in .env{RESET}")
        return
    import httpx
    print(f"{BOLD}Fetching Anthropic model list...{RESET}", flush=True)
    resp = httpx.get(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        timeout=15,
    )
    if not resp.is_success:
        print(f"{RED}Failed: HTTP {resp.status_code}: {resp.text[:200]}{RESET}")
        return
    models = resp.json().get("data", [])
    print(f"\n{BOLD}{'MODEL ID':<45} DISPLAY NAME{RESET}")
    print("─" * 80)
    for m in models:
        print(f"  {CYAN}{m['id']:<43}{RESET}  {DIM}{m.get('display_name', '')}{RESET}")
    print(f"\n{len(models)} models.")


def list_gemini_models():
    """Fetch and print available Gemini models from the Google AI API."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(f"{RED}GEMINI_API_KEY not set in .env{RESET}")
        return
    import httpx
    print(f"{BOLD}Fetching Gemini model list...{RESET}", flush=True)
    resp = httpx.get(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
        timeout=15,
    )
    if not resp.is_success:
        print(f"{RED}Failed: HTTP {resp.status_code}: {resp.text[:200]}{RESET}")
        return
    models = resp.json().get("models", [])
    # Only show generative models (skip embedding, etc.)
    models = [m for m in models if "generateContent" in m.get("supportedGenerationMethods", [])]
    print(f"\n{BOLD}{'MODEL ID':<50} DISPLAY NAME{RESET}")
    print("─" * 90)
    for m in models:
        mid = m["name"].replace("models/", "")  # strip "models/" prefix
        print(f"  {CYAN}{mid:<48}{RESET}  {DIM}{m.get('displayName', '')}{RESET}")
    print(f"\n{len(models)} generative models.")


# ── OpenRouter helpers ────────────────────────────────────────────────────────

def _fetch_openrouter_models(search: str = "", free_only: bool = False) -> list:
    """Fetch and filter OpenRouter model list. Returns list of model dicts."""
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

    if free_only:
        models = [m for m in models if m["id"].endswith(":free")]
    if search:
        s = search.lower()
        models = [m for m in models if s in m["id"].lower() or s in m.get("name", "").lower()]

    models.sort(key=lambda m: (not m["id"].endswith(":free"), m["id"]))
    return models


def _print_model_table(models: list, numbered: bool = False):
    """Print a formatted model table. If numbered=True, prefix each row with an index."""
    num_col = "  #  " if numbered else "  "
    print(f"\n{BOLD}{num_col}{'MODEL ID':<53} {'NAME':<30} {'CTX':>7}  PRICE/1M in{RESET}")
    print("─" * (110 + (5 if numbered else 0)))

    for i, m in enumerate(models):
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

        ctx_str  = f"{ctx // 1000}k" if ctx >= 1000 else str(ctx)
        free_tag = f" {GREEN}[FREE]{RESET}" if is_free else ""
        num_str  = f"{BOLD}{i+1:>3}.{RESET} " if numbered else "  "
        print(f"{num_str}{CYAN}{mid:<53}{RESET}{free_tag:<6}  {DIM}{name:<30}{RESET}  {ctx_str:>6}   {price_str}")


def list_openrouter_models(search: str = "", free_only: bool = False):
    """Print a table of available OpenRouter models."""
    models = _fetch_openrouter_models(search, free_only)
    _print_model_table(models, numbered=False)
    print(f"\n{len(models)} models shown.")
    if not free_only and not search:
        print(f"  Filter: {BOLD}--free{RESET}  or  {BOLD}--list llama{RESET}  to narrow results")
    print(f"\nTo test a model:")
    print(f"  {BOLD}python test_llm_apis.py --model <model-id>{RESET}")
    print(f"Or use interactive picker:")
    print(f"  {BOLD}python test_llm_apis.py --pick{RESET}")


def pick_and_test_openrouter(search: str = "", free_only: bool = False, prompt: str = DEFAULT_PROMPT):
    """Interactive: show numbered list, user picks a number, run the test."""
    models = _fetch_openrouter_models(search, free_only)
    if not models:
        print(f"{YELLOW}No models match your filter.{RESET}")
        return

    _print_model_table(models, numbered=True)
    print(f"\n{len(models)} models shown.  (prompt: \"{prompt}\")\n")

    while True:
        try:
            raw = input(f"Enter number to test (1–{len(models)}) or {BOLD}q{RESET} to quit: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if raw.lower() in ("q", ""):
            return

        try:
            idx = int(raw) - 1
            if not (0 <= idx < len(models)):
                raise ValueError
        except ValueError:
            print(f"  {YELLOW}Enter a number between 1 and {len(models)}{RESET}")
            continue

        model_id = models[idx]["id"]
        print()
        run("openrouter", test_openrouter, model_override=model_id, prompt=prompt)
        print()


# ── Scan all known models ─────────────────────────────────────────────────────

def scan_models(provider: str, prompt: str = DEFAULT_PROMPT):
    """Test every model in the known list for a provider."""
    if provider == "anthropic":
        model_list = ANTHROPIC_MODELS
        test_fn    = test_anthropic
    elif provider == "gemini":
        model_list = GEMINI_MODELS
        test_fn    = test_gemini
    else:
        print(f"{RED}--scan supports: anthropic, gemini{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}Scanning {provider} models{RESET}  (prompt: \"{prompt}\")\n")
    for model in model_list:
        run(f"{provider} / {model}", test_fn, model_override=model, prompt=prompt)
    print()


# ── Provider tests ────────────────────────────────────────────────────────────

def test_anthropic(model_override=None, prompt=DEFAULT_PROMPT):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    model   = model_override or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")
    if not api_key:
        return None, "ANTHROPIC_API_KEY not set — skipped", model

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    t0 = time.time()
    msg = client.messages.create(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - t0
    return elapsed, msg.content[0].text.strip(), model


def test_gemini(model_override=None, prompt=DEFAULT_PROMPT):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model   = model_override or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-001")
    if not api_key:
        return None, "GEMINI_API_KEY not set — skipped", model

    from google import genai
    from google.genai import types
    gc = genai.Client(api_key=api_key)
    t0 = time.time()
    response = gc.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=64),
    )
    elapsed = time.time() - t0
    return elapsed, response.text.strip(), model


def test_openrouter(model_override=None, prompt=DEFAULT_PROMPT):
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model   = model_override or os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
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
            "messages": [{"role": "user", "content": prompt}],
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

def run(name: str, fn, model_override=None, prompt=DEFAULT_PROMPT):
    print(f"Testing {BOLD}{name}{RESET}...", end=" ", flush=True)
    try:
        elapsed, result, model = fn(model_override=model_override, prompt=prompt)
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

    # ── --help ───────────────────────────────────────────────────────────────
    if not args or "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    # ── --prompt "..." ───────────────────────────────────────────────────────
    prompt = DEFAULT_PROMPT
    if "--prompt" in args:
        idx = args.index("--prompt")
        if idx + 1 >= len(args):
            print(f"{RED}--prompt requires a string{RESET}")
            sys.exit(1)
        prompt = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    # ── --list mode ──────────────────────────────────────────────────────────
    if "--list" in args:
        args = [a for a in args if a != "--list"]
        free_only = "--free" in args
        args = [a for a in args if a != "--free"]
        # First non-flag arg: provider name or search term
        first = args[0].lower() if args else ""
        if first == "anthropic":
            list_anthropic_models()
        elif first == "gemini":
            list_gemini_models()
        elif first == "openrouter":
            list_openrouter_models(search="", free_only=free_only)
        elif first == "all" or first == "":
            list_anthropic_models()
            print()
            list_gemini_models()
            print()
            list_openrouter_models(search="", free_only=free_only)
        else:
            # treat as search term for OpenRouter
            list_openrouter_models(search=first, free_only=free_only)
        return

    # ── --pick mode ──────────────────────────────────────────────────────────
    if "--pick" in args:
        args = [a for a in args if a != "--pick"]
        free_only = "--free" in args
        args = [a for a in args if a != "--free"]
        search = args[0] if args else ""
        pick_and_test_openrouter(search=search, free_only=free_only, prompt=prompt)
        return

    # ── --scan provider ──────────────────────────────────────────────────────
    if "--scan" in args:
        idx = args.index("--scan")
        if idx + 1 >= len(args):
            print(f"{RED}--scan requires a provider name: anthropic or gemini{RESET}")
            sys.exit(1)
        provider = args[idx + 1]
        scan_models(provider, prompt=prompt)
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
        if not args:
            args = ["openrouter"]

    # ── normal test mode ─────────────────────────────────────────────────────
    filter_names = [a.lower() for a in args]
    to_run = {k: v for k, v in PROVIDERS.items() if not filter_names or k in filter_names}

    if not to_run:
        print(f"Unknown provider(s): {args}. Choose from: {list(PROVIDERS)}")
        sys.exit(1)

    print(f"\n{BOLD}LLM API Test{RESET}  (prompt: \"{prompt}\")\n")
    results = {name: run(name, fn, model_override=model_override, prompt=prompt)
               for name, fn in to_run.items()}
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
