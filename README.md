# LLM API Tester

A single-file Python script that tests Anthropic, Gemini, and OpenRouter APIs independently. Useful for quickly checking which keys are working, exploring available models, and comparing pricing — without spinning up any server.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

## Usage

### Test all configured providers
```bash
python test_llm_apis.py
```
```
LLM API Test  (prompt: "Say hello in one sentence.")

Testing anthropic...   ✅ PASS  (1.2s)  [claude-opus-4-6]  "Hello! How can I help you today?"
Testing gemini...      ✅ PASS  (0.9s)  [gemini-2.0-flash-001]  "Hello there!"
Testing openrouter...  ✅ PASS  (2.1s)  [meta-llama/llama-3.3-70b-instruct]  "Hello! I'm Llama..."

3 passed  (3 providers)
```

### Test a single provider
```bash
python test_llm_apis.py anthropic
python test_llm_apis.py gemini
python test_llm_apis.py openrouter
```

### Custom prompt
```bash
python test_llm_apis.py --prompt "Explain MPI in one sentence."
python test_llm_apis.py anthropic --prompt "What is a transformer?"
```

### Test a specific model
```bash
python test_llm_apis.py --model deepseek/deepseek-r1:free
python test_llm_apis.py --model openai/gpt-4o --prompt "Summarize quantum computing."
```

---

## Listing models

### All providers
```bash
python test_llm_apis.py --list
```

### One provider
```bash
python test_llm_apis.py --list anthropic
python test_llm_apis.py --list gemini
python test_llm_apis.py --list openrouter
```

### Filter OpenRouter models
```bash
python test_llm_apis.py --list openrouter --free    # free models only
python test_llm_apis.py --list llama                 # search by keyword
python test_llm_apis.py --list deepseek --free       # keyword + free filter
```

Pricing is shown for all providers. Anthropic and Gemini prices come from OpenRouter's live catalog (if `OPENROUTER_API_KEY` is set), otherwise fall back to a static table.

---

## Interactive wizard

```bash
python test_llm_apis.py --pick
```

Multi-step interactive flow:

1. Lists configured providers (only those with API keys in `.env`)
2. Pick a provider
3. Fetches that provider's model list — with pricing
4. For OpenRouter: choose All / Free only / Search by keyword
5. Pick a model
6. Runs the test immediately

---

## Scan all known models

Test every model in the built-in list for a provider — useful for finding which model names are currently valid:

```bash
python test_llm_apis.py --scan anthropic
python test_llm_apis.py --scan gemini
```

---

## Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```env
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-v1-...

# Optional — override default models used for basic tests
ANTHROPIC_MODEL=claude-opus-4-6
GEMINI_MODEL=gemini-2.0-flash-001
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct
```

Any key not set causes that provider to be skipped (not an error).

**Where to get keys:**
- Anthropic: [console.anthropic.com](https://console.anthropic.com)
- Gemini: [aistudio.google.com](https://aistudio.google.com) — use "Create API key" here for free quota (not Google Cloud Console)
- OpenRouter: [openrouter.ai/keys](https://openrouter.ai/keys) — many free models available with `:free` suffix

---

## Requirements

- Python 3.9+
- `anthropic`, `google-genai`, `httpx`, `python-dotenv` (see `requirements.txt`)
