"""One function, any model. Swap providers with env vars — no code changes.

Gemini, Groq, OpenAI, OpenRouter and local Ollama all expose an OpenAI-compatible
/chat/completions endpoint, so a single adapter covers them all.

  LLM_PROVIDER=gemini  GEMINI_API_KEY=...   (free: aistudio.google.com)
  LLM_PROVIDER=groq    GROQ_API_KEY=...     (free: console.groq.com)
  LLM_PROVIDER=proxy   LLM_API_KEY=...  LLM_BASE_URL=http://localhost:8317/v1   (any OpenAI-compatible API)
  LLM_PROVIDER=mock                          (offline, for testing the pipeline)
  LLM_MODEL=...  overrides the default model name for the chosen provider.
"""
import hashlib, json, os, re, time
from pathlib import Path
import requests

PROVIDERS = {
    # Model names change often: check the provider's console and set LLM_MODEL if these 404.
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "groq":   ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    "ollama": ("http://localhost:11434/v1", "", "llama3.1"),
    # Any other OpenAI-compatible endpoint: a local proxy (e.g. EasyCLIProxyAPI on :8317),
    # or an API someone hands you. Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL.
    "proxy": (os.getenv("LLM_BASE_URL", "http://localhost:8317/v1"), "LLM_API_KEY", "gemini-2.5-flash"),
}

CACHE = Path(os.getenv("LLM_CACHE", ".llm_cache"))
MIN_GAP = float(os.getenv("LLM_MIN_GAP_SEC", "4"))  # stay under free-tier rate limits
_last_call = [0.0]

# Usage counters, written to outputs/run_stats.json so cost at scale can be estimated.
# est_* (chars / 4) are counted on every call, cached or not; *_tokens come from the API when it reports them.
STATS = {"calls": 0, "cache_hits": 0, "api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
         "est_prompt_tokens": 0, "est_completion_tokens": 0}


def model_name() -> str:
    p = os.getenv("LLM_PROVIDER", "gemini")
    return "mock" if p == "mock" else os.getenv("LLM_MODEL", PROVIDERS[p][2])


def config_error():
    """A clear message if the model settings can't work, else None. Checked once before a run."""
    p = os.getenv("LLM_PROVIDER", "gemini")
    if p == "mock":
        return None
    if p not in PROVIDERS:
        return f"LLM_PROVIDER='{p}' is not known. Use one of: mock, {', '.join(PROVIDERS)}."
    key_env = PROVIDERS[p][1]
    if key_env and not os.getenv(key_env):
        return f"LLM_PROVIDER={p} needs an API key: set {key_env} (e.g. export {key_env}=your_key)."
    return None


def extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in ``` fences or add chatter; pull out the object."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def _count_out(data: dict) -> dict:
    STATS["est_completion_tokens"] += len(json.dumps(data)) // 4
    return data


def call_llm(system: str, user: str, temperature: float = 0.3, retries: int = 3) -> dict:
    provider = os.getenv("LLM_PROVIDER", "gemini")
    STATS["calls"] += 1
    STATS["est_prompt_tokens"] += (len(system) + len(user)) // 4
    if provider == "mock":
        from mock_llm import mock_response
        return _count_out(mock_response(user))

    base, key_env, _ = PROVIDERS[provider]
    model = model_name()
    key = hashlib.sha256(f"{model}|{temperature}|{system}|{user}".encode()).hexdigest()[:24]
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"{key}.json"
    if cached.exists():                       # reruns are free and deterministic
        STATS["cache_hits"] += 1
        return _count_out(json.loads(cached.read_text()))

    headers = {"Content-Type": "application/json"}
    if key_env:
        headers["Authorization"] = f"Bearer {os.environ[key_env]}"
    body = {"model": model, "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"}}

    for attempt in range(retries):
        wait = MIN_GAP - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            r = requests.post(f"{base}/chat/completions", headers=headers, json=body, timeout=120)
            if r.status_code == 429:          # rate limited: back off and retry
                time.sleep(20 * (attempt + 1))
                continue
            r.raise_for_status()
            payload = r.json()
            data = extract_json(payload["choices"][0]["message"]["content"])
            usage = payload.get("usage") or {}
            STATS["api_calls"] += 1
            STATS["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
            STATS["completion_tokens"] += usage.get("completion_tokens", 0) or 0
            cached.write_text(json.dumps(data))
            return _count_out(data)
        except (requests.RequestException, ValueError, KeyError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"LLM call failed after {retries} tries: {e}")
            time.sleep(5)
    raise RuntimeError("LLM call failed: rate limited")
