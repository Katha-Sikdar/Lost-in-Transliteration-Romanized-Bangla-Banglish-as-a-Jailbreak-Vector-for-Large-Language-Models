"""Uniform single-turn interface over the model providers used in the study.

Every client returns a dict:
    {"text": str, "stop_reason": str | None, "refusal_category": str | None,
     "served_model": str | None}

Provider SDKs are imported lazily, so you only install what you use.

Note on Claude refusal fallbacks: the Anthropic API can re-run a refused
request on a different model ("fallbacks"). It is deliberately NOT enabled
here, because it would silently mix models inside one measurement. An
API-level refusal is recorded as stop_reason="refusal" and scored as R.
"""

import hashlib
import json
import os
import re


class ModelClient:
    def __init__(self, name, model, params=None, **kwargs):
        self.name = name
        self.model = model
        self.params = dict(params or {})
        self.options = kwargs

    def generate(self, prompt, system=None, json_schema=None):
        raise NotImplementedError


class DummyClient(ModelClient):
    """Offline client for dry runs and tests. No network, deterministic."""

    def generate(self, prompt, system=None, json_schema=None):
        if json_schema is not None:
            h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
            choices = json_schema["properties"]["label"]["enum"]
            label = choices[h % len(choices)]
            return {"text": json.dumps({"label": label, "understood": label != "M", "rationale": "dummy"}),
                    "stop_reason": "end_turn", "refusal_category": None, "served_model": "dummy"}
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        text = "I can't help with that." if h % 2 else f"[dummy answer to: {prompt[:40]}]"
        return {"text": text, "stop_reason": "end_turn", "refusal_category": None, "served_model": "dummy"}


class AnthropicClient(ModelClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import anthropic

        self._client = anthropic.Anthropic()

    def generate(self, prompt, system=None, json_schema=None):
        params = dict(self.params)
        params.setdefault("max_tokens", 4096)
        if json_schema is not None:
            output_config = dict(params.pop("output_config", {}))
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
            params["output_config"] = output_config
        if system:
            params["system"] = system
        resp = self._client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **params,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        category = None
        if resp.stop_reason == "refusal" and resp.stop_details:
            category = resp.stop_details.category
        return {"text": text, "stop_reason": resp.stop_reason, "refusal_category": category,
                "served_model": resp.model}


class OpenAIClient(ModelClient):
    """OpenAI API, or any OpenAI-compatible server (e.g. vLLM for open models).

    For an OpenAI-compatible server set `base_url` and optionally
    `api_key_env` in the model config.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import openai

        base_url = self.options.get("base_url")
        api_key = os.environ.get(self.options.get("api_key_env", "OPENAI_API_KEY"), "EMPTY" if base_url else None)
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, prompt, system=None, json_schema=None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        params = dict(self.params)
        if json_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "judgement", "schema": json_schema, "strict": True},
            }
        resp = self._client.chat.completions.create(model=self.model, messages=messages, **params)
        choice = resp.choices[0]
        refusal = getattr(choice.message, "refusal", None)
        text = choice.message.content or refusal or ""
        stop = "refusal" if refusal else choice.finish_reason
        return {"text": text, "stop_reason": stop, "refusal_category": None, "served_model": resp.model}


class GoogleClient(ModelClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from google import genai

        self._client = genai.Client()

    def generate(self, prompt, system=None, json_schema=None):
        config = dict(self.params)
        if system:
            config["system_instruction"] = system
        if json_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = json_schema
        resp = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
        block = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
        if block:
            return {"text": "", "stop_reason": "refusal", "refusal_category": str(block), "served_model": self.model}
        finish = None
        if resp.candidates:
            finish = str(resp.candidates[0].finish_reason)
        text = resp.text or ""
        stop = "refusal" if finish and "SAFETY" in finish and not text else finish
        return {"text": text, "stop_reason": stop, "refusal_category": None, "served_model": self.model}


PROVIDERS = {
    "dummy": DummyClient,
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "openai_compatible": OpenAIClient,
    "google": GoogleClient,
}


def make_client(cfg):
    cfg = dict(cfg)
    provider = cfg.pop("provider")
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
    return PROVIDERS[provider](cfg.pop("name"), cfg.pop("model"), cfg.pop("params", None), **cfg)


def extract_json(text):
    """Parse the first JSON object in a model reply (tolerates code fences)."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError("no JSON object in reply")
    return json.loads(match.group(0))
