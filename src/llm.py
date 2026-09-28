"""The ONLY place that talks to DeepSeek.

Every RAG gets its language model from here. To try a different LLM in the
"different LLM" experiment, change the llm: section of config.yaml.
"""
import os

from lightrag.llm.openai import openai_complete_if_cache


def _settings(cfg):
    c = cfg["llm"]
    api_key = os.getenv(c["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"No API key found. Put {c['api_key_env']}=... in your .env file."
        )
    # DEEPSEEK_BASE_URL is optional; it lets us point at a test server.
    base_url = os.getenv("DEEPSEEK_BASE_URL", c["base_url"])
    extra_body = {"thinking": {"type": c.get("thinking", "disabled")}}
    return c, api_key, base_url, extra_body


def get_llm_func(cfg):
    """Async version, used by LightRAG."""
    c, api_key, base_url, extra_body = _settings(cfg)

    async def llm_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        entity_extraction=False,
        **kwargs,
    ):
        kwargs.setdefault("temperature", c.get("temperature", 0.0))
        return await openai_complete_if_cache(
            c["model"],
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            keyword_extraction=keyword_extraction,
            entity_extraction=entity_extraction,
            base_url=base_url,
            api_key=api_key,
            extra_body=extra_body,
            **kwargs,
        )

    return llm_func


def get_chat_func(cfg):
    """Plain (non-async) version, used by RAPTOR and MedRAG.
    Usage: chat(system_text, user_text, max_tokens=None) -> answer text"""
    from openai import OpenAI
    from tenacity import retry, stop_after_attempt, wait_random_exponential

    c, api_key, base_url, extra_body = _settings(cfg)
    client = OpenAI(api_key=api_key, base_url=base_url)

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def chat(system, user, max_tokens=None):
        kwargs = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = client.chat.completions.create(
            model=c["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=c.get("temperature", 0.0),
            extra_body=extra_body,
            **kwargs,
        )
        return (response.choices[0].message.content or "").strip()

    return chat
