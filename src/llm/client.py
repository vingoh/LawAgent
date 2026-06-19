"""OpenAI-compatible LLM client."""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../..")
load_dotenv(os.path.join(ROOT_DIR, ".env"))

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_id = os.getenv("OPENAI_MODEL_ID")

    missing = [
        name
        for name, val in [
            ("OPENAI_API_KEY", api_key),
            ("OPENAI_BASE_URL", base_url),
            ("OPENAI_MODEL_ID", model_id),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def _strip_markdown_json_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_llm_json(content: str) -> dict:
    text = content.strip()
    for candidate in (text, _strip_markdown_json_fence(text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise json.JSONDecodeError("Expecting value", text, 0)


def chat_json(system: str, user: str) -> dict:
    """Call OpenAI-compatible API with JSON response format."""
    client = _get_client()
    model_id = os.environ["OPENAI_MODEL_ID"]
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty content")
    return _parse_llm_json(content)
