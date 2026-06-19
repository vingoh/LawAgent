import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from llm import client as llm_client


def test_chat_json_parses_response():
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"key": "value"}'))
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch.object(llm_client, "_get_client", return_value=mock_client):
        result = llm_client.chat_json("system", "user")

    assert result == {"key": "value"}
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_chat_json_parses_markdown_fenced_response():
    fenced = '```json\n{"key": "value"}\n```'
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=fenced))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch.object(llm_client, "_get_client", return_value=mock_client):
        result = llm_client.chat_json("system", "user")

    assert result == {"key": "value"}


def test_get_client_raises_when_env_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_ID", raising=False)
    llm_client._client = None

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm_client._get_client()
