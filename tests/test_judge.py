"""Tests for LLM-as-judge (mocked Ollama)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from traceforge.judge import JudgeClient


def make_judge_response(passed=True, reasoning="Looks good", score=0.9):
    msg = MagicMock()
    msg.content = json.dumps({"passed": passed, "reasoning": reasoning, "score": score})
    resp = MagicMock()
    resp.message = msg
    return resp


class TestJudgeClient:
    @patch("traceforge.judge.ollama_client.Client")
    def test_pass(self, mock_cls):
        mock_client = MagicMock()
        mock_client.chat.return_value = make_judge_response(True, "Correct", 1.0)
        mock_cls.return_value = mock_client

        judge = JudgeClient()
        result = judge.judge(
            criterion="Response says 42",
            user_message="What is 6*7?",
            assistant_response="The answer is 42.",
        )
        assert result["passed"] is True
        assert result["score"] == 1.0

    @patch("traceforge.judge.ollama_client.Client")
    def test_fail(self, mock_cls):
        mock_client = MagicMock()
        mock_client.chat.return_value = make_judge_response(False, "Wrong", 0.1)
        mock_cls.return_value = mock_client

        judge = JudgeClient()
        result = judge.judge(
            criterion="Response says 42",
            user_message="What is 6*7?",
            assistant_response="I don't know.",
        )
        assert result["passed"] is False

    @patch("traceforge.judge.ollama_client.Client")
    def test_with_tool_calls(self, mock_cls):
        mock_client = MagicMock()
        mock_client.chat.return_value = make_judge_response(True, "Good", 0.95)
        mock_cls.return_value = mock_client

        judge = JudgeClient()
        result = judge.judge(
            criterion="Used calculator correctly",
            user_message="What is 6*7?",
            assistant_response="42",
            tool_calls=[{"tool": "calc", "args": {"expr": "6*7"}, "response": {"result": 42}}],
        )
        assert result["passed"] is True

    @patch("traceforge.judge.ollama_client.Client")
    def test_invalid_json_response(self, mock_cls):
        mock_client = MagicMock()
        msg = MagicMock()
        msg.content = "not valid json"
        resp = MagicMock()
        resp.message = msg
        mock_client.chat.return_value = resp
        mock_cls.return_value = mock_client

        judge = JudgeClient()
        result = judge.judge(
            criterion="anything",
            user_message="hi",
            assistant_response="hello",
        )
        assert result["passed"] is False
        assert "valid JSON" in result["reasoning"]

    @patch("traceforge.judge.ollama_client.Client")
    def test_exception_handling(self, mock_cls):
        mock_client = MagicMock()
        mock_client.chat.side_effect = Exception("Connection failed")
        mock_cls.return_value = mock_client

        judge = JudgeClient()
        result = judge.judge(
            criterion="anything",
            user_message="hi",
            assistant_response="hello",
        )
        assert result["passed"] is False
        assert "error" in result["reasoning"].lower()
