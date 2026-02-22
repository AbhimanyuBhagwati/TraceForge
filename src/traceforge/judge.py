"""LLM-as-judge module using Ollama."""

import json

import ollama as ollama_client


class JudgeClient:
    """Uses an LLM to judge agent responses against criteria."""

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        temperature: float = 0.0,
        seed: int | None = 42,
        ollama_host: str = "http://localhost:11434",
    ):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self._client = ollama_client.Client(host=ollama_host)

    def judge(
        self,
        criterion: str,
        user_message: str,
        assistant_response: str,
        tool_calls: list[dict] | None = None,
    ) -> dict:
        """Judge whether the assistant response meets the criterion.

        Returns:
            dict with keys: passed (bool), reasoning (str), score (float 0-1)
        """
        tool_info = ""
        if tool_calls:
            tool_info = f"\n\nTool calls made:\n{json.dumps(tool_calls, indent=2)}"

        prompt = f"""You are an impartial judge evaluating an AI assistant's response.

User message: {user_message}
Assistant response: {assistant_response}{tool_info}

Criterion to evaluate: {criterion}

Respond with a JSON object containing:
- "passed": true if the criterion is met, false otherwise
- "reasoning": brief explanation of your judgment
- "score": a float from 0.0 to 1.0 indicating how well the criterion is met

Respond ONLY with the JSON object, no other text."""

        options = {"temperature": self.temperature}
        if self.seed is not None:
            options["seed"] = self.seed

        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options=options,
            )

            content = response.get("message", response) if isinstance(response, dict) else response.message
            text = content.get("content", "") if isinstance(content, dict) else (content.content or "")

            result = json.loads(text)
            return {
                "passed": bool(result.get("passed", False)),
                "reasoning": str(result.get("reasoning", "")),
                "score": float(result.get("score", 0.0)),
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return {
                "passed": False,
                "reasoning": f"Judge failed to produce valid JSON: {e}",
                "score": 0.0,
            }
        except Exception as e:
            return {
                "passed": False,
                "reasoning": f"Judge error: {e}",
                "score": 0.0,
            }
