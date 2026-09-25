from pathlib import Path

from llm.prompty_loader import load_system_prompt, render_prompt


PROMPTY_PATH = Path(__file__).with_name("llm_only.prompty")
SYSTEM_PROMPT = load_system_prompt(str(PROMPTY_PATH))


def build_llm_only_prompt(question: str) -> list[dict]:
	return [
		{
			"role": "system",
			"content": render_prompt(SYSTEM_PROMPT, question=question),
		}
	]
