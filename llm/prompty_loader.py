from pathlib import Path


def load_prompty_body(file_path: str) -> str:
    text = Path(file_path).read_text(encoding="utf-8")
    if "\n---\n" not in text:
        return text.strip()

    parts = text.split("\n---\n", 2)
    if len(parts) < 3:
        return text.strip()
    return parts[2].strip()


def load_system_prompt(file_path: str) -> str:
    body = load_prompty_body(file_path)
    marker = "system:\n"
    start = body.find(marker)
    if start == -1:
        return body.strip()
    return body[start + len(marker):].strip()


def render_prompt(template: str, **variables: str) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered
