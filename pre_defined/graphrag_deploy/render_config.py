import os
import sys
from pathlib import Path
from string import Template
from dotenv import load_dotenv


load_dotenv(override=True)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: render_config.py TEMPLATE OUTPUT")
    template_path, output_path = map(Path, sys.argv[1:])
    required = {
        "TG_HOST",
        "TG_API_KEY",
        "LLM_API_KEY",
        "LLM_BASE_URL",
    }
    missing = sorted(name for name in required if not os.environ.get(name))
    if missing:
        raise SystemExit("Missing required environment variables: " + ", ".join(missing))
    defaults = {
        "TG_RESTPP_PORT": "443",
        "TG_GS_PORT": "443",
        "PREDEFINED_GRAPH": "WikipediaGraph",
        "LLM_EMBEDDING_MODEL": "text-embedding-3-small",
        "LLM_COMPLETION_MODEL": "gpt-4.1-mini",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    os.environ["LLM_BASE_URL"] = os.getenv("LLM_HOST_URL", os.environ["LLM_BASE_URL"]).rstrip("/")
    rendered = Template(template_path.read_text(encoding="utf-8")).substitute(os.environ)
    output_path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()