"""Prompts for the MTP-head self-distillation data: bring your own.

The prompt lists are read from a JSON file you write (TF_HEAD_PROMPTS, default ``prompts.json`` in the data root, the
working directory); none ship with the tools. Its keys, each optional: ``excluded`` (prompts never trained on, such as
your speed suite's), ``systems`` (system prompts cycled over the plain prompts; null for none), ``chat``, ``advice``,
``prose``, ``code``, ``agent``, ``reasoning`` (lists of user prompts), ``tools`` (OpenAI tool schemas),
``tool_tasks`` (user prompts that call them), ``tool_dialogs`` (message lists ending in a tool result), ``bench``
(id -> prompt, recorded for evaluation only) and ``extra`` (``chat``/``prose``/``advice``/``dialogs``: a second set,
see extra_prompts). The local bakeoff suite (``bakeoff-suite.json``) and markdown docs (``docs/``) in the data root
are added when present. Replies are stored as tokens and never executed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path.cwd()  # the data root: run the tools from it
PROMPTS = Path(os.environ.get("TF_HEAD_PROMPTS", str(ROOT / "prompts.json")))


def _load() -> dict:
    return json.loads(PROMPTS.read_text()) if PROMPTS.is_file() else {}


DATA = _load()
EXCLUDED = set(DATA.get("excluded", []))
SYSTEMS = DATA.get("systems") or [None]
CHAT, ADVICE, PROSE = DATA.get("chat", []), DATA.get("advice", []), DATA.get("prose", [])
CODE, AGENT, REASONING = DATA.get("code", []), DATA.get("agent", []), DATA.get("reasoning", [])
TOOLS = DATA.get("tools", [])
TOOL_TASKS, TOOL_DIALOGS = DATA.get("tool_tasks", []), DATA.get("tool_dialogs", [])
BENCH = DATA.get("bench", {})


def _bakeoff() -> list[dict]:
    out = []
    if not (ROOT / "bakeoff-suite.json").is_file():
        return out
    for item in json.loads((ROOT / "bakeoff-suite.json").read_text()):
        text = item["prompt"]
        files = item.get("files") or {}
        if files:
            parts = [f"--- {name} ---\n{body}" for name, body in files.items()]
            text = text + "\n\nWorkspace files:\n" + "\n\n".join(parts)
        kind = {"coding": "code", "everyday": "reasoning", "agentic": "agent", "tool_use": "agent"}[item["category"]]
        out.append({"id": f"bakeoff-{item['id']}", "kind": kind, "messages": [{"role": "user", "content": text}]})
    return out


def _docs() -> list[dict]:
    """Chunks of local markdown docs: summarize / explain / extract (prose over technical text)."""

    tree = ROOT / "docs"
    files = sorted(tree.rglob("*.md")) + sorted(ROOT.glob("[A-Z]*.md"))
    asks = ["Summarize the following notes for a busy reader in a few paragraphs.",
            "Explain the following document to someone new to the project, in plain language.",
            "List the key decisions and open questions in the following notes."]
    out, i = [], 0
    for f in files:
        text = f.read_text(errors="replace")
        for start in range(0, min(len(text), 12000), 4000):
            chunk = text[start:start + 4000].strip()
            if len(chunk) < 800:
                continue
            out.append({"id": f"doc-{f.stem}-{start}", "kind": "docs",
                        "messages": [{"role": "user", "content": f"{asks[i % 3]}\n\n{chunk}"}]})
            i += 1
    return out


def all_prompts() -> list[dict]:
    items: list[dict] = []
    for kind, lst in (("chat", CHAT), ("advice", ADVICE), ("prose", PROSE), ("code", CODE), ("agent", AGENT),
                      ("reasoning", REASONING)):
        for j, text in enumerate(lst):
            if text in EXCLUDED:
                continue
            sysmsg = SYSTEMS[j % len(SYSTEMS)]
            msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [{"role": "user", "content": text}]
            items.append({"id": f"{kind}-{j:03d}", "kind": kind, "messages": msgs})
    for j, text in enumerate(TOOL_TASKS):
        items.append({"id": f"tool-{j:03d}", "kind": "agent", "tools": True,
                      "messages": [{"role": "user", "content": text}]})
    for j, dialog in enumerate(TOOL_DIALOGS):
        items.append({"id": f"tooldlg-{j:03d}", "kind": "agent", "tools": True, "messages": dialog})
    items += _bakeoff()
    items += _docs()
    return items


def split(pid: str) -> str:
    """10% of prompts held out for evaluation, by a hash of the prompt id (all of a prompt's replies together)."""

    return "eval" if int(hashlib.sha256(pid.encode()).hexdigest(), 16) % 10 == 0 else "train"


def render(item: dict, thinking: bool) -> str:
    import jinja2

    tpl = Path("models/qwen38-flash-next/chat_template.jinja").read_text()
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(msg):
        raise ValueError(msg)

    env.globals["raise_exception"] = raise_exception
    t = env.from_string(tpl)
    return t.render(messages=item["messages"], tools=TOOLS if item.get("tools") and TOOLS else None,
                    add_generation_prompt=True, enable_thinking=thinking)


if __name__ == "__main__":
    import collections

    ps = all_prompts()
    print(len(ps), collections.Counter(p["kind"] for p in ps), collections.Counter(split(p["id"]) for p in ps))
    print(render(ps[0], False))
    print(render([p for p in ps if p.get("tools")][-1], True)[-1500:])
