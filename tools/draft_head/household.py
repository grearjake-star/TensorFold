"""A second prompt set (``extra`` in the prompts file, see prompts_v1): chat, prose and advice prompts and short
dialogs, the kinds where a checkpoint's own MTP head tends to draft worst. Bring your own; none ship."""

from __future__ import annotations

from prompts_v1 import DATA, EXCLUDED, SYSTEMS

EXTRA = DATA.get("extra", {})


def household_prompts() -> list[dict]:
    """The extra set as work items (ids hchat-/hprose-/hadvice-/hdlg-NNN, kept for data made by earlier runs)."""

    items: list[dict] = []
    for kind, pre in (("chat", "hchat"), ("prose", "hprose"), ("advice", "hadvice")):
        for j, text in enumerate(EXTRA.get(kind, [])):
            if text in EXCLUDED:
                raise ValueError(f"an extra {kind} prompt is in the excluded set: {text[:60]!r}")
            sysmsg = SYSTEMS[(j + 1) % len(SYSTEMS)]
            msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [{"role": "user", "content": text}]
            items.append({"id": f"{pre}-{j:03d}", "kind": kind, "messages": msgs})
    for j, dialog in enumerate(EXTRA.get("dialogs", [])):
        items.append({"id": f"hdlg-{j:03d}", "kind": "chat", "messages": dialog})
    return items
