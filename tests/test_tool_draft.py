"""Tool-call drafts follow the schema."""

from __future__ import annotations

from tensorfold.engine.tool_draft import ToolCallProposer, tool_schema

TOOLS = [
    {"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {"path": {}, "offset": {}, "limit": {}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "edit", "parameters": {
        "type": "object", "properties": {"path": {}, "old_text": {}, "new_text": {}},
        "required": ["path", "old_text", "new_text"]}}},
]


class _Tok:
    def convert_tokens_to_ids(self, token: str) -> int:
        return 7


def _next(text: str) -> str | None:
    return ToolCallProposer(_Tok(), TOOLS, 0).structure(text)


def test_schema_puts_required_parameters_first():
    assert tool_schema(TOOLS) == {"read": ["path", "offset", "limit"], "edit": ["path", "old_text", "new_text"]}


def test_structure_follows_the_call():
    call = "<tool_call>\n<function="
    assert _next("") == "<tool_call>\n<function="
    assert _next(call) is None                      # the model picks the tool
    assert _next(call + "edit") == ">\n<parameter=path>\n"
    assert _next(call + "edit>") == "\n<parameter=path>\n"
    value = call + "edit>\n<parameter=path>\ncalc.py\n"
    assert _next(value) == "</parameter>\n<parameter=old_text>\n"
    assert _next(value + "</parameter>") == "\n<parameter=old_text>\n"
    done = call + "read>\n<parameter=path>\ncalc.py\n"
    assert _next(done) == "</parameter>\n<parameter=offset>\n"
    assert _next(call + "read>\n<parameter=path>\ncalc.py\n</parameter>\n</function>") == "\n</tool_call>"
    assert _next("Some prose first") is None
    # a multi-line value is never closed early
    assert _next(call + "edit>\n<parameter=path>\na.py\n</parameter>\n<parameter=old_text>\ndef f():\n") is None


def test_streamed_tool_call_arguments_equal_the_parsed_call():
    import json

    from tensorfold.server.tools import parse_tool_calls_from_content
    from tensorfold.engine.tool_draft import ToolCallStreamer

    tools = [{"type": "function", "function": {"name": "write", "parameters": {
        "type": "object", "properties": {"path": {}, "content": {}}, "required": ["path", "content"]}}}]
    full = ('Writing it.\n<tool_call>\n<function=write>\n<parameter=path>\nsite/index.html\n</parameter>\n'
            '<parameter=content>\n<!DOCTYPE html>\n<p class="x">Hi "there" \\ ok</p>\n  \n</parameter>\n'
            '</function>\n</tool_call>')
    streamer = ToolCallStreamer(tools)
    deltas = []
    for n in range(1, len(full) + 1, 5):
        deltas += streamer.feed(full[:n])
    deltas += streamer.feed(full)
    parts = [d["tool_calls"][0]["function"] for d in deltas]
    assert parts[0]["name"] == "write" and streamer.streamed
    arguments = "".join(p.get("arguments", "") for p in parts)
    _, calls = parse_tool_calls_from_content(full, tools)
    assert json.loads(arguments) == json.loads(calls[0]["function"]["arguments"])


def test_unknown_tool_is_not_streamed():
    from tensorfold.engine.tool_draft import ToolCallStreamer

    streamer = ToolCallStreamer([{"type": "function", "function": {"name": "read", "parameters": {}}}])
    assert streamer.feed("<tool_call>\n<function=bash>\n<parameter=command>\nls\n") == []
    assert not streamer.streamed


def test_parameter_values_keep_their_own_whitespace():
    """One framing newline a side is the template's; the rest belongs to the value.

    A written file's last newline (and an edit's indentation) must reach the client, and
    the client's resent history must then render to the tokens the model wrote: stripping
    the value re-prefilled a whole 15,007-token file write on the next turn (51 s).
    """
    import json

    from tensorfold.server.tools import parse_tool_calls_from_content
    from tensorfold.engine.tool_draft import ToolCallStreamer

    tools = [{"type": "function", "function": {"name": "write", "parameters": {
        "type": "object", "properties": {"path": {}, "content": {}}}}}]
    values = {"path": "site/index.html", "content": "  <html>\n    <p>x</p>\n</html>\n"}
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in values.items())
    full = f"Writing it.\n\n<tool_call>\n<function=write>\n{body}</function>\n</tool_call>"
    _, calls = parse_tool_calls_from_content(full, tools)
    assert json.loads(calls[0]["function"]["arguments"]) == values
    for step in (1, 3, 7):          # the framing newline may arrive after its tag
        streamer = ToolCallStreamer(tools)
        deltas = []
        for n in range(1, len(full) + 1, step):
            deltas += streamer.feed(full[:n])
        deltas += streamer.feed(full)
        arguments = "".join(d["tool_calls"][0]["function"].get("arguments", "") for d in deltas)
        assert json.loads(arguments) == values, step


def test_the_family_rounds_copy_gate_takes_structure_and_the_fallbacks_copies():
    from types import SimpleNamespace

    class Copies:
        last_match = 0

        def propose(self, context, max_draft):
            self.last_match = 9
            return [5, 6, 7]

    blank = SimpleNamespace(decode=lambda ids: "", encode=lambda text, **_: [7, 8], convert_tokens_to_ids=lambda t: -1)
    opening = ToolCallProposer(blank, TOOLS, 3, fallback=Copies())
    assert opening.propose([1, 2, 3, 4], 15) == [7, 8] and opening.last_match >= 1 << 20   # the call's opening
    prose = SimpleNamespace(decode=lambda ids: "Some prose.", encode=lambda text, **_: [1],
                            convert_tokens_to_ids=lambda t: -1)
    fallback = ToolCallProposer(prose, TOOLS, 3, fallback=Copies())
    assert fallback.propose([1, 2, 3, 4], 15) == [5, 6, 7] and fallback.last_match == 9    # the fallback's copy
