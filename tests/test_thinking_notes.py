"""Thinking's two notes: a startup line when replies think by default, a warning when one ran out of tokens thinking."""

from __future__ import annotations

import json

import pytest

from tensorfold.server import thinking_notes

EOS = 3


def test_the_startup_line_names_the_switch_only_when_the_template_has_one(tmp_path):
    assert thinking_notes.startup(tmp_path, True) is None                     # no template
    (tmp_path / "chat_template.jinja").write_text("{%- if enable_thinking is defined %}x{% endif %}")
    line = thinking_notes.startup(tmp_path, True)
    assert "thinking on" in line and "--no-thinking" in line and '"enable_thinking": false' in line
    assert thinking_notes.startup(tmp_path, False) is None
    other = tmp_path / "other"
    other.mkdir()
    (other / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ enable_thinking }}"}))
    assert thinking_notes.template_thinks(other)
    (other / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    assert not thinking_notes.template_thinks(other)


@pytest.mark.parametrize("finish, thinking, content, calls, warned", [
    ("length", True, "", None, True), ("length", True, "  \n", None, True), ("length", True, "Hi", None, False),
    ("stop", True, "", None, False), ("length", False, "", None, False), ("length", True, "", [{"id": 1}], False)])
def test_the_warning_is_for_a_reply_cut_while_thinking(finish, thinking, content, calls, warned):
    line = thinking_notes.unanswered(finish, thinking, content, calls)
    assert (line is not None) == warned
    if warned:
        assert "max_tokens" in line and "reasoning_content" in line and "--no-thinking" in line


PIECES = ["<p>", "<q>", "<a>", "<eos>", "Let me think. ", "</think>", "The answer is 4."]


