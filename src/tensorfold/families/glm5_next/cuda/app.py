"""GLM request routes validate context before streaming and render an empty think block without a reasoning-effort line when thinking is off."""

from __future__ import annotations

from typing import Any, Callable

from tensorfold.cuda.server import App, PreparedRequest, RequestError
from tensorfold.families.glm5_next.prompts import clear_thinking, thinking_off
from tensorfold.server.errors import CONTEXT_LIMIT


def request_policy(body: dict[str, Any]) -> str | None:
    """A request's own draft policy: its ``tf_policy`` field, else the model id's ``@policy`` suffix; None for neither."""

    model = str(body.get("model") or "")
    return body.get("tf_policy") or (model.split("@", 1)[1] if "@" in model else None)


def policy_problem(body: dict[str, Any], engine: Any) -> str | None:
    """Why a request's own draft policy is malformed, so it is refused with a 400 before streaming; None if valid.

    The engine's own parser decides (``GlmEngine.check_policy``); an engine without one takes the policy as it is."""

    spec = request_policy(body)
    check = getattr(engine, "check_policy", None)
    if spec is None or check is None:
        return None
    try:
        check(spec)
    except ValueError as exc:
        return str(exc)
    return None


class ThinkingOffTemplate:
    """The checkpoint template as GLM-5.3's thinking-off template renders it (``prompts.thinking_off``), with earlier
    turns' reasoning kept unless the request's ``chat_template_kwargs.clear_thinking`` says otherwise
    (``prompts.clear_thinking``)."""

    def __init__(self, inner, clear: bool | None = None) -> None:
        self.inner = inner
        self.efforts = getattr(inner, "efforts", frozenset())
        self.clear = clear_thinking() if clear is None else clear

    def render(self, messages, *, tools, enable_thinking, extra=None) -> str:
        extra = {"clear_thinking": self.clear, **(extra or {})}       # a request's own value wins
        text = self.inner.render(messages, tools=tools, enable_thinking=enable_thinking, extra=extra)
        return text if enable_thinking else thinking_off(text)


class GlmApp(App):
    reads_ignore_eos = True             # ``run`` hands it to the engine's request

    def __init__(self, engine, model_dir, served: str, **kwargs: Any) -> None:
        super().__init__(engine, model_dir, served, **kwargs)
        self.template = ThinkingOffTemplate(self.template)

    def check(self, body: dict[str, Any], *, prepared: PreparedRequest | None = None) -> str | None:
        """Validate the rendered prompt plus max_tokens against the engine context limit before streaming."""

        problem = self._check_fields(body) or policy_problem(body, self.engine)
        limit = getattr(self.engine, "limit", None)
        if problem or limit is None:
            return problem or super().check(body, prepared=prepared)
        if prepared is None:
            try:
                prepared = self._prepare(body, "messages" in body)
            except RequestError as exc:
                return str(exc)
        prompt = len(prepared.prompt)
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        need = prompt + (int(asked) if asked else 1)
        if need <= limit:
            return super().check(body, prepared=prepared)
        detail = f"{prompt} prompt tokens plus max_tokens {int(asked)}" if asked else f"a {prompt}-token prompt"
        # OpenAI's wording, so prepare refuses it as context_length_exceeded (clients compact on it)
        return (f"{CONTEXT_LIMIT} {limit} tokens: this request needs a {need}-token context ({detail}), which exceeds "
                f"the context window this server was started for; shorten the prompt or reply"
                f"{self._restart(need, ' both ranks')}")

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        self.engine.request.policy = request_policy(body)
        self.engine.request.stop_eos = not bool(body.get("ignore_eos", False))
        return super().run(body, chat, emit, prepared=prepared, cancelled=cancelled)
