"""Resolve chat models and embeddings without binding the package to a provider.

Strings go to LangChain's ``init_chat_model`` / ``init_embeddings``
(``provider:model-name``). Instances of ``BaseChatModel`` and ``Embeddings``
are used as given. API keys are passed through only when the caller supplies
them; otherwise the provider SDK reads its own environment variables.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

_SECRET_TOKENS = ("key", "token", "secret", "password", "credential")


class BackendError(TypeError):
    """Raised when a model or embeddings argument cannot be resolved."""


def describe_model(model: BaseChatModel | str) -> str:
    if isinstance(model, str):
        return model
    name = getattr(model, "model_name", None) or getattr(model, "model", None)
    class_name = type(model).__name__
    if isinstance(name, str) and name:
        return f"{class_name}:{name}"
    return class_name


def prompt_cache_supported(model: BaseChatModel, model_name: str = "") -> bool:
    """Whether to attach an explicit cache breakpoint on the system prefix.

    Anthropic reads ``cache_control`` on content blocks. Other providers in
    auto mode keep a stable system prefix (some cache that prefix themselves)
    and are not sent ``cache_control``, which several of them reject.
    """
    llm_type = ""
    try:
        llm_type = str(model._llm_type)
    except Exception:
        llm_type = ""
    blob = " ".join(
        (
            type(model).__module__,
            type(model).__name__,
            llm_type,
            model_name,
        )
    ).casefold()
    return "anthropic" in blob


def resolve_chat_model(
    model: BaseChatModel | str,
    *,
    api_key: str | None = None,
    temperature: float | None = None,
    llm_seed: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Return a chat model.

    A ``BaseChatModel`` is returned unchanged. ``api_key`` applies only when
    ``model`` is a ``provider:model-name`` string. When ``api_key`` is omitted,
    nothing is injected and the provider package uses its usual environment
    variable.
    """
    if isinstance(model, BaseChatModel):
        return model
    if not isinstance(model, str) or not model.strip():
        raise BackendError(
            "model must be a 'provider:model-name' string (passed to "
            "langchain.chat_models.init_chat_model) or a LangChain BaseChatModel instance."
        )
    kwargs: dict[str, Any] = dict(model_kwargs or {})
    if temperature is not None and "temperature" not in kwargs:
        kwargs["temperature"] = temperature
    if api_key is not None:
        kwargs["api_key"] = api_key
    if llm_seed is not None and "seed" not in kwargs:
        kwargs["seed"] = llm_seed
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise BackendError(
            "langchain.chat_models.init_chat_model is unavailable. Install the 'langchain' package."
        ) from exc
    try:
        resolved = init_chat_model(model.strip(), **kwargs)
    except ImportError as exc:
        raise BackendError(
            f"Could not initialize {model!r}. Install the LangChain integration extra "
            "for that provider (for example the openai, anthropic, google, mistral, "
            "groq, ollama, azure, or aws extra) and pass api_key or set the provider's "
            f"environment variable. Underlying error: {exc}"
        ) from exc
    if not isinstance(resolved, BaseChatModel):
        raise BackendError(
            f"init_chat_model({model!r}) did not return a BaseChatModel. "
            "Pass a fully specified provider:model-name string or a chat model instance."
        )
    return resolved


def resolve_embeddings(
    embeddings: Embeddings | str | None,
    *,
    api_key: str | None = None,
    embeddings_kwargs: dict[str, Any] | None = None,
) -> Embeddings | None:
    """Return embeddings, or None when the caller wants lexical comparisons.

    A string is sent to ``langchain.embeddings.init_embeddings``. An
    ``Embeddings`` instance is returned unchanged.
    """
    if embeddings is None:
        return None
    if isinstance(embeddings, Embeddings):
        return embeddings
    if not isinstance(embeddings, str) or not embeddings.strip():
        raise BackendError(
            "embeddings must be a 'provider:model-name' string, a LangChain Embeddings "
            "instance, or None for lexical merge candidates."
        )
    kwargs: dict[str, Any] = dict(embeddings_kwargs or {})
    if api_key is not None:
        kwargs["api_key"] = api_key
    try:
        from langchain.embeddings import init_embeddings
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise BackendError(
            "langchain.embeddings.init_embeddings is unavailable. Install the 'langchain' package."
        ) from exc
    try:
        resolved = init_embeddings(embeddings.strip(), **kwargs)
    except ImportError as exc:
        raise BackendError(
            f"Could not initialize embeddings {embeddings!r}. Install that provider's "
            f"LangChain integration. Underlying error: {exc}"
        ) from exc
    if not isinstance(resolved, Embeddings):
        raise BackendError(f"init_embeddings({embeddings!r}) did not return an Embeddings instance.")
    return resolved


def redact(mapping: dict[str, Any] | None) -> dict[str, Any]:
    """Copy ``mapping`` with secret-looking values replaced."""
    if not mapping:
        return {}
    redacted: dict[str, Any] = {}
    for key, value in mapping.items():
        if any(token in str(key).casefold() for token in _SECRET_TOKENS):
            redacted[key] = "***"
        elif isinstance(value, dict):
            redacted[key] = redact(value)
        else:
            redacted[key] = value
    return redacted
