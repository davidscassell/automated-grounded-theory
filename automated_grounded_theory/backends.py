"""Resolve chat models and embeddings without binding the package to a provider.

Strings go to LangChain's ``init_chat_model`` / ``init_embeddings``
(``provider:model-name``). Instances of ``BaseChatModel`` and ``Embeddings``
are used as given. API keys are passed through only when the caller supplies
them; otherwise the provider SDK reads its own environment variables.
"""

from __future__ import annotations

import asyncio
import contextvars
import weakref
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

_SECRET_TOKENS = ("key", "token", "secret", "password", "credential")

# Applied when this package builds a model from a provider:model string.
DEFAULT_REQUEST_TIMEOUT = 120.0


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


class _Missing:
    pass


_MISSING = _Missing()

# Non-secret settings that change model output. Keys, tokens, and client
# objects are not read. ``model_kwargs`` is copied through :func:`redact`.
_CHAT_SETTING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "n",
    "reasoning_effort",
)
_CHAT_FLOAT_FIELDS = frozenset({"temperature", "top_p", "presence_penalty", "frequency_penalty"})
_CHAT_INT_FIELDS = frozenset(
    {"top_k", "max_tokens", "max_completion_tokens", "max_output_tokens", "seed", "n"}
)
_EMBED_SETTING_FIELDS = (
    "dimensions",
    "embedding_dimensions",
    "output_dimensionality",
    "azure_endpoint",
    "azure_deployment",
    "deployment",
    "deployment_name",
    "base_url",
    "openai_api_base",
    "api_base",
    "endpoint",
    "api_version",
    "openai_api_version",
    "chunk_size",
)


def _declared_setting(obj: Any, name: str) -> Any:
    """A setting the instance actually declares, or ``_MISSING``.

    Parent classes expose many attributes. Only fields this class declares,
    or values stored on the instance, are part of the fingerprint.
    """
    fields = getattr(type(obj), "model_fields", None)
    declared = isinstance(fields, dict) and name in fields
    stored = name in getattr(obj, "__dict__", {})
    if not declared and not stored:
        return _MISSING
    try:
        value = getattr(obj, name)
    except Exception:
        return _MISSING
    if callable(value):
        return _MISSING
    return value


def _provider_name(model: Any) -> str:
    for name in ("model_provider", "provider"):
        value = _declared_setting(model, name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    try:
        llm_type = str(model._llm_type).strip()
    except Exception:
        llm_type = ""
    if llm_type:
        return llm_type
    return type(model).__module__


def _model_id(model: Any) -> str:
    for name in ("model", "model_name", "model_id"):
        value = _declared_setting(model, name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _clean_setting(name: str, value: Any) -> Any:
    # Allowlisted names such as max_tokens contain the substring "token".
    # Secrets are kept out by the allowlist, and model_kwargs goes through redact.
    if name in _CHAT_FLOAT_FIELDS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return _MISSING
    if name in _CHAT_INT_FIELDS:
        if isinstance(value, bool) or value is None:
            return _MISSING
        try:
            return int(value)
        except (TypeError, ValueError):
            return _MISSING
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return redact(value)
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return _MISSING


def chat_model_settings(model: BaseChatModel | str | None) -> dict[str, Any]:
    """Non-secret settings from the chat model that will actually run.

    A ``BaseChatModel`` keeps the temperature and limits it was built with.
    The ``temperature=`` argument of :func:`open_code` is not applied to an
    instance, so the fingerprint reads the instance. Secrets are omitted.
    """
    if model is None:
        return {}
    if isinstance(model, str):
        provider = model.split(":", 1)[0] if ":" in model else ""
        return {"model": model, "provider": provider}
    settings: dict[str, Any] = {
        "class": type(model).__name__,
        "provider": _provider_name(model),
    }
    model_id = _model_id(model)
    if model_id:
        settings["model"] = model_id
    for name in _CHAT_SETTING_FIELDS:
        value = _declared_setting(model, name)
        if value is _MISSING or value is None or value == "":
            continue
        cleaned = _clean_setting(name, value)
        if cleaned is _MISSING:
            continue
        settings[name] = cleaned
    kwargs = _declared_setting(model, "model_kwargs")
    if isinstance(kwargs, dict) and kwargs:
        settings["model_kwargs"] = redact(kwargs)
    return settings


def embedding_model_settings(embeddings: Embeddings | str | None) -> dict[str, Any]:
    """Non-secret settings from the embedding model that will actually run.

    ``None`` is the lexical retriever. Dimensions, deployment, and endpoint
    are included. API keys are not.
    """
    if embeddings is None:
        return {"provider": "lexical", "model": "lexical"}
    if isinstance(embeddings, str):
        provider = embeddings.split(":", 1)[0] if ":" in embeddings else ""
        return {"model": embeddings, "provider": provider}
    settings: dict[str, Any] = {
        "class": type(embeddings).__name__,
        "provider": _provider_name(embeddings),
    }
    model_id = _model_id(embeddings)
    if model_id:
        settings["model"] = model_id
    for name in _EMBED_SETTING_FIELDS:
        value = _declared_setting(embeddings, name)
        if value is _MISSING or value is None or value == "" or value == {}:
            continue
        cleaned = _clean_setting(name, value)
        if cleaned is _MISSING:
            continue
        settings[name] = cleaned
    kwargs = _declared_setting(embeddings, "model_kwargs")
    if isinstance(kwargs, dict) and kwargs:
        redacted = redact(kwargs)
        settings["model_kwargs"] = redacted
        for name in ("dimensions", "embedding_dimensions", "output_dimensionality"):
            if name not in settings and name in redacted:
                settings[name] = redacted[name]
    return settings


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


def provider_http_timeout(seconds: float):
    """connect, read, write, and pool timeouts for one provider HTTP client.

    A single float leaves the pool and connect phases unbounded on some
    clients. ``None`` when ``seconds`` is 0 or less, which means no deadline.
    """
    seconds = float(seconds)
    if seconds <= 0:
        return None
    try:
        import httpx
    except ImportError:
        return seconds
    return httpx.Timeout(connect=min(10.0, seconds), read=seconds, write=seconds, pool=seconds)


def provider_socket_timeout(seconds: float) -> float:
    """Read/pool backstop slightly past the call deadline.

    The monotonic watchdog is the deadline. This socket timeout is what closes
    a connection the watchdog already abandoned, including a client the SDK
    builds during the call. It sits a little past the watchdog so a healthy
    loop still reports the configured request timeout.
    """
    seconds = float(seconds)
    if seconds <= 0:
        return 0.0
    cushion = max(0.5, min(5.0, seconds * 0.1))
    return seconds + cushion


_HTTP_TIMEOUT_CAP: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "agt_http_timeout_cap",
    default=None,
)
_HTTPX_PATCHED = False


def _tighten_timeout(given: Any, seconds: float):
    """Return an httpx timeout no looser than ``seconds``.

    ``None`` and an httpx timeout with an unset phase are not left open.
    A shorter caller timeout is kept.
    """
    cap = provider_http_timeout(seconds)
    if cap is None or isinstance(cap, (int, float)):
        return given
    try:
        import httpx
    except ImportError:
        return given
    if given is None or given is getattr(httpx, "USE_CLIENT_DEFAULT", None):
        return cap
    if isinstance(given, (int, float)):
        if float(given) <= 0:
            return cap
        return provider_http_timeout(min(float(given), seconds)) or cap
    if isinstance(given, httpx.Timeout):

        def phase(name: str) -> float:
            value = getattr(given, name, None)
            limit = float(getattr(cap, name))
            if value is None:
                return limit
            return min(float(value), limit)

        return httpx.Timeout(connect=phase("connect"), read=phase("read"), write=phase("write"), pool=phase("pool"))
    return cap


def install_httpx_timeout_cap() -> None:
    """Make httpx clients created or used during a timed call honor the cap.

    The xAI and OpenAI-compatible SDKs often build the httpx client inside
    ``ainvoke``, or pass ``timeout=None`` on the request, which overrides the
    client default. Patching construction and ``request`` applies the cap to
    that client. The cap is a context variable, so a call that did not set one
    is unchanged.
    """
    global _HTTPX_PATCHED
    if _HTTPX_PATCHED:
        return
    try:
        import httpx
    except ImportError:
        return

    def wrap_init(original):
        def __init__(self, *args, **kwargs):
            seconds = _HTTP_TIMEOUT_CAP.get()
            if seconds:
                kwargs["timeout"] = _tighten_timeout(kwargs.get("timeout"), float(seconds))
            return original(self, *args, **kwargs)

        return __init__

    def wrap_request(original):
        def request(self, method, url, *args, **kwargs):
            seconds = _HTTP_TIMEOUT_CAP.get()
            if seconds:
                current = kwargs.get("timeout", httpx.USE_CLIENT_DEFAULT)
                if current is httpx.USE_CLIENT_DEFAULT:
                    current = getattr(self, "timeout", None)
                kwargs["timeout"] = _tighten_timeout(current, float(seconds))
            return original(self, method, url, *args, **kwargs)

        return request

    httpx.Client.__init__ = wrap_init(httpx.Client.__init__)
    httpx.AsyncClient.__init__ = wrap_init(httpx.AsyncClient.__init__)
    httpx.Client.request = wrap_request(httpx.Client.request)
    httpx.AsyncClient.request = wrap_request(httpx.AsyncClient.request)
    _HTTPX_PATCHED = True


def bind_http_timeout_cap(seconds: float):
    """Set the socket-timeout cap for clients built in this context. Returns the token."""
    install_httpx_timeout_cap()
    return _HTTP_TIMEOUT_CAP.set(provider_socket_timeout(seconds))


def reset_http_timeout_cap(token) -> None:
    _HTTP_TIMEOUT_CAP.reset(token)


def install_provider_http_timeout(model, seconds: float | None):
    """Set connect/read/write/pool timeouts on a chat model and any live client.

    ``init_chat_model`` still receives the numeric ``timeout`` argument.
    This writes the structured timeout onto the instance the call will use,
    including a client that was already constructed. A model without those
    attributes is left unchanged.
    """
    if model is None or seconds is None or float(seconds) <= 0:
        return model
    timeout = provider_http_timeout(float(seconds))
    if timeout is None:
        return model
    fields = getattr(type(model), "model_fields", None)
    for name in ("timeout", "request_timeout"):
        if isinstance(fields, dict) and name in fields:
            try:
                setattr(model, name, timeout)
            except Exception:
                continue
    for name in (
        "http_client",
        "http_async_client",
        "client",
        "async_client",
        "root_client",
        "root_async_client",
    ):
        client = getattr(model, name, None)
        if client is None or not hasattr(client, "timeout"):
            continue
        try:
            client.timeout = timeout
        except Exception:
            continue
    return model


_OWNER_ATTR = "_agt_owner_loop"
_FACTORY_ATTR = "_agt_client_factory"


class CrossLoopClientError(RuntimeError):
    """A client bound to another loop was not built by :class:`ProviderClientFactory`."""


class ProviderClientFactory:
    """Configuration for one provider HTTP client identity.

    A cross-loop replacement is built from this configuration, not from a few
    attributes copied off a live client. Two factories stay two clients even
    when they share a base URL. Auth, cookies, params, proxies, TLS, and
    ``trust_env`` are part of the identity.
    """

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)
        self._clients: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def build(self, *, timeout: Any, loop: Any) -> Any:
        import httpx

        kwargs = dict(self.config)
        kwargs["timeout"] = timeout
        client = httpx.AsyncClient(**kwargs)
        try:
            setattr(client, _OWNER_ATTR, loop)
            setattr(client, _FACTORY_ATTR, self)
        except Exception:
            pass
        return client

    def for_loop(self, loop: Any, timeout: Any) -> Any:
        cached = self._clients.get(loop)
        if cached is not None:
            try:
                if not cached.is_closed:
                    cached.timeout = timeout
                    return cached
            except Exception:
                pass
        client = self.build(timeout=timeout, loop=loop)
        self._clients[loop] = client
        return client


def _holds_foreign_asyncio_primitive(root: Any, loop: Any) -> bool:
    """True when a pooled connection still holds an asyncio primitive from another loop.

    httpx 0.28 does not store the loop on the connection pool. A keepalive
    stream does: its protocol events are created on the loop that opened the
    connection. Reading that binding is how a foreign client is refused. The
    client is not rebuilt from its URL and headers.
    """
    pool = getattr(getattr(root, "_transport", None), "_pool", None)
    if pool is None:
        return False
    seen: set[int] = set()
    stack: list[tuple[Any, int]] = [(pool, 0)]
    while stack:
        obj, depth = stack.pop()
        if obj is None or depth > 8:
            continue
        ident = id(obj)
        if ident in seen:
            continue
        seen.add(ident)
        if len(seen) > 200:
            break
        module = type(obj).__module__
        if module.startswith("asyncio"):
            bound = getattr(obj, "_loop", None)
            if bound is not None and bound is not loop:
                return True
            continue
        if not (module.startswith(("httpx", "httpcore", "anyio")) or isinstance(obj, (list, tuple))):
            continue
        children: list[Any] = []
        if isinstance(obj, (list, tuple)):
            children.extend(obj)
        else:
            try:
                children.extend(vars(obj).values())
            except Exception:
                pass
            slots = getattr(type(obj), "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if not isinstance(slot, str) or slot.startswith("__"):
                    continue
                try:
                    children.append(getattr(obj, slot))
                except Exception:
                    continue
        for child in children:
            if child is None or isinstance(child, (str, bytes, int, float, bool)):
                continue
            stack.append((child, depth + 1))
    return False


def client_for_running_loop(template: Any, timeout: Any) -> Any:
    """Return a client for the running loop without inventing a new configuration.

    A client this factory built can be rebuilt on the running loop with the
    same auth, cookies, and transport settings. A client that was not built
    here and is already bound to another loop raises :class:`CrossLoopClientError`.
    Binding is an owner stamp or a keepalive connection whose asyncio primitives
    belong to another loop. The client is not replaced by one reconstructed
    from the base URL and headers. A client with no recorded owner and no
    foreign connection is left in place and only has its timeout set.
    """
    loop = asyncio.get_running_loop()
    factory = getattr(template, _FACTORY_ATTR, None)
    if factory is not None:
        return factory.for_loop(loop, timeout)
    owned = getattr(template, _OWNER_ATTR, None)
    if (owned is not None and owned is not loop) or _holds_foreign_asyncio_primitive(template, loop):
        raise CrossLoopClientError(
            "This HTTP client is bound to another event loop and was not created by "
            "ProviderClientFactory. The package will not substitute a client rebuilt "
            "from a few attributes. Build it with ProviderClientFactory so auth, "
            "cookies, params, proxies, TLS, and trust_env stay with that identity."
        )
    try:
        if not template.is_closed:
            template.timeout = timeout
            return template
    except Exception:
        pass
    raise CrossLoopClientError(
        "This HTTP client cannot be moved onto the running loop because it was not "
        "created by ProviderClientFactory."
    )


def _httpx_client_slots(root: Any):
    """``(owner, name, client)`` for each pooled async client under ``root``."""
    try:
        import httpx
    except ImportError:
        return
    seen: set[int] = set()
    stack: list[tuple[Any, Any, Any, int]] = [(None, None, root, 0)]
    names = (
        "http_client",
        "http_async_client",
        "async_client",
        "root_async_client",
        "client",
        "root_client",
        "_client",
        "bound",
    )
    while stack:
        owner, name, obj, depth = stack.pop()
        if obj is None or id(obj) in seen or depth > 6:
            continue
        seen.add(id(obj))
        if isinstance(obj, (str, bytes, int, float, bool)):
            continue
        if isinstance(obj, httpx.AsyncClient):
            if owner is not None and name is not None:
                yield owner, name, obj
            continue
        for attr in names:
            try:
                child = getattr(obj, attr, None)
            except Exception:
                child = None
            if child is not None and child is not obj:
                stack.append((obj, attr, child, depth + 1))
        steps = getattr(obj, "steps", None)
        if isinstance(steps, list):
            for index, step in enumerate(steps):
                stack.append((steps, index, step, depth + 1))


async def arm_provider_clients(root: Any, seconds: float) -> None:
    """Give each pooled async client to the running loop and set phase timeouts.

    A client that already served a request on another loop is not reused here.
    This loop gets a client constructed on it. Connect, read, write, and pool
    each use ``seconds``.
    """
    if root is None or seconds is None or float(seconds) <= 0:
        return
    timeout = provider_http_timeout(float(seconds))
    if timeout is None or isinstance(timeout, (int, float)):
        return
    try:
        import httpx  # noqa: F401
    except ImportError:
        return
    for owner, name, client in _httpx_client_slots(root):
        try:
            client.timeout = timeout
        except Exception:
            pass
        try:
            replacement = client_for_running_loop(client, timeout)
        except CrossLoopClientError:
            raise
        except Exception:
            continue
        if replacement is client:
            continue
        try:
            if isinstance(owner, list):
                owner[name] = replacement
            else:
                setattr(owner, name, replacement)
        except Exception:
            continue


def resolve_chat_model(
    model: BaseChatModel | str,
    *,
    api_key: str | None = None,
    temperature: float | None = None,
    llm_seed: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
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
    if (
        timeout is not None
        and timeout > 0
        and "timeout" not in kwargs
        and "request_timeout" not in kwargs
    ):
        kwargs["timeout"] = timeout
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
    applied = kwargs.get("timeout", timeout)
    if isinstance(applied, (int, float)) and not isinstance(applied, bool):
        install_provider_http_timeout(resolved, float(applied))
    return resolved


def resolve_embeddings(
    embeddings: Embeddings | str | None,
    *,
    api_key: str | None = None,
    embeddings_kwargs: dict[str, Any] | None = None,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
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
    if (
        timeout is not None
        and timeout > 0
        and "timeout" not in kwargs
        and "request_timeout" not in kwargs
    ):
        kwargs["timeout"] = timeout
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
        elif isinstance(value, (list, tuple)):
            redacted[key] = [_redact_value(item) for item in value]
        else:
            redacted[key] = _redact_value(value)
    return redacted


def _redact_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return redact(value)
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    if callable(value):
        return None
    return repr(value)
