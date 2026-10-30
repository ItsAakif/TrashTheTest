"""User provider credentials: encrypted per-user LLM keys + request override."""

from .catalog import (
    PROVIDERS,
    app_default,
    clear_catalog_cache,
    get_catalog,
    get_live_catalog,
    validate_model_id,
    validate_provider,
)
from .crypto import decrypt_key, encrypt_key, mask_hint
from .override import (
    EffectiveLLMProvider,
    build_provider,
    current_request_llm,
    reset_request_llm,
    resolve_for_request,
    set_request_llm,
)
from .routes import router
from .service import (
    ProviderTestError,
    clear_project_override,
    get_project_override,
    get_status,
    remove_credential,
    resolve_effective,
    save_credential,
    set_project_override,
    test_connection,
)

__all__ = [
    "PROVIDERS",
    "ProviderTestError",
    "app_default",
    "build_provider",
    "clear_project_override",
    "current_request_llm",
    "decrypt_key",
    "encrypt_key",
    "get_catalog",
    "get_live_catalog",
    "clear_catalog_cache",
    "get_project_override",
    "get_status",
    "mask_hint",
    "remove_credential",
    "reset_request_llm",
    "resolve_effective",
    "resolve_for_request",
    "router",
    "save_credential",
    "set_project_override",
    "set_request_llm",
    "test_connection",
    "validate_model_id",
    "validate_provider",
    "EffectiveLLMProvider",
]
