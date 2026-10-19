from .base import ContextManager, messages_tokens
from .compressed import ABLATIONS, CompressedContext
from .naive import NaiveContext

REGISTRY: dict[str, type[ContextManager]] = {
    "naive": NaiveContext,
    "compressed": CompressedContext,
    **{cls.name: cls for cls in ABLATIONS},
}


def make_context(name: str, **kwargs) -> ContextManager:
    if name not in REGISTRY:
        raise ValueError(f"unknown context strategy '{name}'. Available: {sorted(REGISTRY)}")
    cls = REGISTRY[name]
    if not issubclass(cls, CompressedContext):
        kwargs.pop("extractor", None)
    return cls(**kwargs)


__all__ = ["CompressedContext", "ContextManager", "NaiveContext", "REGISTRY", "make_context", "messages_tokens"]
