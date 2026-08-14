"""Fresh, machine-readable model metadata and prices."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .locks import exclusive_lock
from .worktrees import AOPError


CATALOG_URL = "https://models.dev/api.json"
CATALOG_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class ModelCatalog:
    providers: dict[str, Any]
    fetched_at: float
    source: str
    sha256: str

    @property
    def version(self) -> str:
        return f"models-dev-{time.strftime('%Y-%m-%d', time.gmtime(self.fetched_at))}-{self.sha256[:12]}"

    def model(self, provider: str, model: str) -> dict[str, Any] | None:
        provider_data = self.providers.get(provider)
        if not isinstance(provider_data, dict):
            return None
        models = provider_data.get("models")
        if not isinstance(models, dict):
            return None
        value = models.get(model)
        return value if isinstance(value, dict) else None


def ensure_catalog_fresh(*, force: bool = False) -> ModelCatalog:
    """Return a catalog no older than 24 hours, refreshing it atomically if needed."""
    path = _cache_path()
    cached = _read_cache(path)
    if not force and cached is not None and _is_fresh(cached):
        return cached

    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(
        path.with_suffix(f"{path.suffix}.lock"),
        "model catalog refresh",
        blocking=True,
    ):
        cached = _read_cache(path)
        if not force and cached is not None and _is_fresh(cached):
            return cached
        try:
            catalog = _download_catalog()
        except (OSError, ValueError, json.JSONDecodeError) as error:
            age = _age_description(cached)
            raise AOPError(
                f"could not refresh model prices from {CATALOG_URL}; "
                f"cached data is {age}: {error}"
            ) from error
        _write_cache(path, catalog)
        return catalog


def _cache_path() -> Path:
    override = os.environ.get("AOP_MODEL_CATALOG_CACHE")
    if override:
        return Path(override).expanduser()
    cache_home = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ).expanduser()
    return cache_home / "aop" / "models-dev.json"


def _is_fresh(catalog: ModelCatalog) -> bool:
    age = time.time() - catalog.fetched_at
    return 0 <= age < CATALOG_TTL_SECONDS


def _age_description(catalog: ModelCatalog | None) -> str:
    if catalog is None:
        return "missing"
    hours = max(0.0, time.time() - catalog.fetched_at) / 3600
    return f"{hours:.1f} hours old"


def _download_catalog() -> ModelCatalog:
    request = urllib.request.Request(
        CATALOG_URL,
        headers={"Accept": "application/json", "User-Agent": "aop-model-catalog/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            content = response.read(20_000_001)
    except urllib.error.URLError as error:
        raise OSError(str(error.reason)) from error
    if len(content) > 20_000_000:
        raise ValueError("model catalog exceeded 20 MB")
    providers = json.loads(content)
    _validate_providers(providers)
    return ModelCatalog(
        providers=providers,
        fetched_at=time.time(),
        source=CATALOG_URL,
        sha256=_catalog_hash(providers),
    )


def _read_cache(path: Path) -> ModelCatalog | None:
    try:
        value = json.loads(path.read_text())
        providers = value["providers"]
        _validate_providers(providers)
        fetched_at = float(value["fetched_at"])
        source = value["source"]
        sha256 = value["sha256"]
        if not isinstance(source, str) or not isinstance(sha256, str):
            return None
        if sha256 != _catalog_hash(providers):
            return None
        return ModelCatalog(providers, fetched_at, source, sha256)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, catalog: ModelCatalog) -> None:
    value = {
        "fetched_at": catalog.fetched_at,
        "providers": catalog.providers,
        "schema_version": 1,
        "sha256": catalog.sha256,
        "source": catalog.source,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(f"{json.dumps(value, separators=(',', ':'))}\n")
    os.replace(temporary, path)


def _validate_providers(value: object) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError("model catalog is not a nonempty object")
    for provider in value.values():
        if not isinstance(provider, dict) or not isinstance(
            provider.get("models"), dict
        ):
            raise ValueError("model catalog contains an invalid provider")


def _catalog_hash(providers: dict[str, Any]) -> str:
    canonical = json.dumps(providers, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()
