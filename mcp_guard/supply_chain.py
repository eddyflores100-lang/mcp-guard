"""Supply chain verification for npm-distributed MCP servers.

Checks the npm registry for sigstore attestations (provenance / SLSA)
published alongside a package version. The registry contract used here:

- ``GET /{name}`` — package metadata (``dist-tags``); 404 means the
  package does not exist.
- ``GET /-/npm/v1/packages/attestations/{name}/{version}`` — the
  package's attestations; 404 means the version exists but nothing was
  published with provenance (unsigned).

All network access goes through an injectable ``JsonFetcher`` so the
module stays fully offline-testable (no network calls in the test suite,
mirroring the canary extra's philosophy).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from enum import Enum
from typing import Any, NamedTuple, cast
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

NPM_REGISTRY = "https://registry.npmjs.org"
NPM_ATTENDATIONS_PATH = "/-/npm/v1/packages/attestations"

#: Predicate types that carry a build provenance statement (SLSA).
PROVENANCE_PREDICATE_PREFIX = "https://slsa.dev/provenance"

#: Allowed characters in a version tag (semver core, prerelease and build
#: metadata; deliberately permissive — the registry is the final judge).
_VERSION_RE = re.compile(r"^[A-Za-z0-9.\-_+]+$")

#: Allowed characters for unscoped package names (npm naming rules).
_NAME_RE = re.compile(r"^[a-zA-Z0-9\-_~][a-zA-Z0-9\-._~]*$")

#: Allowed characters for scope and package parts inside a scoped name.
_PART_RE = re.compile(r"^[a-zA-Z0-9\-_~][a-zA-Z0-9\-._~]*$")

FetchJson = Callable[[str], "dict[str, Any]"]


class InvalidPackageRefError(ValueError):
    """Raised when a package reference cannot be parsed."""


class RegistryNotFoundError(Exception):
    """The requested registry resource does not exist (HTTP 404)."""


class RegistryError(Exception):
    """The registry could not be reached or answered unexpectedly."""


class NpmPackageRef(NamedTuple):
    """A parsed npm package reference: name plus optional pinned version."""

    name: str
    version: str | None


class SupplyChainStatus(str, Enum):
    """Outcome of a supply chain check."""

    SIGNED = "signed"
    UNSIGNED = "unsigned"
    NOT_FOUND = "not_found"
    REGISTRY_ERROR = "registry_error"


class AttestationInfo(BaseModel):
    """One sigstore attestation published with a package version."""

    predicate_type: str
    bundle_media_type: str
    is_provenance: bool


class SupplyChainResult(BaseModel):
    """Complete supply chain verification result for one package version."""

    package: str
    version: str
    status: SupplyChainStatus
    attestations: list[AttestationInfo] = Field(default_factory=list[AttestationInfo])
    has_provenance: bool = False
    message: str = ""


def parse_npm_ref(package_ref: str) -> NpmPackageRef:
    """Parse an npm package reference into name and optional version.

    Accepts ``name``, ``name@version``, ``@scope/name`` and
    ``@scope/name@version``. Raises :class:`InvalidPackageRefError` for
    anything that cannot be a valid npm reference.
    """
    ref = package_ref.strip()
    if not ref or any(ch.isspace() for ch in ref):
        raise InvalidPackageRefError(f"Invalid npm package reference: {package_ref!r}")

    version: str | None
    version_pinned = False
    if ref.startswith("@"):
        body = ref[1:]
        if "@" in body:
            scoped, raw_version = body.rsplit("@", 1)
            name = f"@{scoped}"
            version_pinned = True
        else:
            name = ref
            raw_version = ""
    else:
        if "@" in ref:
            name, raw_version = ref.rsplit("@", 1)
            version_pinned = True
        else:
            name = ref
            raw_version = ""

    if version_pinned and not raw_version:
        raise InvalidPackageRefError(f"Empty version in package reference: {package_ref!r}")
    if raw_version:
        if not _VERSION_RE.match(raw_version):
            raise InvalidPackageRefError(f"Invalid version in package reference: {raw_version!r}")
        version = raw_version
    else:
        version = None

    _validate_name(name)
    return NpmPackageRef(name=name, version=version)


def _validate_name(name: str) -> None:
    """Validate a (possibly scoped) npm package name."""
    if name.startswith("@"):
        if "/" not in name:
            raise InvalidPackageRefError(f"Scoped name missing '/': {name!r}")
        scope, _, pkg = name.partition("/")
        if not scope[1:] or not pkg or not _PART_RE.match(scope[1:]) or not _PART_RE.match(pkg):
            raise InvalidPackageRefError(f"Invalid scoped package name: {name!r}")
        return
    if not _NAME_RE.match(name):
        raise InvalidPackageRefError(f"Invalid package name: {name!r}")


def metadata_url(package_name: str) -> str:
    """Registry URL for a package's metadata document."""
    return f"{NPM_REGISTRY}/{quote(package_name, safe='@/')}"


def attestations_url(package_name: str, version: str) -> str:
    """Registry URL for a package version's attestations."""
    quoted = quote(package_name, safe="@/")
    return f"{NPM_REGISTRY}{NPM_ATTENDATIONS_PATH}/{quoted}/{quote(version, safe='')}"


def default_fetch_json(url: str) -> dict[str, Any]:
    """Fetch a JSON document from the registry (stdlib urllib, no deps).

    Raises :class:`RegistryNotFoundError` on HTTP 404 and :class:`RegistryError`
    for any other transport or decoding failure.
    """
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=15) as response:
            payload = response.read()
    except HTTPError as e:
        if e.code == 404:
            raise RegistryNotFoundError(url) from e
        raise RegistryError(f"HTTP {e.code} for {url}") from e
    except OSError as e:
        raise RegistryError(f"Cannot reach registry: {e}") from e

    try:
        data: Any = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RegistryError(f"Registry returned non-JSON payload for {url}") from e
    if not isinstance(data, dict):
        raise RegistryError(f"Registry returned unexpected payload for {url}")
    return cast("dict[str, Any]", data)


def _extract_latest_version(metadata: dict[str, Any]) -> str:
    """Read ``dist-tags.latest`` from a package metadata document."""
    dist_tags: object = metadata.get("dist-tags", {})
    if not isinstance(dist_tags, dict):
        raise RegistryError("Metadata payload has no dist-tags object")
    latest: object = cast("dict[str, Any]", dist_tags).get("latest")
    if not isinstance(latest, str) or not latest:
        raise RegistryError("Metadata payload has no latest dist-tag")
    return latest


def _parse_attestations(payload: dict[str, Any]) -> list[AttestationInfo]:
    """Extract attestation descriptors from the registry payload.

    Tolerates both documented payload shapes (top-level list, or a list
    nested under an ``attestations`` key) and skips malformed entries.
    """
    raw: object = payload.get("attestations", [])
    if raw is None:
        raw = []
    elif isinstance(raw, dict):
        raw = cast("dict[str, Any]", raw).get("attestations", [])
    if not isinstance(raw, list):
        raise RegistryError("Unexpected attestations payload shape")

    entries: list[AttestationInfo] = []
    for item in cast("list[object]", raw):
        if not isinstance(item, dict):
            continue
        entry = cast("dict[str, Any]", item)
        predicate: object = entry.get("predicateType", "")
        bundle: object = entry.get("bundle", "")
        media_type: object = (
            cast("dict[str, Any]", bundle).get("mediaType", "")
            if isinstance(bundle, dict)
            else ""
        )
        predicate_str = predicate if isinstance(predicate, str) else ""
        media_str = media_type if isinstance(media_type, str) else ""
        entries.append(
            AttestationInfo(
                predicate_type=predicate_str,
                bundle_media_type=media_str,
                is_provenance=predicate_str.startswith(PROVENANCE_PREDICATE_PREFIX),
            )
        )
    return entries


def verify_npm_package(
    package_ref: str,
    fetch_json: FetchJson = default_fetch_json,
) -> SupplyChainResult:
    """Verify the supply chain of an npm package version.

    Resolves the version (from the reference or ``dist-tags.latest``),
    then checks the npm registry for sigstore attestations published with
    that version. Never raises for registry-side outcomes; only
    :class:`InvalidPackageRefError` propagates for malformed input.
    """
    ref = parse_npm_ref(package_ref)

    try:
        metadata = fetch_json(metadata_url(ref.name))
    except RegistryNotFoundError:
        return SupplyChainResult(
            package=ref.name,
            version=ref.version or "unknown",
            status=SupplyChainStatus.NOT_FOUND,
            message=f"Package {ref.name} does not exist on the registry",
        )
    except RegistryError as e:
        return SupplyChainResult(
            package=ref.name,
            version=ref.version or "unknown",
            status=SupplyChainStatus.REGISTRY_ERROR,
            message=str(e),
        )

    version = ref.version
    if version is None:
        try:
            version = _extract_latest_version(metadata)
        except RegistryError as e:
            return SupplyChainResult(
                package=ref.name,
                version="unknown",
                status=SupplyChainStatus.REGISTRY_ERROR,
                message=str(e),
            )

    try:
        payload = fetch_json(attestations_url(ref.name, version))
    except RegistryNotFoundError:
        return SupplyChainResult(
            package=ref.name,
            version=version,
            status=SupplyChainStatus.UNSIGNED,
            message=(
                "No sigstore attestations published for "
                f"{ref.name}@{version} (no provenance)"
            ),
        )
    except RegistryError as e:
        return SupplyChainResult(
            package=ref.name,
            version=version,
            status=SupplyChainStatus.REGISTRY_ERROR,
            message=str(e),
        )

    try:
        attestations = _parse_attestations(payload)
    except RegistryError as e:
        return SupplyChainResult(
            package=ref.name,
            version=version,
            status=SupplyChainStatus.REGISTRY_ERROR,
            message=str(e),
        )

    if not attestations:
        return SupplyChainResult(
            package=ref.name,
            version=version,
            status=SupplyChainStatus.UNSIGNED,
            message=f"No sigstore attestations published for {ref.name}@{version}",
        )

    return SupplyChainResult(
        package=ref.name,
        version=version,
        status=SupplyChainStatus.SIGNED,
        attestations=attestations,
        has_provenance=any(a.is_provenance for a in attestations),
        message=f"{len(attestations)} attestation(s) published for {ref.name}@{version}",
    )
