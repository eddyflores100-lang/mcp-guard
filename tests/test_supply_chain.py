"""Tests for supply chain verification (npm attestations / provenance)."""

from __future__ import annotations

import io
import json
from typing import Any, cast
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from mcp_guard.cli import main
from mcp_guard.supply_chain import (
    NPM_REGISTRY,
    InvalidPackageRefError,
    RegistryError,
    RegistryNotFoundError,
    SupplyChainResult,
    SupplyChainStatus,
    default_fetch_json,
    manifest_url,
    metadata_url,
    parse_npm_ref,
    verify_npm_package,
)

#: Attestations URL exactly as the live registry advertises it in a
#: version manifest's ``dist.attestations.url`` (percent-encoded scope).
SIGNED_ATTESTATIONS_URL = f"{NPM_REGISTRY}/-/npm/v1/attestations/signed-pkg@1.2.3"

SIGNED_MANIFEST = {
    "name": "signed-pkg",
    "version": "1.2.3",
    "dist": {
        "tarball": "https://registry.npmjs.org/signed-pkg/-/signed-pkg-1.2.3.tgz",
        "attestations": {
            "url": SIGNED_ATTESTATIONS_URL,
            "provenance": {"predicateType": "https://slsa.dev/provenance/v1"},
        },
    },
}

SIGNED_PACKUMENT = {
    "name": "signed-pkg",
    "dist-tags": {"latest": "1.2.3"},
    "versions": {"1.2.3": SIGNED_MANIFEST},
}

SIGNED_ATTESTATIONS = {
    "attestations": [
        {
            "predicateType": "https://slsa.dev/provenance/v0.2",
            "bundle": {"mediaType": "application/vnd.dev.sigstore.bundle+json;type=0.1"},
        },
        {
            "predicateType": "https://npmjs.org/attestations/ci/v1",
            "bundle": {"mediaType": "application/vnd.dev.sigstore.bundle+json;type=0.1"},
        },
    ]
}


class FakeRegistry:
    """Fake JSON fetcher routing URLs to canned registry responses.

    The routing mirrors the real contract: the discovered attestations
    endpoint is always under ``/-/npm/v1/attestations/``, while the
    packument (``GET /{name}``) and the version manifest
    (``GET /{name}/{version}``) are served from the ``metadata`` fixture —
    exactly one of them is requested per verification flow.
    """

    def __init__(
        self,
        metadata: dict[str, Any] | Exception = SIGNED_PACKUMENT,
        attestations: dict[str, Any] | Exception | None = SIGNED_ATTESTATIONS,
    ) -> None:
        self.metadata = metadata
        self.attestations = attestations
        self.urls: list[str] = []

    def __call__(self, url: str) -> dict[str, Any]:
        self.urls.append(url)
        if "/-/npm/v1/attestations/" in url:
            if isinstance(self.attestations, Exception):
                raise self.attestations
            return self.attestations or {}
        if isinstance(self.metadata, Exception):
            raise self.metadata
        return self.metadata


class TestParseNpmRef:
    """Test npm package reference parsing."""

    @pytest.mark.parametrize(
        ("ref", "name", "version"),
        [
            ("pkg", "pkg", None),
            ("pkg@1.0.0", "pkg", "1.0.0"),
            ("pkg@1.0.0-beta.1", "pkg", "1.0.0-beta.1"),
            ("@scope/pkg", "@scope/pkg", None),
            ("@scope/pkg@2.0.0", "@scope/pkg", "2.0.0"),
            ("@scope/pkg@1.0.0+build.5", "@scope/pkg", "1.0.0+build.5"),
            ("  pkg  ", "pkg", None),
        ],
    )
    def test_valid_refs(self, ref: str, name: str, version: str | None) -> None:
        parsed = parse_npm_ref(ref)
        assert parsed.name == name
        assert parsed.version == version

    @pytest.mark.parametrize(
        "ref",
        [
            "",
            "   ",
            "with space",
            "pkg@",
            "pkg@v 1",
            "pkg@1.0.0@x",
            "@scope@1.0.0",
            "@scopeonly",
            "@/pkg",
            "@scope/",
            "@/pkg@1.0.0",
            "@scope/@1.0.0",
            "a/b",
            "pkg@1.0.0;rm -rf",
        ],
    )
    def test_invalid_refs(self, ref: str) -> None:
        with pytest.raises(InvalidPackageRefError):
            parse_npm_ref(ref)


class TestUrls:
    """Test registry URL construction."""

    def test_metadata_url_unscoped(self) -> None:
        assert metadata_url("pkg") == f"{NPM_REGISTRY}/pkg"

    def test_metadata_url_scoped(self) -> None:
        assert metadata_url("@scope/pkg") == f"{NPM_REGISTRY}/@scope/pkg"

    def test_manifest_url_unscoped(self) -> None:
        assert manifest_url("pkg", "1.0.0") == f"{NPM_REGISTRY}/pkg/1.0.0"

    def test_manifest_url_scoped(self) -> None:
        assert manifest_url("@scope/pkg", "2.0.0") == (f"{NPM_REGISTRY}/@scope/pkg/2.0.0")


class TestVerifyNpmPackage:
    """Test the verification state machine against a fake registry."""

    def test_signed_with_provenance(self) -> None:
        result = verify_npm_package("signed-pkg", fetch_json=FakeRegistry())
        assert result.status is SupplyChainStatus.SIGNED
        assert result.version == "1.2.3"
        assert result.has_provenance is True
        assert len(result.attestations) == 2
        assert result.attestations[0].is_provenance is True
        assert result.attestations[1].is_provenance is False

    def test_signed_without_provenance(self) -> None:
        payload = {
            "attestations": [
                {"predicateType": "https://npmjs.org/attestations/ci/v1", "bundle": {}},
            ]
        }
        result = verify_npm_package("signed-pkg", fetch_json=FakeRegistry(attestations=payload))
        assert result.status is SupplyChainStatus.SIGNED
        assert result.has_provenance is False
        assert result.attestations[0].bundle_media_type == ""

    def test_unsigned_manifest_makes_single_request(self) -> None:
        """No ``dist.attestations`` in the manifest: unsigned, no 2nd request.

        The registry decides via the manifest, so an unsigned version is
        answered from the manifest alone (the maintainer's #86 review:
        "if ``dist.attestations`` is absent … you never make the second
        request").
        """
        unsigned_manifest = {"name": "pkg", "dist": {"tarball": "https://x/pkg.tgz"}}
        packument = {
            "dist-tags": {"latest": "1.0.0"},
            "versions": {"1.0.0": unsigned_manifest},
        }
        registry = FakeRegistry(
            metadata=packument, attestations=RegistryNotFoundError("never requested")
        )
        result = verify_npm_package("pkg", fetch_json=cast("Any", registry))
        assert result.status is SupplyChainStatus.UNSIGNED
        assert "No sigstore attestations" in result.message
        assert registry.urls == [f"{NPM_REGISTRY}/pkg"]

    def test_second_request_uses_manifest_discovered_url(self) -> None:
        """The attestations request is the URL the manifest advertised.

        Asserts the second request rather than assuming it (maintainer's
        #86 review): it must be the discovered path, not a constructed
        ``/-/npm/v1/packages/attestations/{name}/{version}`` URL.
        """
        registry = FakeRegistry()
        verify_npm_package("signed-pkg", fetch_json=registry)
        assert registry.urls[0] == f"{NPM_REGISTRY}/signed-pkg"
        assert registry.urls[1] == SIGNED_ATTESTATIONS_URL

    def test_attestations_url_host_is_not_followed(self) -> None:
        """Only the discovered URL's path is kept, on the registry base.

        pacote's normalization: a manifest advertising attestations on a
        third-party host must not redirect the fetch there.
        """
        manifest = {
            "dist": {
                "attestations": {
                    "url": "https://evil.example.com/-/npm/v1/attestations/signed-pkg@1.2.3"
                }
            }
        }
        registry = FakeRegistry(metadata=manifest)
        result = verify_npm_package("signed-pkg@1.2.3", fetch_json=cast("Any", registry))
        assert result.status is SupplyChainStatus.SIGNED
        assert registry.urls == [
            f"{NPM_REGISTRY}/signed-pkg/1.2.3",
            f"{NPM_REGISTRY}/-/npm/v1/attestations/signed-pkg@1.2.3",
        ]

    def test_manifest_without_dist_is_unsigned(self) -> None:
        registry = FakeRegistry(metadata={"name": "pkg"})
        result = verify_npm_package("pkg@1.0.0", fetch_json=cast("Any", registry))
        assert result.status is SupplyChainStatus.UNSIGNED

    def test_registry_error_on_pinned_manifest_fetch(self) -> None:
        registry = FakeRegistry(metadata=RegistryError("boom"))
        result = verify_npm_package("pkg@1.0.0", fetch_json=registry)
        assert result.status is SupplyChainStatus.REGISTRY_ERROR
        assert result.message == "boom"

    def test_attestations_url_without_path_is_registry_error(self) -> None:
        manifest = {"dist": {"attestations": {"url": "https://registry.npmjs.org"}}}
        registry = FakeRegistry(metadata=manifest)
        result = verify_npm_package("pkg@1.0.0", fetch_json=cast("Any", registry))
        assert result.status is SupplyChainStatus.REGISTRY_ERROR
        assert "no path" in result.message
        assert registry.urls == [f"{NPM_REGISTRY}/pkg/1.0.0"]

    def test_malformed_dist_attestations_is_unsigned(self) -> None:
        for dist in ({"attestations": "garbage"}, {"attestations": {"url": 42}}):
            manifest = {"dist": dist}
            registry = FakeRegistry(metadata=manifest)
            result = verify_npm_package("pkg@1.0.0", fetch_json=cast("Any", registry))
            assert result.status is SupplyChainStatus.UNSIGNED, dist

    def test_registry_error_when_manifest_promises_attestations_but_404(self) -> None:
        """A manifest-advertised URL that 404s is a registry inconsistency.

        Unlike a manifest without ``dist.attestations`` (a definitive
        unsigned verdict), a broken promise is reported as an error so CI
        can distinguish "unsigned" from "registry said one thing and
        served another".
        """
        registry = FakeRegistry(attestations=RegistryNotFoundError("x"))
        result = verify_npm_package("signed-pkg", fetch_json=registry)
        assert result.status is SupplyChainStatus.REGISTRY_ERROR
        assert "404" in result.message

    def test_unsigned_when_attestations_list_empty(self) -> None:
        result = verify_npm_package(
            "signed-pkg", fetch_json=FakeRegistry(attestations={"attestations": []})
        )
        assert result.status is SupplyChainStatus.UNSIGNED

    def test_not_found(self) -> None:
        result = verify_npm_package(
            "ghost-pkg", fetch_json=FakeRegistry(metadata=RegistryNotFoundError("x"))
        )
        assert result.status is SupplyChainStatus.NOT_FOUND
        assert result.version == "unknown"

    def test_not_found_when_pinned_version_missing(self) -> None:
        """A pinned version that 404s is not_found, not unsigned."""
        registry = FakeRegistry(metadata=RegistryNotFoundError("x"))
        result = verify_npm_package("signed-pkg@9.9.9", fetch_json=registry)
        assert result.status is SupplyChainStatus.NOT_FOUND
        assert result.version == "9.9.9"
        assert registry.urls == [f"{NPM_REGISTRY}/signed-pkg/9.9.9"]

    def test_pinned_version_fetches_manifest_directly(self) -> None:
        registry = FakeRegistry(metadata=SIGNED_MANIFEST)
        result = verify_npm_package("signed-pkg@1.2.3", fetch_json=registry)
        assert result.status is SupplyChainStatus.SIGNED
        assert result.version == "1.2.3"
        assert registry.urls[0] == f"{NPM_REGISTRY}/signed-pkg/1.2.3"

    def test_registry_error_on_metadata(self) -> None:
        result = verify_npm_package("pkg", fetch_json=FakeRegistry(metadata=RegistryError("boom")))
        assert result.status is SupplyChainStatus.REGISTRY_ERROR
        assert result.message == "boom"

    def test_registry_error_on_dist_tags(self) -> None:
        result = verify_npm_package("pkg", fetch_json=FakeRegistry(metadata={"x": 1}))
        assert result.status is SupplyChainStatus.REGISTRY_ERROR

    def test_registry_error_on_missing_versions_object(self) -> None:
        metadata = {"dist-tags": {"latest": "1.0.0"}}
        result = verify_npm_package("pkg", fetch_json=FakeRegistry(metadata=metadata))
        assert result.status is SupplyChainStatus.REGISTRY_ERROR
        assert "versions" in result.message

    def test_registry_error_when_latest_missing_from_versions(self) -> None:
        metadata = {"dist-tags": {"latest": "1.0.0"}, "versions": {"0.9.0": {}}}
        result = verify_npm_package("pkg", fetch_json=FakeRegistry(metadata=metadata))
        assert result.status is SupplyChainStatus.REGISTRY_ERROR
        assert "1.0.0" in result.message

    def test_registry_error_on_attestations(self) -> None:
        result = verify_npm_package(
            "pkg", fetch_json=FakeRegistry(attestations=RegistryError("net down"))
        )
        assert result.status is SupplyChainStatus.REGISTRY_ERROR
        assert result.message == "net down"

    def test_registry_error_on_malformed_attestations(self) -> None:
        result = verify_npm_package(
            "pkg", fetch_json=FakeRegistry(attestations={"attestations": "garbage"})
        )
        assert result.status is SupplyChainStatus.REGISTRY_ERROR

    def test_malformed_entries_are_skipped(self) -> None:
        payload = {
            "attestations": [
                "not-a-dict",
                {"predicateType": 42, "bundle": "not-a-dict-either"},
                {"predicateType": "https://slsa.dev/provenance/v1"},
            ]
        }
        result = verify_npm_package("pkg", fetch_json=FakeRegistry(attestations=payload))
        assert result.status is SupplyChainStatus.SIGNED
        # Non-dict entries are skipped; dict entries with unexpected types are coerced.
        assert len(result.attestations) == 2
        assert result.attestations[0].predicate_type == ""
        assert result.attestations[1].is_provenance is True

    def test_empty_latest_dist_tag_is_registry_error(self) -> None:
        metadata = {"dist-tags": {"latest": ""}}
        result = verify_npm_package("pkg", fetch_json=FakeRegistry(metadata=metadata))
        assert result.status is SupplyChainStatus.REGISTRY_ERROR
        assert "latest" in result.message

    def test_non_dict_dist_tags_is_registry_error(self) -> None:
        metadata = {"dist-tags": "garbage"}
        result = verify_npm_package("pkg", fetch_json=FakeRegistry(metadata=metadata))
        assert result.status is SupplyChainStatus.REGISTRY_ERROR
        assert "dist-tags" in result.message

    def test_attestations_payload_without_key_is_unsigned(self) -> None:
        result = verify_npm_package("pkg", fetch_json=FakeRegistry(attestations={"unrelated": 1}))
        assert result.status is SupplyChainStatus.UNSIGNED

    def test_null_attestations_is_unsigned(self) -> None:
        result = verify_npm_package(
            "pkg", fetch_json=FakeRegistry(attestations={"attestations": None})
        )
        assert result.status is SupplyChainStatus.UNSIGNED

    def test_invalid_version_charset_is_rejected(self) -> None:
        with pytest.raises(InvalidPackageRefError):
            parse_npm_ref("pkg@1.0.0!")

    def test_nested_attestations_shape(self) -> None:
        payload = {
            "attestations": {
                "attestations": [
                    {"predicateType": "https://slsa.dev/provenance/v0.2", "bundle": {}},
                ]
            }
        }
        result = verify_npm_package("pkg", fetch_json=FakeRegistry(attestations=payload))
        assert result.status is SupplyChainStatus.SIGNED
        assert result.has_provenance is True

    def test_two_requests_issued(self) -> None:
        registry = FakeRegistry()
        verify_npm_package("signed-pkg", fetch_json=registry)
        assert len(registry.urls) == 2
        assert registry.urls[1] == SIGNED_ATTESTATIONS_URL


class _FakeResponse:
    """Minimal context-manager response object for urlopen patching."""

    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)

    def read(self) -> bytes:
        return self._stream.read()

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class TestDefaultFetchJson:
    """Test the stdlib fetcher against patched urlopen."""

    def test_returns_dict(self) -> None:
        with patch("mcp_guard.supply_chain.urlopen", return_value=_FakeResponse(b'{"a": 1}')):
            assert default_fetch_json("https://x") == {"a": 1}

    def test_404_raises_not_found(self) -> None:
        import urllib.error

        error = urllib.error.HTTPError("u", 404, "Not Found", None, io.BytesIO(b""))
        with (
            patch("mcp_guard.supply_chain.urlopen", side_effect=error),
            pytest.raises(RegistryNotFoundError),
        ):
            default_fetch_json("https://x")

    def test_http_error_raises_registry_error(self) -> None:
        import urllib.error

        error = urllib.error.HTTPError("u", 503, "Unavailable", None, io.BytesIO(b""))
        with (
            patch("mcp_guard.supply_chain.urlopen", side_effect=error),
            pytest.raises(RegistryError),
        ):
            default_fetch_json("https://x")

    def test_oserror_raises_registry_error(self) -> None:
        with (
            patch("mcp_guard.supply_chain.urlopen", side_effect=OSError("dns fail")),
            pytest.raises(RegistryError),
        ):
            default_fetch_json("https://x")

    def test_non_json_raises_registry_error(self) -> None:
        with (
            patch("mcp_guard.supply_chain.urlopen", return_value=_FakeResponse(b"not json")),
            pytest.raises(RegistryError),
        ):
            default_fetch_json("https://x")

    def test_non_object_json_raises_registry_error(self) -> None:
        with (
            patch("mcp_guard.supply_chain.urlopen", return_value=_FakeResponse(b"[1, 2]")),
            pytest.raises(RegistryError),
        ):
            default_fetch_json("https://x")


class TestVerifyCLI:
    """Test the `mcp-guard verify` command."""

    def _result(self, status: SupplyChainStatus, **kw: Any) -> SupplyChainResult:
        defaults: dict[str, Any] = {
            "package": "cli-pkg",
            "version": "1.0.0",
            "status": status,
        }
        defaults.update(kw)
        return SupplyChainResult(**defaults)

    def test_signed_report_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "mcp_guard.cli.verify_npm_package",
            lambda ref: self._result(SupplyChainStatus.SIGNED, message="ok"),
        )
        runner = CliRunner()
        outcome = runner.invoke(main, ["verify", "cli-pkg"])
        assert outcome.exit_code == 0
        assert "cli-pkg" in outcome.output

    def test_signed_strict_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "mcp_guard.cli.verify_npm_package",
            lambda ref: self._result(SupplyChainStatus.SIGNED),
        )
        runner = CliRunner()
        outcome = runner.invoke(main, ["verify", "cli-pkg", "--policy", "strict"])
        assert outcome.exit_code == 0

    def test_unsigned_report_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "mcp_guard.cli.verify_npm_package",
            lambda ref: self._result(SupplyChainStatus.UNSIGNED, message="no attestations"),
        )
        runner = CliRunner()
        outcome = runner.invoke(main, ["verify", "cli-pkg"])
        assert outcome.exit_code == 0
        assert "unsigned" in outcome.output

    def test_unsigned_strict_exits_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "mcp_guard.cli.verify_npm_package",
            lambda ref: self._result(SupplyChainStatus.UNSIGNED),
        )
        runner = CliRunner()
        outcome = runner.invoke(main, ["verify", "cli-pkg", "--policy", "strict"])
        assert outcome.exit_code == 1

    def test_not_found_exits_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "mcp_guard.cli.verify_npm_package",
            lambda ref: self._result(
                SupplyChainStatus.NOT_FOUND, version="unknown", message="missing"
            ),
        )
        runner = CliRunner()
        outcome = runner.invoke(main, ["verify", "ghost"])
        assert outcome.exit_code == 1
        assert "not_found" in outcome.output

    def test_registry_error_exits_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "mcp_guard.cli.verify_npm_package",
            lambda ref: self._result(
                SupplyChainStatus.REGISTRY_ERROR, version="unknown", message="boom"
            ),
        )
        runner = CliRunner()
        outcome = runner.invoke(main, ["verify", "pkg"])
        assert outcome.exit_code == 1

    def test_json_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "mcp_guard.cli.verify_npm_package",
            lambda ref: self._result(
                SupplyChainStatus.SIGNED,
                message="1 attestation(s)",
            ),
        )
        runner = CliRunner()
        outcome = runner.invoke(main, ["verify", "cli-pkg", "--format", "json"])
        assert outcome.exit_code == 0
        payload = json.loads(outcome.output)
        assert payload["status"] == "signed"
        assert payload["package"] == "cli-pkg"

    def test_invalid_ref_exits_error(self) -> None:
        runner = CliRunner()
        outcome = runner.invoke(main, ["verify", "not a ref"])
        assert outcome.exit_code == 1
        assert "Error" in outcome.output

    def test_signed_prints_provenance_lines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mcp_guard.supply_chain import AttestationInfo

        monkeypatch.setattr(
            "mcp_guard.cli.verify_npm_package",
            lambda ref: self._result(
                SupplyChainStatus.SIGNED,
                attestations=[
                    AttestationInfo(
                        predicate_type="https://slsa.dev/provenance/v0.2",
                        bundle_media_type="application/vnd.dev.sigstore.bundle+json",
                        is_provenance=True,
                    )
                ],
                has_provenance=True,
                message="1 attestation(s) published",
            ),
        )
        runner = CliRunner()
        outcome = runner.invoke(main, ["verify", "cli-pkg"])
        assert outcome.exit_code == 0
        assert "Provenance" in outcome.output
        assert "slsa.dev" in outcome.output


class TestResultModel:
    """Test SupplyChainResult serialization defaults."""

    def test_defaults(self) -> None:
        result = SupplyChainResult(package="p", version="1.0.0", status=SupplyChainStatus.UNSIGNED)
        assert result.attestations == []
        assert result.has_provenance is False
        assert result.message == ""
