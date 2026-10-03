"""Tests for MCP Guard."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mcp_guard.models import (
    MCPCapability,
    MCPCapabilityType,
    MCPManifest,
    RiskFinding,
    RiskLevel,
    ScanResult,
)
from mcp_guard.parser import MCPParser
from mcp_guard.rules import (
    ALL_RULES,
    DestructiveWithoutConfirmationRule,
    ExcessivePermissionsRule,
    NoDescriptionRule,
    SecurityRule,
    UnauthenticatedDestructiveRule,
    UnauthenticatedWriteRule,
    WriteWithoutReadRule,
)
from mcp_guard.scanner import Scanner


class TestModels:
    """Test data models."""

    def test_risk_level_enum(self):
        assert RiskLevel.LOW.value == "LOW"
        assert RiskLevel.CRITICAL.value == "CRITICAL"

    def test_capability_creation(self):
        cap = MCPCapability(
            name="test_tool",
            type=MCPCapabilityType.TOOL,
            description="A test tool",
        )
        assert cap.name == "test_tool"
        assert cap.has_auth is False
        assert cap.is_destructive is False

    def test_manifest_creation(self):
        manifest = MCPManifest(
            name="test-server",
            version="1.0.0",
            capabilities=[
                MCPCapability(name="tool1", type=MCPCapabilityType.TOOL),
            ],
        )
        assert manifest.name == "test-server"
        assert len(manifest.capabilities) == 1

    def test_scan_result_summary(self):
        manifest = MCPManifest(
            name="test",
            capabilities=[
                MCPCapability(name="t1", type=MCPCapabilityType.TOOL),
                MCPCapability(name="t2", type=MCPCapabilityType.TOOL),
            ],
        )
        finding = RiskFinding(
            rule_id="MCP001",
            level=RiskLevel.HIGH,
            message="test",
            capability_name="t1",
            capability_type=MCPCapabilityType.TOOL,
            suggestion="fix",
        )
        result = ScanResult(
            manifest=manifest,
            findings=[finding],
        )
        assert result.summary["total_capabilities"] == 2
        assert result.summary["high"] == 1
        assert result.risk_score == RiskLevel.HIGH


class TestParser:
    """Test MCP manifest parser."""

    def test_from_dict_minimal(self):
        data = {
            "name": "test-server",
            "version": "1.0.0",
        }
        manifest = MCPParser.from_dict(data)
        assert manifest.name == "test-server"
        assert manifest.capabilities == []

    def test_from_dict_with_tools(self):
        data = {
            "name": "test",
            "tools": [
                {"name": "create_user", "description": "Create a user"},
            ],
        }
        manifest = MCPParser.from_dict(data)
        assert len(manifest.capabilities) == 1
        assert manifest.capabilities[0].name == "create_user"
        assert manifest.capabilities[0].is_write is True

    def test_from_dict_with_resources(self):
        data = {
            "name": "test",
            "resources": [
                {"name": "user_data", "description": "User data resource"},
            ],
        }
        manifest = MCPParser.from_dict(data)
        assert len(manifest.capabilities) == 1
        assert manifest.capabilities[0].type == MCPCapabilityType.RESOURCE

    def test_from_dict_with_prompts(self):
        data = {
            "name": "test",
            "prompts": [
                {"name": "greeting", "description": "Greeting prompt"},
            ],
        }
        manifest = MCPParser.from_dict(data)
        assert len(manifest.capabilities) == 1
        assert manifest.capabilities[0].type == MCPCapabilityType.PROMPT

    def test_from_json_file(self):
        data = {"name": "test", "tools": [{"name": "t1"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            manifest = MCPParser.from_json(f.name)
        assert manifest.name == "test"
        assert len(manifest.capabilities) == 1

    def test_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "mcp.json"
            config.write_text(json.dumps({"name": "test", "tools": [{"name": "t1"}]}))
            manifest = MCPParser.from_directory(tmpdir)
        assert manifest.name == "test"

    def test_from_directory_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                MCPParser.from_directory(tmpdir)


class TestKeywordBoundaryMatching:
    """Write/destructive keywords match whole words, not substrings (#84)."""

    @staticmethod
    def parse_tool(name: str, description: str) -> MCPCapability:
        manifest = MCPParser.from_dict(
            {"name": "srv", "tools": [{"name": name, "description": description}]}
        )
        return manifest.capabilities[0]

    def test_read_only_names_no_longer_match_substring_keywords(self):
        """The false positives reproduced in the issue stop matching."""
        for name in ("get_address", "read_settings", "search_update_records"):
            cap = self.parse_tool(name, "Return records")
            assert cap.is_write is False, name

    def test_get_clear_status_is_not_destructive(self):
        """A leading read verb marks the capability read-only."""
        cap = self.parse_tool("get_clear_status", "Return records")
        assert cap.is_destructive is False
        assert cap.is_write is False

    def test_description_substring_words_do_not_match(self):
        """'created'/'settings'/'input' no longer trip create/set/put."""
        cap = self.parse_tool("list_records", "Return the list of created records")
        assert cap.is_write is False

        cap = self.parse_tool("read_config", "Manage the settings for this server")
        assert cap.is_write is False

        cap = self.parse_tool("query_status", "Read the input and report")
        assert cap.is_write is False

    def test_description_word_boundary_hits_still_match(self):
        """Genuine keyword mentions in descriptions still classify."""
        cap = self.parse_tool("apply_config", "Set the temperature")
        assert cap.is_write is True

        cap = self.parse_tool("cleanup", "Delete all records")
        assert cap.is_destructive is True

    def test_true_positive_names_still_match(self):
        """Whole identifier segments keep matching their keywords."""
        for name, write, destructive in (
            ("delete_repo", False, True),
            ("clear_cache", False, True),
            ("create_user", True, False),
            ("update_records", True, False),
            ("add_user", True, False),
            ("delete", False, True),
        ):
            cap = self.parse_tool(name, "Operate on data")
            assert cap.is_write is write, name
            assert cap.is_destructive is destructive, name

    def test_camel_case_and_kebab_case_names_are_tokenized(self):
        """camelCase and kebab-case identifiers split into the same tokens."""
        cap = self.parse_tool("deleteFile", "Operate on data")
        assert cap.is_destructive is True

        cap = self.parse_tool("drop-table", "Operate on data")
        assert cap.is_destructive is True

    def test_read_verb_guard_does_not_suppress_description_matches(self):
        """The guard only suppresses name matches; descriptions still count."""
        cap = self.parse_tool("get_user", "Delete the user permanently")
        assert cap.is_destructive is True

        cap = self.parse_tool("get_user", "Get user by ID")
        assert cap.is_destructive is False
        assert cap.is_write is False

    def test_capability_without_name_uses_description_only(self):
        """A missing name degrades to description-only matching."""
        manifest = MCPParser.from_dict(
            {"name": "srv", "tools": [{"description": "Delete everything"}]}
        )
        assert manifest.capabilities[0].is_destructive is True

    def test_name_tokens_split_identifiers(self):
        """Tokenization covers snake, kebab, camel and single words."""
        assert MCPParser._name_tokens("get_address") == ["get", "address"]
        assert MCPParser._name_tokens("drop-in") == ["drop", "in"]
        assert MCPParser._name_tokens("deleteFile") == ["delete", "file"]
        assert MCPParser._name_tokens("delete") == ["delete"]

    def test_conjunction_reenables_name_matching(self):
        """A read verb + conjunction + keyword is a second operation."""
        for name in ("get_and_delete_user", "list_and_remove_items", "fetch_or_drop_table"):
            cap = self.parse_tool(name, "Operate on data")
            assert cap.is_destructive is True, name

    def test_conjunction_check_covers_every_hit(self):
        """The conjunction rule runs for every keyword hit, not just the first.

        `get_delete_and_remove_user` has its first keyword adjacent to the
        read verb, but the conjunction before the second one reveals the
        compound operation — it must still be flagged.
        """
        cap = self.parse_tool("get_delete_and_remove_user", "Operate on data")
        assert cap.is_destructive is True

    def test_then_and_ampersand_conjunctions(self):
        """`then` and `&` also separate a second operation."""
        for name in ("get_then_delete_user", "get_&_delete_user"):
            cap = self.parse_tool(name, "Operate on data")
            assert cap.is_destructive is True, name

    def test_no_conjunction_still_suppresses(self):
        """The adjacency rule keeps #84's read-only names suppressed."""
        for name in ("search_update_records", "get_clear_status", "get_address"):
            cap = self.parse_tool(name, "Return records")
            assert cap.is_destructive is False, name
            assert cap.is_write is False, name

    def test_description_third_person_inflections(self):
        """Present-tense descriptions are the common register (#88 review)."""
        cap = self.parse_tool("fetch_records", "Deletes all records")
        assert cap.is_destructive is True

        cap = self.parse_tool("fetch_records", "Updates the user settings and clears cache")
        assert cap.is_write is True
        assert cap.is_destructive is True

        cap = self.parse_tool("fetch_records", "Removes stale entries")
        assert cap.is_destructive is True

        cap = self.parse_tool("fetch_records", "Writes output to disk")
        assert cap.is_write is True

    def test_description_e_dropping_gerunds(self):
        """-ing forms of e-final verbs: deleting, updating, writing, removing."""
        cap = self.parse_tool("fetch_records", "Deleting all records first")
        assert cap.is_destructive is True

        cap = self.parse_tool("fetch_records", "Updating the configuration")
        assert cap.is_write is True

        cap = self.parse_tool("fetch_records", "Writing to disk")
        assert cap.is_write is True

        cap = self.parse_tool("fetch_records", "Removing stale entries")
        assert cap.is_destructive is True

    def test_description_regular_past_and_gerund(self):
        """Consonant-final verbs: cleared/posted/inserted and posting/inserting."""
        cap = self.parse_tool("fetch_records", "Clears the cache before returning")
        assert cap.is_destructive is True

        cap = self.parse_tool("fetch_records", "Cleared the cache before returning")
        assert cap.is_destructive is True

        cap = self.parse_tool("fetch_records", "Posting the result")
        assert cap.is_write is True

        cap = self.parse_tool("fetch_records", "Inserted the row")
        assert cap.is_write is True

    def test_description_cvc_doubling(self):
        """CVC verbs double their final consonant: dropping, putting, dropped."""
        cap = self.parse_tool("fetch_records", "Dropping all collections")
        assert cap.is_destructive is True

        cap = self.parse_tool("fetch_records", "Dropped the table")
        assert cap.is_destructive is True

        cap = self.parse_tool("fetch_records", "Putting the file in the bucket")
        assert cap.is_write is True

    def test_description_y_conjugation(self):
        """Consonant+y keywords: modifies/modified still match `modify`."""
        cap = self.parse_tool("fetch_records", "Modifies user records")
        assert cap.is_write is True

        cap = self.parse_tool("fetch_records", "Modified the configuration")
        assert cap.is_write is True

    def test_description_nouns_stay_unflagged(self):
        """The #84 noun protections survive the inflection patterns."""
        cap = self.parse_tool("list_records", "Checks the created date")
        assert cap.is_write is False

        cap = self.parse_tool("read_config", "Manage the settings for this server")
        assert cap.is_write is False

        cap = self.parse_tool("read_config", "Returns the current setting")
        assert cap.is_write is False

        cap = self.parse_tool("query_status", "Read the input and report")
        assert cap.is_write is False


class TestRules:
    """Test security rules."""

    def test_unauthenticated_write(self):
        rule = UnauthenticatedWriteRule()
        cap = MCPCapability(
            name="create_user",
            type=MCPCapabilityType.TOOL,
            is_write=True,
            has_auth=False,
        )
        manifest = MCPManifest(name="test", capabilities=[cap])
        findings = rule.check(cap, manifest)
        assert len(findings) == 1
        assert findings[0].level == RiskLevel.HIGH

    def test_authenticated_write_no_finding(self):
        rule = UnauthenticatedWriteRule()
        cap = MCPCapability(
            name="create_user",
            type=MCPCapabilityType.TOOL,
            is_write=True,
            has_auth=True,
        )
        manifest = MCPManifest(name="test", capabilities=[cap])
        findings = rule.check(cap, manifest)
        assert len(findings) == 0

    def test_unauthenticated_destructive(self):
        rule = UnauthenticatedDestructiveRule()
        cap = MCPCapability(
            name="delete_user",
            type=MCPCapabilityType.TOOL,
            is_destructive=True,
            has_auth=False,
        )
        manifest = MCPManifest(name="test", capabilities=[cap])
        findings = rule.check(cap, manifest)
        assert len(findings) == 1
        assert findings[0].level == RiskLevel.CRITICAL

    def test_excessive_permissions(self):
        rule = ExcessivePermissionsRule()
        cap = MCPCapability(
            name="admin_action",
            type=MCPCapabilityType.TOOL,
            permissions=["read", "write", "delete", "admin", "sudo", "root"],
        )
        manifest = MCPManifest(name="test", capabilities=[cap])
        findings = rule.check(cap, manifest)
        assert len(findings) == 1
        assert findings[0].level == RiskLevel.MEDIUM

    def test_no_description(self):
        rule = NoDescriptionRule()
        cap = MCPCapability(
            name="some_tool",
            type=MCPCapabilityType.TOOL,
            description="",
        )
        manifest = MCPManifest(name="test", capabilities=[cap])
        findings = rule.check(cap, manifest)
        assert len(findings) == 1
        assert findings[0].level == RiskLevel.LOW

    def test_short_description(self):
        rule = NoDescriptionRule()
        cap = MCPCapability(
            name="some_tool",
            type=MCPCapabilityType.TOOL,
            description="Short",
        )
        manifest = MCPManifest(name="test", capabilities=[cap])
        findings = rule.check(cap, manifest)
        assert len(findings) == 1

    def test_write_without_read(self):
        rule = WriteWithoutReadRule()
        cap = MCPCapability(
            name="create_user",
            type=MCPCapabilityType.TOOL,
            is_write=True,
        )
        manifest = MCPManifest(name="test", capabilities=[cap])
        findings = rule.check(cap, manifest)
        assert len(findings) == 1
        assert findings[0].level == RiskLevel.MEDIUM

    def test_write_with_read_no_finding(self):
        rule = WriteWithoutReadRule()
        write_cap = MCPCapability(
            name="create_user",
            type=MCPCapabilityType.TOOL,
            is_write=True,
        )
        read_cap = MCPCapability(
            name="get_user",
            type=MCPCapabilityType.TOOL,
        )
        manifest = MCPManifest(name="test", capabilities=[write_cap, read_cap])
        findings = rule.check(write_cap, manifest)
        assert len(findings) == 0

    def test_write_with_read_multi_underscore_prefix_match(self):
        rule = WriteWithoutReadRule()
        write_cap = MCPCapability(
            name="create_user_profile",
            type=MCPCapabilityType.TOOL,
            is_write=True,
        )
        read_cap = MCPCapability(
            name="get_user_profile_settings",
            type=MCPCapabilityType.TOOL,
        )
        manifest = MCPManifest(name="test", capabilities=[write_cap, read_cap])
        findings = rule.check(write_cap, manifest)
        assert len(findings) == 0

    def test_write_with_read_multi_underscore_exact_match(self):
        rule = WriteWithoutReadRule()
        write_cap = MCPCapability(
            name="create_user_profile",
            type=MCPCapabilityType.TOOL,
            is_write=True,
        )
        read_cap = MCPCapability(
            name="get_user_profile",
            type=MCPCapabilityType.TOOL,
        )
        manifest = MCPManifest(name="test", capabilities=[write_cap, read_cap])
        findings = rule.check(write_cap, manifest)
        assert len(findings) == 0

    def test_write_with_read_multi_underscore_unmatched(self):
        rule = WriteWithoutReadRule()
        write_cap = MCPCapability(
            name="create_user_profile",
            type=MCPCapabilityType.TOOL,
            is_write=True,
        )
        read_cap = MCPCapability(
            name="get_organization",
            type=MCPCapabilityType.TOOL,
        )
        manifest = MCPManifest(name="test", capabilities=[write_cap, read_cap])
        findings = rule.check(write_cap, manifest)
        assert len(findings) == 1

    def test_destructive_without_confirmation(self):
        rule = DestructiveWithoutConfirmationRule()
        cap = MCPCapability(
            name="delete_all",
            type=MCPCapabilityType.TOOL,
            is_destructive=True,
            input_schema={"properties": {"name": {"type": "string"}}},
        )
        manifest = MCPManifest(name="test", capabilities=[cap])
        findings = rule.check(cap, manifest)
        assert len(findings) == 1
        assert findings[0].level == RiskLevel.HIGH

    def test_destructive_with_confirmation(self):
        rule = DestructiveWithoutConfirmationRule()
        cap = MCPCapability(
            name="delete_all",
            type=MCPCapabilityType.TOOL,
            is_destructive=True,
            input_schema={"properties": {"confirm": {"type": "boolean"}}},
        )
        manifest = MCPManifest(name="test", capabilities=[cap])
        findings = rule.check(cap, manifest)
        assert len(findings) == 0


class TestScanner:
    """Test scanner engine."""

    def test_scan_clean_manifest(self):
        manifest = MCPManifest(
            name="clean-server",
            capabilities=[
                MCPCapability(
                    name="get_status",
                    type=MCPCapabilityType.TOOL,
                    description="Get server status",
                ),
            ],
        )
        scanner = Scanner()
        result = scanner.scan(manifest)
        assert result.risk_score == RiskLevel.LOW
        assert len(result.findings) == 0

    def test_scan_risky_manifest(self):
        manifest = MCPManifest(
            name="risky-server",
            capabilities=[
                MCPCapability(
                    name="delete_database",
                    type=MCPCapabilityType.TOOL,
                    description="Delete the entire database",
                    is_destructive=True,
                    is_write=True,
                    has_auth=False,
                ),
            ],
        )
        scanner = Scanner()
        result = scanner.scan(manifest)
        assert result.risk_score == RiskLevel.CRITICAL
        assert len(result.findings) >= 2

    def test_scan_multiple_capabilities(self):
        manifest = MCPManifest(
            name="multi-server",
            capabilities=[
                MCPCapability(
                    name="get_user",
                    type=MCPCapabilityType.TOOL,
                    description="Get user by ID",
                ),
                MCPCapability(
                    name="create_user",
                    type=MCPCapabilityType.TOOL,
                    description="Create a new user",
                    is_write=True,
                    has_auth=True,
                ),
                MCPCapability(
                    name="delete_user",
                    type=MCPCapabilityType.TOOL,
                    description="Delete a user",
                    is_destructive=True,
                    is_write=True,
                    has_auth=True,
                    input_schema={"properties": {"confirm": {"type": "boolean"}}},
                ),
            ],
        )
        scanner = Scanner()
        result = scanner.scan(manifest)
        assert result.summary["total_capabilities"] == 3


class TestFormatters:
    """Test output formatters."""

    def test_to_dict(self):
        manifest = MCPManifest(name="test", capabilities=[])
        result = ScanResult(manifest=manifest, findings=[])
        d = result.summary
        assert d["total_capabilities"] == 0

    def test_to_json_output(self):
        manifest = MCPManifest(name="test", capabilities=[])
        result = ScanResult(manifest=manifest, findings=[])
        from mcp_guard.formatters import to_json
        json_str = to_json(result)
        parsed = json.loads(json_str)
        assert parsed["server"]["name"] == "test"

    def test_to_sarif(self):
        manifest = MCPManifest(
            name="test",
            capabilities=[
                MCPCapability(
                    name="delete_all",
                    type=MCPCapabilityType.TOOL,
                    is_destructive=True,
                    has_auth=False,
                ),
            ],
        )
        scanner = Scanner()
        result = scanner.scan(manifest)
        from mcp_guard.formatters import to_sarif
        sarif = to_sarif(result)
        assert sarif["version"] == "2.1.0"
        assert len(sarif["runs"][0]["results"]) > 0


class TestEndToEnd:
    """End-to-end tests."""

    def test_full_scan_pipeline(self):
        """Test complete pipeline: dict -> parse -> scan -> format."""
        data = {
            "name": "e2e-test",
            "version": "1.0.0",
            "tools": [
                {
                    "name": "get_item",
                    "description": "Get an item by ID",
                },
                {
                    "name": "create_item",
                    "description": "Create a new item",
                    "inputSchema": {
                        "properties": {
                            "name": {"type": "string"},
                        },
                    },
                },
                {
                    "name": "delete_item",
                    "description": "Delete an item permanently",
                    "inputSchema": {
                        "properties": {
                            "id": {"type": "string"},
                        },
                    },
                },
            ],
        }

        manifest = MCPParser.from_dict(data)
        assert len(manifest.capabilities) == 3

        scanner = Scanner()
        result = scanner.scan(manifest)

        # delete_item should trigger destructive without auth + without confirmation
        destructive_findings = [
            f for f in result.findings if f.capability_name == "delete_item"
        ]
        assert len(destructive_findings) >= 1

        # Verify JSON output works
        from mcp_guard.formatters import to_json
        json_str = to_json(result)
        parsed = json.loads(json_str)
        assert parsed["server"]["name"] == "e2e-test"


class TestScannerCustomRules:
    """Test Scanner.add_rule and custom rule registration."""

    def test_add_rule_registers_custom_rule(self):
        """A custom rule added to a scanner is applied during scan."""

        class AlwaysFlagRule(SecurityRule):
            rule_id = "CUSTOM001"
            description = "Always flags a capability"

            def check(
                self, capability: MCPCapability, manifest: MCPManifest
            ) -> list[RiskFinding]:
                return [
                    RiskFinding(
                        rule_id=self.rule_id,
                        level=RiskLevel.HIGH,
                        message=f"Custom rule matched {capability.name}",
                        capability_name=capability.name,
                    )
                ]

        manifest = MCPParser.from_dict(
            {
                "name": "custom-test",
                "tools": [{"name": "get_data", "description": "Get data from API"}],
            }
        )
        scanner = Scanner(rules=[])
        assert scanner.scan(manifest).findings == []

        scanner.add_rule(AlwaysFlagRule())
        findings = scanner.scan(manifest).findings

        assert [f.rule_id for f in findings] == ["CUSTOM001"]

    def test_default_rules_are_used_when_none_given(self):
        """Constructing Scanner without rules still applies ALL_RULES."""
        scanner = Scanner()
        assert len(scanner.rules) == len(ALL_RULES)


class TestBaseSecurityRule:
    """Test the abstract SecurityRule contract."""

    def test_base_check_raises_not_implemented(self):
        """SecurityRule.check must be overridden by subclasses."""
        manifest = MCPParser.from_dict({"name": "s", "tools": []})
        capability = MCPCapability(
            name="t", type=MCPCapabilityType.TOOL, description="d"
        )

        with pytest.raises(NotImplementedError):
            SecurityRule().check(capability, manifest)
