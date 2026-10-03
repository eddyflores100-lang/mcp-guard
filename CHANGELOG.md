# Changelog

## [Unreleased]

### Added
- `MCP008` prompt injection detection for server and capability metadata, including
  every `inputSchema` key and string value, with SARIF/JSON classification properties
- `--strict-injection` flag and optional `canary` extra that add Little Canary's
  structural filter (no model or network calls)
- `mcp-guard verify` command and `supply_chain` module: npm supply chain
  verification via sigstore attestations / SLSA provenance, with
  `--policy strict` enforcement and JSON output (stdlib-only, offline tests)
- Initial release
