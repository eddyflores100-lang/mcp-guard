# Changelog

## [Unreleased]

### Fixed
- `scan --fail-on low` no longer exits 1 on a zero-findings scan: an empty
  result now reads as below every threshold, so the lowest gate works as a
  "fail on anything at all" CI tripwire (#82)
- Write/destructive keyword classification now matches whole identifier segments
  and word boundaries: read-only names like `get_address`, `read_settings`,
  `get_clear_status` or `search_update_records` (and descriptions mentioning
  `created`/`settings`/`input`) no longer produce MCP001/MCP002/MCP006
  findings against safe servers, while `delete_repo`, `clear_cache`,
  `update_records`, camelCase and kebab-case names still match (#84)

### Added
- `MCP009` command execution detection: a third classification dimension
  (`is_command_execution`) for capabilities that run arbitrary code or spawn
  processes (`exec`, `execute`, `eval`, `spawn`, `shell`, `bash`, `cmd`,
  `command`, `subprocess`, `popen`, `terminal` — whole identifier segments with
  the same read-verb suppression and description inflection matching as
  MCP001/MCP002, so `get_exec_summary` stays unflagged). `MCP009` fires at HIGH
  when such a capability has no authentication, with message text that says the
  tool executes commands rather than "performs write operations". It is
  intentionally not coupled into MCP005/MCP006/MCP007: a shell tool has no
  meaningful "corresponding read" (#89)
- `MCP008` prompt injection detection for server and capability metadata, including
  every `inputSchema` key and string value, with SARIF/JSON classification properties
- `--strict-injection` flag and optional `canary` extra that add Little Canary's
  structural filter (no model or network calls)
- `mcp-guard verify` command and `supply_chain` module: npm supply chain
  verification via sigstore attestations / SLSA provenance, with
  `--policy strict` enforcement and JSON output (stdlib-only, offline tests)
- Initial release
