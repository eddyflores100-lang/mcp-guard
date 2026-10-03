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
- `MCP008` prompt injection detection for server and capability metadata, including
  every `inputSchema` key and string value, with SARIF/JSON classification properties
- `--strict-injection` flag and optional `canary` extra that add Little Canary's
  structural filter (no model or network calls)
- Initial release
