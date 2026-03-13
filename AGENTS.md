# Agent Instructions

## Before committing

- Run `ruff format .` to auto-format all files
- Run `ruff check .` to lint
- Run `mypy src/opensearch_genai_observability_sdk_py --ignore-missing-imports` for type checking
- Run `pytest tests/` to verify tests pass

## Code style

- Line length: 100 (configured in pyproject.toml)
- Formatter: ruff
- Linter: ruff (rules: E, F, W, I, N, UP, S, B)

## Git

- Always sign off commits: `git commit --signoff`
- Never push to `upstream` remote (opensearch-project). Only push to `origin` (fork).
- Keep PRs small and incremental. One logical change per PR.

## Documentation

- When changing public API (function signatures, parameters, exports), cross-check README.md examples still work.
- Verify any code snippets in README.md reflect current API before submitting PR.
- When adding new APIs or functionality, include a concise usage example in README.md. No boilerplate or filler.
