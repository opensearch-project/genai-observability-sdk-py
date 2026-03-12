# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-20

### Changed
- **BREAKING**: Removed automatic AWS endpoint detection from `auth="auto"`
- Users must now explicitly specify `auth="sigv4"` for AWS SigV4 authentication
- Updated documentation and examples to reflect the new authentication behavior

### Added
- Comprehensive GitHub Actions workflows for CI/CD
- Security scanning with Bandit and Safety
- Dependency vulnerability checks with pip-audit
- Automated release workflow with PyPI publishing
- PR validation with conventional commits
- Dependabot configuration for dependency updates
- Issue and PR templates for better contribution workflow
- Proper package metadata and classifiers

### Fixed
- Improved repository structure for production readiness
- Updated all examples to use explicit AWS authentication

### Removed
- `_is_aws_endpoint()` function (no longer needed)

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of opensearch-genai-observability-sdk-py
- OTEL-native tracing with decorators (`@workflow`, `@task`, `@agent`, `@tool`)
- Auto-instrumentation for popular LLM libraries
- Scoring functionality with `score()` function
- AWS SigV4 authentication support
- HTTP and gRPC OTLP export support
- Comprehensive examples and documentation