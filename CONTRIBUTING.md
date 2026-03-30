# Contributing to OpenSearch GenAI SDK

Thank you for your interest in contributing to the OpenSearch GenAI SDK! We welcome contributions from the community.

## Code of Conduct

This project adheres to the OpenSearch [Code of Conduct](https://opensearch.org/codeofconduct.html). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/genai-observability-sdk-py.git
   cd genai-observability-sdk-py
   ```

3. Install the development dependencies:
   ```bash
   pip install -e ".[dev,aws]"
   ```

4. Run the tests to ensure everything is working:
   ```bash
   pytest tests/
   ```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment details (OS, Python version, package version)
- Code examples that demonstrate the issue

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

### Suggesting Features

Feature requests are welcome! Please use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) and include:

- A clear description of the problem or limitation
- Your proposed solution
- Example usage of the proposed feature
- Alternative approaches you've considered

### Pull Requests

1. **Fork and Branch**: Create a feature branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Follow Coding Standards**:
   - Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages
   - Follow the existing code style (enforced by Ruff)
   - Add docstrings for public functions and classes
   - Include type hints where appropriate

3. **Write Tests**: Add or update tests for your changes
   ```bash
   pytest tests/ -v
   ```

4. **Run Quality Checks**:
   ```bash
   # Linting and formatting
   ruff check .
   ruff format .

   # Type checking
   mypy src/opensearch_genai_observability_sdk_py --ignore-missing-imports
   ```

5. **Update Documentation**: Update the README, docstrings, or examples as needed

6. **Submit PR**: Create a pull request with:
   - A clear title following conventional commit format
   - Description of changes made
   - Reference to related issues
   - Test instructions for reviewers

## Development Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use Ruff for linting and formatting (configured in `pyproject.toml`)
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`

Examples:
- `feat: add support for gRPC metadata headers`
- `fix: resolve SigV4 signing issue with temporary credentials`
- `docs: update AWS authentication examples`

### Testing

- Write unit tests for new functionality
- Use pytest with clear, descriptive test names
- Mock external dependencies (AWS services, OTEL exporters)
- Maintain or improve test coverage

### Documentation

- Update README.md for user-facing changes
- Add docstrings for all public functions and classes
- Update examples in the `examples/` directory
- Include type hints for better IDE support

## Release Process

Releases are handled by maintainers:

1. Update version in `pyproject.toml`
2. Create and push a version tag
3. GitHub Actions automatically builds and publishes to PyPI (release notes are auto-generated)

## Getting Help

- **Questions**: Open a [discussion](https://github.com/opensearch-project/genai-observability-sdk-py/discussions)
- **Issues**: Use the [issue tracker](https://github.com/opensearch-project/genai-observability-sdk-py/issues)
- **Security**: Report security vulnerabilities privately to security@opensearch.org

## License

By contributing to this project, you agree that your contributions will be licensed under the Apache 2.0 License.