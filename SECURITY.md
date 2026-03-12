# Security Policy

## Supported Versions

We support the current major version with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: security@opensearch.org

Include the following information:
- Type of issue (e.g. buffer overflow, SQL injection, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

We will acknowledge receipt of your vulnerability report within 48 hours and send a more detailed response within 7 days indicating the next steps in handling your report.

## Security Considerations

When using opensearch-genai-observability-sdk-py:

1. **Credentials**: Ensure AWS credentials are stored securely and follow the principle of least privilege
2. **Network**: Use HTTPS/TLS for all OTLP connections in production
3. **Data**: Be mindful of sensitive data in traces - the SDK does not automatically scrub PII
4. **Dependencies**: Keep dependencies up to date using tools like Dependabot

## Responsible Disclosure

We follow responsible disclosure practices and will work with security researchers to understand and address security vulnerabilities before public disclosure.