# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < Latest| :x:                |

We recommend always using the latest version of DeepFilterNet to ensure you have the latest security patches.

## Reporting a Vulnerability

We take the security of DeepFilterNet seriously. If you discover a security vulnerability, please report it privately.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them using one of the following methods:

1. **Preferred**: Use GitHub's private vulnerability reporting feature:
   - Go to the [Security Advisories](https://github.com/Rikorose/DeepFilterNet/security/advisories) page
   - Click "Report a vulnerability"
   - Fill out the form with details about the vulnerability

2. **Alternative**: Email the maintainers directly (if contact info is available in the repository)

### What to Include

When reporting a vulnerability, please include:

- Type of vulnerability (e.g., code injection, memory corruption, etc.)
- Affected component(s) (Python package, Rust library, LADSPA plugin, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: We aim to acknowledge receipt of your vulnerability report within 48 hours
- **Status Updates**: We will provide status updates at least every 7 days
- **Resolution**: We will work to validate and address confirmed vulnerabilities as quickly as possible
- **Disclosure**: We will coordinate with you on the disclosure timeline

### What to Expect

1. Your report will be reviewed and validated
2. We will work on a fix and coordinate a release timeline with you
3. We will credit you for the discovery (unless you prefer to remain anonymous)
4. We will publish a security advisory once the fix is released

## Security Best Practices for Users

### When Using DeepFilterNet

1. **Keep Updated**: Always use the latest version
2. **Validate Inputs**: Be cautious when processing untrusted audio files
3. **Model Files**: Only use model files from trusted sources
4. **Dependencies**: Keep Python and Rust dependencies updated
5. **Permissions**: Run with minimal required permissions

### For the LADSPA Plugin

1. Only load the plugin from trusted sources
2. Keep the plugin updated
3. Be aware that audio processing runs in the same security context as the audio server

### For Development

1. Review dependencies for known vulnerabilities regularly
2. Use official package sources (crates.io, PyPI)
3. Enable all security features when building from source
4. Run security scanning tools on your builds

## Known Security Considerations

### Audio Processing

- Audio files can contain malformed data that may cause parsing issues
- Large audio files may consume significant memory
- Real-time processing requires appropriate resource limits

### Model Files

- Model files are loaded and executed during inference
- Only use model files from trusted sources
- Model files are not cryptographically verified (feature consideration)

### Dependencies

- The project depends on numerous third-party packages
- We use Dependabot to monitor for vulnerable dependencies
- Review dependency updates in pull requests

## Security Tooling

This repository uses:

- **CodeQL**: Automated security scanning for Python and Rust code
- **Dependency Review**: Checks for vulnerable dependencies in PRs
- **Dependabot**: Automatic dependency updates
- **GitHub Security Advisories**: Track and disclose security issues

## Contact

For security-related questions that are not vulnerabilities, please open a discussion in the [GitHub Discussions](https://github.com/Rikorose/DeepFilterNet/discussions) area.

Thank you for helping keep DeepFilterNet secure!
