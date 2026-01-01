# Contributing to FaithForge

Thank you for your interest in contributing to FaithForge! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- Redis (optional, for queue features)
- Git

### Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
cp .env.example .env  # fill in your API keys
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Running Tests

```bash
# Backend tests
cd backend
pytest -v

# With coverage
pytest --cov=app --cov-report=html

# Specific test file
pytest tests/test_verifier.py -v
```

### Code Quality

We use `ruff` for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

## Code Style

### Python

- Follow PEP 8 style guide
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- Use `UPPER_SNAKE_CASE` for constants

### Docstring Format

We use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> dict:
    """Short description of the function.

    Longer description if needed, explaining the approach
    and any important details.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When param1 is empty.
        RuntimeError: When something goes wrong.
    """
```

### TypeScript/React

- Use TypeScript for all new code
- Follow the existing component patterns
- Use functional components with hooks
- Keep components small and focused

## Git Workflow

### Branch Naming

- `feature/description` — New features
- `fix/description` — Bug fixes
- `docs/description` — Documentation changes
- `refactor/description` — Code refactoring

### Commit Messages

Use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(verifier): add QLoRA fine-tuning loop
fix(retriever): handle empty BM25 index gracefully
docs(api): add SSE streaming documentation
test(graph): add conditional edge tests
```

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Write/update tests
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

### PR Description Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

## Architecture Decisions

When making significant architectural decisions, please:

1. Discuss in an issue first
2. Document the decision in `docs/ARCHITECTURE.md`
3. Include rationale and alternatives considered

## Adding New Features

### New Agent

1. Create `app/agents/your_agent.py`
2. Implement the agent class with `load()` and main method
3. Add to the LangGraph in `app/agents/graph.py`
4. Add tests in `tests/test_your_agent.py`
5. Update documentation

### New API Endpoint

1. Create or update route in `app/api/`
2. Add request/response models in `app/models/schemas.py`
3. Add tests in `tests/test_api.py`
4. Update API documentation

### New Retrieval Method

1. Implement in `app/services/retriever.py`
2. Add configuration in `app/core/config.py`
3. Add tests in `tests/test_retriever.py`
4. Document the new method

## Reporting Issues

When reporting issues, please include:

1. Steps to reproduce
2. Expected behavior
3. Actual behavior
4. Environment details (OS, Python version, etc.)
5. Error logs if applicable

## Questions?

Open an issue for any questions about contributing.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
