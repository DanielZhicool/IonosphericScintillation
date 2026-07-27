# Contributing to IonosphericScintillation

Thank you for your interest in contributing to **IonosphericScintillation**! We welcome bug reports, feature requests, scientific enhancements, and code contributions.

---

## Development Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/DanielZhicool/IonosphericScintillation.git
   cd IonosphericScintillation
   ```

2. **Environment Management with `uv`**:
   We use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency management:
   ```bash
   uv sync
   ```

---

## Code Quality & Guidelines

Before submitting a pull request, ensure your code passes all linting, static type checking, and unit tests.

### 1. Linting & Formatting with `ruff`
We use `ruff` to enforce PEP 8 standards and code cleanups:
```bash
# Run lint checks
uv run ruff check .

# Apply automatic fixes
uv run ruff check --fix .

# Code formatting check
uv run ruff format --check .
```

### 2. Static Type Checking with `mypy`
Strict type annotations are required across all `core/` and `tests/` modules:
```bash
uv run mypy tests core
```

### 3. Automated Testing with `pytest`
Ensure all unit tests and benchmarks execute cleanly:
```bash
# Run full test suite
uv run pytest

# Run benchmark suite
uv run pytest --benchmark-only
```

---

## Pull Request Workflow

1. **Fork & Branch**: Create a feature branch from `master` (`git checkout -b feature/my-feature`).
2. **Commit Messages**: Use clean, descriptive commit messages adhering to Conventional Commits:
   - `feat(core): ...`
   - `fix(gui): ...`
   - `test(dsp): ...`
   - `docs: ...`
3. **Submit PR**: Open a Pull Request detailing the changes, theoretical justification, and test results.

---

## License

By contributing to this repository, you agree that your contributions will be licensed under the project's [BSD 3-Clause License](LICENSE).
