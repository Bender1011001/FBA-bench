# How to Contribute to FBA-Bench Development

We welcome contributions from the community to enhance FBA-Bench! This guide outlines the process for contributing code, documentation, and other improvements to the project. By following these guidelines, you help maintain code quality, consistency, and a smooth collaboration workflow.

## 1. Getting Started

-   **Fork the Repository**: Start by forking the FBA-Bench (https://github.com/FBA-Bench-Org/FBA-Bench) repository on GitHub.
-   **Clone Your Fork**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/fba-bench.git
    cd fba-bench
    ```
-   **Set Upstream Remote**: Add the original FBA-Bench repository as an "upstream" remote:
    ```bash
    git remote add upstream https://github.com/FBA-Bench-Org/fba-bench.git
    ```
-   **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows: .venv\Scripts\activate
    ```
-   **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt # Assuming a file for development dependencies
    ```

## 2. Code Style Guidelines and Standards

To ensure a consistent and readable codebase, all contributions must adhere to the following standards:

-   **Python Formatting**: Use `black` for code formatting.
    -   Install: `pip install black`
    -   Run: `black .` from the project root.
-   **Linting**: Use `flake8` for linting.
    -   Install: `pip install flake8`
    -   Run: `flake8 .`
-   **Type Hinting**: All new Python code should extensively use type hints as per PEP 484.
-   **Docstrings**: All functions, methods, and classes should have clear and concise docstrings following the Google-style format.
-   **Naming Conventions**: Follow PEP 8 for naming conventions (e.g., `snake_case` for functions/variables, `PascalCase` for classes).
-   **Test Coverage**: New features and bug fixes must include corresponding unit and integration tests.

## 3. Testing Requirements and Procedures

All contributions, especially code changes, must be thoroughly tested.

-   **Unit Tests**: Write unit tests for individual functions and classes using `pytest`. Tests should cover typical use cases, edge cases, and error conditions.
    -   Run all unit tests: `pytest tests/unit`
    -   Run specific test file: `pytest tests/unit/my_module_test.py`
-   **Integration Tests**: For features spanning multiple modules or interacting with external systems (mocked), write integration tests.
    -   Run integration tests: `pytest tests/integration`
-   **Reproducibility Testing**: If your changes affect LLM interactions or stochastic processes, ensure reproducibility tests pass.
    -   Run reproducibility tests: `pytest tests/reproducibility`
-   **Performance Benchmarks**: If your changes impact performance, run relevant benchmarks and include results in your pull request.

## 4. Pull Request and Review Process

1.  **Create a New Branch**:
    ```bash
    git checkout -b feature/my-new-feature-name # For new features
    git checkout -b bugfix/issue-description # For bug fixes
    ```
2.  **Make Changes**: Implement your feature or fix, adhering to code style and testing guidelines.
3.  **Commit Changes**: Write clear, concise commit messages.
    ```bash
    git commit -m "feat: Add new skill module for forecasting"
    ```
    -   Use conventional commit messages (e.g., `feat:`, `fix:`, `docs:`, `chore:`).
4.  **Push to Your Fork**:
    ```bash
    git push origin feature/my-new-feature-name
    ```
5.  **Open a Pull Request (PR)**:
    -   Go to the FBA-Bench GitHub repository and click on "New Pull Request".
    -   Ensure your branch is selected, and target the `main` branch of the upstream repository.
    -   Provide a detailed description of your changes, including:
        -   What problem does this PR solve?
        -   How was it solved?
        -   Any relevant testing steps or benchmark results.
        -   References to any open issues (e.g., `Fixes #123`).
6.  **Address Feedback**: Respond to comments from reviewers and make necessary changes. Your PR will be merged once it meets all requirements and receives approval.

## 5. Documentation Contributions

-   **Update Existing Documentation**: If your code changes affect existing features, update the relevant documentation files (e.g., in `docs/`).
-   **New Feature Documentation**: For new major features, create new documentation files in the appropriate `docs/` subdirectory (e.g., `docs/quick-start/`, `docs/feature-specific/`).
-   **Clarity and Conciseness**: Write clear, consistent, and user-friendly documentation. Include code examples where appropriate.

Your contributions are highly valued and help make FBA-Bench a robust and accessible platform!