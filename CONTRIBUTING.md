# Contributing to FBA-Bench v3

We welcome contributions from the community to help improve and expand FBA-Bench v3! Your input helps us make this toolkit more robust, versatile, and useful for financial and business agent benchmarking.

Please take a moment to review this document to understand our contribution process.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## How Can I Contribute?

There are several ways you can contribute to FBA-Bench v3:

*   **Report Bugs**: If you find a bug, please open an issue on our GitHub repository. Provide a clear description, steps to reproduce, and expected behavior.
*   **Suggest Enhancements**: Have an idea for a new feature or an improvement to existing functionality? Open an issue to discuss it.
*   **Write Code**: Contribute bug fixes, new features, or improvements to documentation.
*   **Improve Documentation**: Help us make our documentation clearer, more comprehensive, and easier to understand.
*   **Share your Scenarios/Agents**: If you develop interesting scenarios or agents, consider sharing them with the community!

## Getting Started with Code Contributions

Follow these steps to set up your development environment and make your first contribution:

1.  **Fork the Repository**: Go to the [FBA-Bench v3 GitHub repository](https://github.com/your-org/fba-bench) and click the "Fork" button.
2.  **Clone Your Fork**:
    ```bash
    git clone https://github.com/your-username/fba-bench.git
    cd fba-bench
    ```
3.  **Set Up Your Environment**:
    *   **Backend (Python via Poetry)**:
        ```bash
        pip install -U pip setuptools wheel
        pip install poetry
        poetry install --with dev
        # Optional: activate a Poetry-managed virtualenv shell
        poetry shell
        ```
    *   **Frontend (Node.js/TypeScript)**:
        ```bash
        cd frontend
        npm install # or yarn install
        cd ..
        ```
    *   **Database**: For local development, an SQLite database is configured by default. No extra setup might be needed. For distributed features, ensure Redis is running (e.g., via Docker).
4.  **Create a New Branch**:
    ```bash
    git checkout -b feature/your-feature-name # for features
    git checkout -b bugfix/your-bug-fix-name # for bug fixes
    ```
5.  **Make Your Changes**:
    *   Implement your feature or bug fix.
    *   Write clean, well-commented, and production-ready code.
    *   Ensure all new code adheres to the project's coding style and best practices.
    *   **Write Tests**: All new features and bug fixes should be accompanied by appropriate tests (unit, integration, or E2E).
    *   **Update Documentation**: If your changes introduce new functionality or modify existing APIs, update the relevant docstrings, inline comments, and markdown documentation files (like `README.md`, `API_REFERENCE.md`).
6.  **Run Tests**: Before submitting, ensure all tests pass.
    *   **Backend**: `poetry run pytest` (from root directory)
    *   **Frontend**: `cd frontend && npm test` (from `frontend/` directory)
7.  **Format and Lint**: Ensure your code is formatted correctly.
    *   **Python**: Use Black, Ruff/Flake8, and isort via Poetry.
        ```bash
        poetry run black .
        poetry run ruff check .
        poetry run flake8 .
        poetry run isort .
        ```
    *   **TypeScript/React**: Use Prettier and ESLint.
        ```bash
        cd frontend
        npm run format # Example command if configured in package.json
        npm run lint # Example command if configured
        cd ..
        ```
8.  **Commit Your Changes**: Write clear, concise commit messages.
    ```bash
    git add .
    git commit -m "feat: Add new awesome feature"
    # or "fix: Resolve critical bug in module X"
    ```
9.  **Push to Your Fork**:
    ```bash
    git push origin feature/your-feature-name
    ```
10. **Create a Pull Request (PR)**:
    *   Go to your forked repository on GitHub and click "Compare & pull request."
    *   Provide a descriptive title and detailed description of your changes.
    *   Reference any related issues (e.g., "Closes #123" or "Fixes #456").

## Code Style and Standards

*   **Python**: Adhere to PEP 8. Use clear, descriptive variable and function names. Employ `dataclasses` where appropriate for data structures.
*   **TypeScript/React**: Follow standard React best practices. Use TypeScript interfaces for clear type definitions.
*   **Documentation**:
    *   **Docstrings**: Use Google-style docstrings for all public functions, methods, and classes.
    *   **Inline Comments**: Add comments for non-obvious logic, complex algorithms, or workaround explanations.
    *   **Markdown**: Follow a consistent style for all `.md` files.

## Review Process

*   All pull requests will be reviewed by maintainers.
*   Feedback will be provided on code quality, correctness, style, and adherence to requirements.
*   Be prepared to iterate on your changes based on feedback.

Thank you for contributing to FBA-Bench v3!