# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive `README.md` for project overview, features, architecture, installation, usage, and testing.
- `CONTRIBUTING.md` with detailed guidelines for community contributions (setup, code style, testing, pull request process).
- `LICENSE` file for MIT License terms.
- Initial project audit and architectural mapping.

### Changed
- Refined initial docstrings in core modules for improved clarity and consistency. (Planned)
- Enhanced inline comments in complex logic sections. (Planned)

### Fixed
- (To be filled with any specific bug fixes identified during audit)

### Removed
- (To be filled if any parts are removed before release)

## [2.0.0] - 2025-08-16

### Added
- Initial commit of FBA-Bench v3 project structure.
- Core event-driven simulation framework (`EventBus`, `BaseEvent`, `TickEvent`).
- `SimulationOrchestrator` for deterministic time progression.
- `FinancialAuditService` for financial integrity validation.
- Modular agent system with `BaseSkill` and `SkillCoordinator`.
- `MultiDomainController` for strategic agent arbitration.
- LLM integration layer (`llm_interface/`) with support for OpenAI and OpenRouter.
- `ScenarioEngine` and `ScenarioConfig` for flexible scenario definition.
- `ToolboxAPIService` as an agent-world interaction facade.
- FastAPI-based backend API (`fba_bench_api/`) with real-time capabilities.
- React/TypeScript frontend application (`frontend/`).
- Basic CLI for running experiments (`experiment_cli.py`).
- Deterministic `Money` type for precise financial calculations.
- Distributed event bus infrastructure.

### Changed
- (Initial version, so few "changed" entries)

### Fixed
- (Initial version, so few "fixed" entries)

### Removed
- (Initial version, so few "removed" entries)
