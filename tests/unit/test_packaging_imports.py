import importlib
import types


def _import(module_name: str) -> types.ModuleType:
    mod = importlib.import_module(module_name)
    assert mod is not None, f"Failed to import {module_name}"
    return mod


def test_import_top_level_packages():
    # Top-level packages should be importable from the wheel
    _import("fba_bench")
    _import("fba_bench_api")
    _import("agent_runners")
    _import("benchmarking")
    _import("integration")
    _import("observability")
    _import("services")
    _import("models")
    _import("metrics")
    _import("memory_experiments")
    _import("reproducibility")
    _import("redteam")
    _import("scenarios")
    _import("plugins")


def test_import_representative_submodules():
    # Submodules to ensure subpackages are included without runtime side effects
    fm = _import("metrics.finance_metrics")
    cm = _import("metrics.cognitive_metrics")
    assert hasattr(fm, "__file__")
    assert hasattr(cm, "__file__")

    dmm = _import("memory_experiments.dual_memory_manager")
    mv = _import("memory_experiments.memory_validator")
    assert hasattr(dmm, "__file__")
    assert hasattr(mv, "__file__")

    prod = _import("models.product")
    assert hasattr(prod, "__file__")

    bsp = _import("plugins.scenario_plugins.base_scenario_plugin")
    assert hasattr(bsp, "__file__")

    aei = _import("redteam.adversarial_event_injector")
    assert hasattr(aei, "__file__")