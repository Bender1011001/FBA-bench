"""
Agent Registry for the FBA-Bench benchmarking framework.

This module provides a centralized registry to manage and access different
types of agents used in benchmark scenarios.

Implements a version-aware AgentRegistry with deprecation shims and a
rich AgentDescriptor structure suitable for plugins and core agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Union, Iterable
import inspect
import warnings

# Prefer packaging.version.Version if available for robust version comparison
try:
    from packaging.version import Version as _SemVer  # type: ignore
except Exception:  # Lightweight fallback
    _SemVer = None  # type: ignore


def _to_version_key(v: Optional[str]) -> tuple:
    """
    Convert a version string into a sortable key.

    Uses packaging.version.Version when available; else falls back
    to comparing tuples of ints for dotted versions and strings.

    Examples:
    - "1.2.10" > "1.2.2"
    - "1.2.0" > "1.1.9"
    """
    if not v:
        return tuple()
    if _SemVer is not None:
        try:
            return (_SemVer(v),)
        except Exception:
            # Fall back to simple tuple-of-ints with string tie-breaker
            pass
    parts: List[Union[int, str]] = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(p)
    return tuple(parts)  # type: ignore[return-value]


@dataclass
class AgentDescriptor:
    """
    Describes a registered agent.

    Fields:
      - slug: Unique agent slug identifier (e.g., "gpt4o_mini")
      - display_name: Human-friendly agent name
      - constructor: Callable or pre-instantiated object
      - supported_capabilities: Capabilities this agent supports
      - default_config_model: Optional Pydantic model class for config validation
      - version: Optional version string (semver preferred)
      - provenance: "core" or "plugin"
      - framework: Optional framework name
      - tags: Arbitrary list of tags
      - help: Optional help/description text
      - enabled: Whether this agent is enabled for discovery/creation
    """
    slug: str
    display_name: str
    constructor: Union[Callable[..., Any], Any]
    supported_capabilities: List[str] = field(default_factory=list)
    default_config_model: Optional[type] = None
    version: Optional[str] = None
    provenance: str = "core"
    framework: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    help: Optional[str] = None
    enabled: bool = True


class AgentRegistry:
    """
    Registry for agents with version management and instantiation helpers.

    Internal maps:
      - _by_slug: dict[str, AgentDescriptor] default descriptor for slug
      - _by_slug_version: dict[str, dict[str, AgentDescriptor]] all versions per slug

    Backward-compatible shims:
      - register(...) -> register_agent(...) [DeprecationWarning]
      - get(...) -> get_agent(...) [DeprecationWarning]
    """

    def __init__(self) -> None:
        self._by_slug: Dict[str, AgentDescriptor] = {}
        self._by_slug_version: Dict[str, Dict[str, AgentDescriptor]] = {}

    # ---------------------------- Registration ----------------------------

    def register_agent(self, descriptor: AgentDescriptor, *, default_latest: bool = True) -> None:
        """
        Register an AgentDescriptor.

        Behavior:
          - Validate slug and constructor presence.
          - If version provided, insert into _by_slug_version[slug][version].
          - If default for slug not set, or default_latest is True and version is higher,
            set _by_slug[slug] to this descriptor.
          - On duplicate slug+version, raise ValueError with:
            "Duplicate agent registration for slug {slug} version {version}"

        Examples:
          - Register two versions and resolve default:
              reg.register_agent(AgentDescriptor(slug="my", display_name="My", constructor=My, version="1.0.0"))
              reg.register_agent(AgentDescriptor(slug="my", display_name="My v2", constructor=My2, version="2.0.0"))
              # Default resolves to v2 when default_latest=True

          - Register a pre-instantiated fake agent for tests:
              fake = object()
              reg.register_agent(AgentDescriptor(slug="fake", display_name="Fake", constructor=fake))
        """
        if not descriptor.slug or descriptor.constructor is None:
            raise ValueError("AgentDescriptor must include slug and constructor")

        slug = descriptor.slug
        ver = descriptor.version

        if ver:
            if slug not in self._by_slug_version:
                self._by_slug_version[slug] = {}
            if ver in self._by_slug_version[slug]:
                raise ValueError(f"Duplicate agent registration for slug {slug} version {ver}")
            self._by_slug_version[slug][ver] = descriptor

        # Determine default
        current_default = self._by_slug.get(slug)
        if current_default is None:
            self._by_slug[slug] = descriptor
        elif default_latest:
            # Compare versions when possible; if new one is higher, update
            new_key = _to_version_key(ver)
            old_key = _to_version_key(current_default.version if current_default else None)
            if new_key and (not old_key or new_key > old_key):
                self._by_slug[slug] = descriptor

        # If no version tracking but default is unset, ensure default is present
        if slug not in self._by_slug and descriptor.enabled:
            self._by_slug[slug] = descriptor

    # ------------------------------ Querying ------------------------------

    def get_agent(self, slug: str, version: Optional[str] = None) -> AgentDescriptor:
        """
        Retrieve a registered agent descriptor.

        Raises KeyError with exact messages:
          - "Agent {slug}@{version} not found" when version provided
          - "Agent {slug} not found" when version not provided
        """
        if version:
            versions = self._by_slug_version.get(slug) or {}
            desc = versions.get(version)
            if not desc:
                raise KeyError(f"Agent {slug}@{version} not found")
            return desc

        desc = self._by_slug.get(slug)
        if not desc:
            raise KeyError(f"Agent {slug} not found")
        return desc

    def has_agent(self, slug: str, version: Optional[str] = None) -> bool:
        """Return True if an agent with slug (and version when provided) exists."""
        try:
            _ = self.get_agent(slug, version)
            return True
        except KeyError:
            return False

    def unregister_agent(self, slug: str, version: Optional[str] = None) -> None:
        """
        Unregister an agent. Adjust default if needed.

        Behavior:
          - Remove from version map when version provided
          - If removing default and others remain, set highest semver as new default
          - If no versions remain, remove slug from _by_slug
        """
        if version:
            versions = self._by_slug_version.get(slug)
            if versions and version in versions:
                # Remove specified version
                removed = versions.pop(version, None)
                # Adjust default if it pointed to removed descriptor
                if removed and self._by_slug.get(slug) is removed:
                    if versions:
                        # Choose highest version as default
                        highest_ver = max(versions.keys(), key=_to_version_key)
                        self._by_slug[slug] = versions[highest_ver]
                    else:
                        self._by_slug.pop(slug, None)
                # Clean empty version map
                if versions is not None and not versions:
                    self._by_slug_version.pop(slug, None)
            else:
                # No-op if not present
                return
        else:
            # Remove default and all versions
            self._by_slug.pop(slug, None)
            self._by_slug_version.pop(slug, None)

    def list_agents(
        self,
        *,
        provenance: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        enabled: Optional[bool] = None,
    ) -> List[AgentDescriptor]:
        """
        List all registered agent descriptors across versions with filtering.

        Filters:
          - provenance: "core" or "plugin"
          - capabilities: all requested must be included
          - tags: all requested must be included
          - enabled: exact match if provided
        """
        # Aggregate all descriptors across versions plus defaults (avoid duplicates by id)
        seen: set[int] = set()
        all_descs: List[AgentDescriptor] = []

        for slug, desc in self._by_slug.items():
            if id(desc) not in seen:
                all_descs.append(desc)
                seen.add(id(desc))
        for slug, versions in self._by_slug_version.items():
            for ver_desc in versions.values():
                if id(ver_desc) not in seen:
                    all_descs.append(ver_desc)
                    seen.add(id(ver_desc))

        def includes_all(haystack: Iterable[str], needles: Iterable[str]) -> bool:
            h = set(haystack or [])
            return all(n in h for n in (needles or []))

        result: List[AgentDescriptor] = []
        for d in all_descs:
            if provenance is not None and d.provenance != provenance:
                continue
            if enabled is not None and d.enabled is not enabled:
                continue
            if capabilities and not includes_all(d.supported_capabilities, capabilities):
                continue
            if tags and not includes_all(d.tags, tags):
                continue
            result.append(d)
        return result

    # --------------------------- Instantiation ----------------------------

    def create_agent(
        self,
        slug: str,
        *,
        version: Optional[str] = None,
        config: Union[Dict[str, Any], Any, None] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create an agent instance via its descriptor.

        Behavior:
          - Resolve descriptor via get_agent(slug, version)
          - If constructor is a pre-instantiated object: return it directly
          - If default_config_model is provided:
              - If config is dict, instantiate/validate via model(**config)
              - If config is already a model instance, use it
              - On validation errors, raise ValueError with the model's error text
          - Invoke constructor:
              - If first positional parameter is for config, pass it as first arg;
                else pass as keyword 'config=config'
              - Forward additional **kwargs
          - On constructor failure, wrap in:
              ValueError(f"Agent {slug} creation failed: {exc}") chaining original exception
        """
        descriptor = self.get_agent(slug, version)
        ctor = descriptor.constructor

        # Pre-instantiated instance
        if not callable(ctor):
            return ctor

        # Validate/prepare config using default_config_model if provided
        model_cls = descriptor.default_config_model
        cfg_obj = config
        if model_cls is not None:
            try:
                if config is None:
                    cfg_obj = None
                elif isinstance(config, model_cls):
                    cfg_obj = config
                elif isinstance(config, dict):
                    # pydantic-like model; instantiate and let it validate
                    cfg_obj = model_cls(**config)  # type: ignore[misc]
                else:
                    # Attempt direct cast
                    cfg_obj = model_cls(**getattr(config, "model_dump", lambda: {})())  # type: ignore[misc]
            except Exception as exc:
                # Ensure ValueError with model error text
                raise ValueError(str(exc)) from exc

        # Inspect constructor to determine how to pass config
        try:
            sig = inspect.signature(ctor)
            params = list(sig.parameters.values())
            args: List[Any] = []
            kwargs_final: Dict[str, Any] = dict(kwargs)

            if params:
                # If first parameter is not self and is positional-or-keyword, assume it's config
                first = params[0]
                if first.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    # Heuristic: If parameter name hints config or model provided, pass positionally
                    if first.name in {"config", "cfg", "agent_config"}:
                        args.append(cfg_obj)
                    else:
                        # If only one param, often it is config; pass positionally
                        args.append(cfg_obj)
                else:
                    # Fallback to keyword
                    kwargs_final["config"] = cfg_obj
            else:
                # No parameters; if config exists, pass as keyword (constructor may accept **kwargs)
                if cfg_obj is not None:
                    kwargs_final["config"] = cfg_obj

            return ctor(*args, **kwargs_final)  # type: ignore[call-arg]
        except Exception as exc:
            raise ValueError(f"Agent {slug} creation failed: {exc}") from exc

    # -------------------------- Deprecation Shims -------------------------

    def register(self, *args: Any, **kwargs: Any) -> None:
        """
        Deprecated. Use register_agent(AgentDescriptor) instead.

        Supports legacy calls: register(slug, cls_or_instance, ...)
        """
        warnings.warn(
            "AgentRegistry.register is deprecated; use register_agent with AgentDescriptor",
            DeprecationWarning,
            stacklevel=2,
        )
        if args and isinstance(args[0], AgentDescriptor):
            return self.register_agent(args[0], **kwargs)  # type: ignore[misc]

        # Legacy: register(slug, agent_class_or_instance, **extras)
        if not args:
            raise ValueError("register requires at least a slug and constructor")
        slug = args[0]
        ctor = args[1] if len(args) > 1 else kwargs.get("constructor")
        descriptor = AgentDescriptor(
            slug=slug,
            display_name=kwargs.get("display_name", slug),
            constructor=ctor,
            version=kwargs.get("version"),
            provenance=kwargs.get("provenance", "core"),
            framework=kwargs.get("framework"),
            supported_capabilities=kwargs.get("supported_capabilities", []) or [],
            tags=kwargs.get("tags", []) or [],
            help=kwargs.get("help"),
            enabled=kwargs.get("enabled", True),
            default_config_model=kwargs.get("default_config_model"),
        )
        self.register_agent(descriptor, default_latest=kwargs.get("default_latest", True))

    def get(self, slug: str, version: Optional[str] = None) -> AgentDescriptor:
        """
        Deprecated. Use get_agent(slug, version) instead.
        """
        warnings.warn(
            "AgentRegistry.get is deprecated; use get_agent",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_agent(slug, version)


# Module-level singleton
agent_registry = AgentRegistry()


def describe_agent(slug: str, version: Optional[str] = None) -> Dict[str, Any]:
    """
    Serialize an agent descriptor to a dict for API usage.
    Raises KeyError if not found with exact messages defined in get_agent.
    """
    desc = agent_registry.get_agent(slug, version)
    data = asdict(desc)
    return data