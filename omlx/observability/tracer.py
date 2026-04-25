# SPDX-License-Identifier: Apache-2.0
"""Module / class function tracer.

Wraps every callable in a target module or class with the observability
timers without modifying source. The analyst can instrument any of the
implementer's classes (DuoKVCache, SnapKV, TQ3, etc.) without violating
the "do not touch the implementation" rule:

    from omlx.observability.tracer import trace_class, untrace
    from omlx.duo_kv_cache import DuoKVCache

    handle = trace_class(DuoKVCache, prefix="duokv", mlx=True)
    # ... run benchmark ...
    untrace(handle)
    registry.print_summary()

Every method now records a timer like "duokv.update_and_fetch" plus a
counter "duokv.update_and_fetch.calls". `mlx=True` forces graph
materialization at exit so async GPU work counts.

Tracing module-level functions:

    from omlx.observability.tracer import trace_module
    handle = trace_module("omlx.patches.snapkv", prefix="snapkv")

Notes
-----
- Dunder methods (`__init__`, `__repr__`, etc.) are skipped by default.
  Pass `include_dunder=True` to wrap them.
- Properties, classmethods, staticmethods are wrapped correctly.
- Calling `trace_class` twice on the same class is a no-op for already-
  wrapped methods (idempotent).
- Wrapped methods preserve `__wrapped__` so introspection still works.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from omlx.observability.counters import counter
from omlx.observability.timers import mlx_timer, timer


@dataclass
class TraceHandle:
    """Restoration handle. Pass to `untrace()` to revert."""
    target: Any  # Module or class
    originals: List[Tuple[str, Callable]] = field(default_factory=list)
    prefix: str = ""

    def restore(self) -> None:
        for name, original in self.originals:
            try:
                setattr(self.target, name, original)
            except Exception:
                pass
        self.originals.clear()


def _is_traceable(obj: Any) -> bool:
    """Skip non-callables, types, and already-wrapped functions."""
    if not callable(obj):
        return False
    if inspect.isclass(obj):
        return False
    if getattr(obj, "_omlx_traced", False):
        return False
    return True


def _make_wrapper(fn: Callable, name: str, mlx: bool) -> Callable:
    """Build a timing wrapper preserving signature and metadata."""
    timer_factory = mlx_timer if mlx else timer
    call_counter = counter(f"{name}.calls")

    if inspect.iscoroutinefunction(fn):
        async def async_wrapper(*args, **kwargs):
            call_counter.add(1)
            with timer_factory(name):
                return await fn(*args, **kwargs)
        async_wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        async_wrapper._omlx_traced = True  # type: ignore[attr-defined]
        try:
            async_wrapper.__name__ = fn.__name__
            async_wrapper.__qualname__ = fn.__qualname__
            async_wrapper.__doc__ = fn.__doc__
        except Exception:
            pass
        return async_wrapper

    def sync_wrapper(*args, **kwargs):
        call_counter.add(1)
        with timer_factory(name):
            return fn(*args, **kwargs)
    sync_wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    sync_wrapper._omlx_traced = True  # type: ignore[attr-defined]
    try:
        sync_wrapper.__name__ = fn.__name__
        sync_wrapper.__qualname__ = fn.__qualname__
        sync_wrapper.__doc__ = fn.__doc__
    except Exception:
        pass
    return sync_wrapper


def trace_class(
    cls: type,
    prefix: Optional[str] = None,
    mlx: bool = False,
    include_dunder: bool = False,
    skip: Optional[List[str]] = None,
) -> TraceHandle:
    """Wrap every method on `cls` with a timer + call counter.

    Parameters
    ----------
    cls : type
        Target class (mutated in place — the rest of the program will
        see wrapped methods until `untrace()` is called).
    prefix : str, optional
        Timer name prefix. Defaults to `cls.__qualname__.lower()`.
    mlx : bool
        If True, use `mlx_timer` (forces graph materialization on exit).
        Use for methods that return MLX arrays.
    include_dunder : bool
        Wrap `__*__` methods too. Default False (avoids spam from `__repr__`).
    skip : list[str], optional
        Method names to skip.

    Returns
    -------
    TraceHandle — pass to `untrace()` when done.
    """
    if prefix is None:
        prefix = cls.__qualname__.lower()
    skip_set = set(skip or [])
    handle = TraceHandle(target=cls, prefix=prefix)

    for attr_name, attr in list(vars(cls).items()):
        if attr_name in skip_set:
            continue
        if not include_dunder and attr_name.startswith("__"):
            continue

        # classmethod / staticmethod descriptors — unwrap, wrap, rewrap.
        if isinstance(attr, classmethod):
            inner = attr.__func__
            if not _is_traceable(inner):
                continue
            wrapped = _make_wrapper(inner, f"{prefix}.{attr_name}", mlx)
            new_attr = classmethod(wrapped)
            handle.originals.append((attr_name, attr))
            setattr(cls, attr_name, new_attr)
            continue

        if isinstance(attr, staticmethod):
            inner = attr.__func__
            if not _is_traceable(inner):
                continue
            wrapped = _make_wrapper(inner, f"{prefix}.{attr_name}", mlx)
            new_attr = staticmethod(wrapped)
            handle.originals.append((attr_name, attr))
            setattr(cls, attr_name, new_attr)
            continue

        if isinstance(attr, property):
            # Properties: wrap getter/setter individually if traceable.
            new_fget = attr.fget
            new_fset = attr.fset
            new_fdel = attr.fdel
            changed = False
            if attr.fget is not None and _is_traceable(attr.fget):
                new_fget = _make_wrapper(attr.fget, f"{prefix}.{attr_name}.get", mlx)
                changed = True
            if attr.fset is not None and _is_traceable(attr.fset):
                new_fset = _make_wrapper(attr.fset, f"{prefix}.{attr_name}.set", mlx)
                changed = True
            if changed:
                handle.originals.append((attr_name, attr))
                setattr(cls, attr_name, property(new_fget, new_fset, new_fdel, attr.__doc__))
            continue

        if not _is_traceable(attr):
            continue

        wrapped = _make_wrapper(attr, f"{prefix}.{attr_name}", mlx)
        handle.originals.append((attr_name, attr))
        setattr(cls, attr_name, wrapped)

    return handle


def trace_module(
    mod: Union[str, Any],
    prefix: Optional[str] = None,
    mlx: bool = False,
    include_dunder: bool = False,
    skip: Optional[List[str]] = None,
) -> TraceHandle:
    """Wrap every module-level function in `mod` with a timer + counter.

    Parameters
    ----------
    mod : str | module
        Module name (`"omlx.patches.snapkv"`) or already-imported module.
    Other params : same as `trace_class`.

    Does NOT recurse into nested classes — call `trace_class` on those
    explicitly.
    """
    if isinstance(mod, str):
        mod = importlib.import_module(mod)
    if prefix is None:
        prefix = mod.__name__.split(".")[-1]
    skip_set = set(skip or [])
    handle = TraceHandle(target=mod, prefix=prefix)

    for attr_name in list(vars(mod).keys()):
        if attr_name in skip_set:
            continue
        if not include_dunder and attr_name.startswith("__"):
            continue
        attr = getattr(mod, attr_name)
        # Only wrap items defined IN this module — skip imports.
        if not _is_traceable(attr):
            continue
        if getattr(attr, "__module__", None) != mod.__name__:
            continue

        wrapped = _make_wrapper(attr, f"{prefix}.{attr_name}", mlx)
        handle.originals.append((attr_name, attr))
        setattr(mod, attr_name, wrapped)

    return handle


def untrace(handle: TraceHandle) -> None:
    """Restore original methods/functions on the target."""
    handle.restore()


# Convenience: trace common hypercar internals in one shot.

def trace_hypercar_internals(mlx: bool = True) -> List[TraceHandle]:
    """Wrap the suspected hot classes/modules with one call.

    Returns a list of handles. Pass each to `untrace()` when done, or
    iterate: `for h in handles: untrace(h)`.

    Skips imports that fail (e.g. when a module is renamed).
    """
    targets: List[Tuple[str, Optional[str]]] = [
        # (importable, class_name_or_None_for_module)
        ("omlx.duo_kv_cache", "DuoKVCache"),
        ("omlx.streaming_attention", None),
        ("omlx.turboquant_kv", "TurboQuantKVCache"),
        ("omlx.shadowkv_cache", None),
        ("omlx.patches.snapkv", None),
    ]
    handles: List[TraceHandle] = []
    for mod_name, cls_name in targets:
        try:
            mod = importlib.import_module(mod_name)
            if cls_name is None:
                handles.append(trace_module(mod, mlx=mlx))
            else:
                cls = getattr(mod, cls_name, None)
                if cls is None:
                    continue
                handles.append(trace_class(cls, mlx=mlx))
        except Exception:
            continue
    return handles
