"""A Flask-test-client driver that executes real Dash callbacks in-process.

Until #399 nothing under ``tests/`` ever ran a callback. The unit tier calls
helper functions directly, and the smoke tier calls ``layout()`` and asserts
the result is not ``None`` — neither one crosses the wiring between them. A
callback whose ``Output`` id no longer exists in the layout, or whose return
value cannot be serialised to JSON, passed both tiers and broke in a browser.

This driver closes that gap without a browser. Dash's client is not magic: a
callback invocation is a ``POST /_dash-update-component`` carrying the output
spec, the input values and the state values, and Dash's own callback registry
is served at ``GET /_dash-dependencies``. So the driver reads the registry,
builds a well-formed request for the callback under test, and posts it through
``app.server.test_client()``. Dash routes it, resolves the callback, invokes
the real function, and serialises the real return value.

What that buys over the smoke tier:

* The callback **function actually runs**, with Dash's argument ordering — a
  registrar that wires inputs in the wrong order is visible here and nowhere
  else.
* An ``Output`` whose id is absent from the layout raises inside Dash's
  dispatch, so orphaned callbacks fail loudly instead of silently never
  firing.
* The return value goes through Plotly/Dash JSON serialisation, so a figure
  or component tree that cannot be sent to a browser fails here.

What it still does not buy: no browser, so no CSS, no clientside callbacks, no
"the chart rendered but is blank". Those need a real browser tier, which the
repo does not have. See ``tests/TEST_PYRAMID.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Dash appends ``@<hash>`` to an output in the callback-map key when the
# output is declared with ``allow_duplicate=True``, so two callbacks can
# target the same component property. Lookups strip it.
_DUP_SUFFIX = "@"


def _flatten_output_key(key: str) -> list[str]:
    """Split a callback-map key into its ``"id.property"`` components.

    Dash encodes a multi-output callback as ``"..a.x...b.y.."`` and a
    single-output one as ``"a.x"``. Duplicate-allowed outputs carry an
    ``@hash`` suffix that is not part of the component identity.
    """
    parts = key.strip(".").split("...") if key.startswith("..") else [key]
    return [p.split(_DUP_SUFFIX)[0] for p in parts]


def component_text(node: Any) -> str:
    """Collect every string in a serialised Dash component tree.

    Callback responses come back as nested ``{"props": {...}, "type": ...}``
    dicts. Assertions want to say "the region name reached the heading", not
    to index seven levels of props, so this flattens the tree to its text.
    """
    out: list[str] = []

    def walk(n: Any) -> None:
        if isinstance(n, str):
            out.append(n)
        elif isinstance(n, (int, float)):
            out.append(str(n))
        elif isinstance(n, list):
            for c in n:
                walk(c)
        elif isinstance(n, dict):
            # Component nodes nest under "props"; plain dicts (a Plotly
            # figure, a store payload) are walked by value.
            for key, value in n.items():
                if key in {"type", "namespace"}:
                    continue
                walk(value)

    walk(node)
    return " ".join(out)


@dataclass
class DispatchResult:
    """The outcome of one ``/_dash-update-component`` round trip."""

    status_code: int
    payload: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.status_code == 200

    @property
    def prevented(self) -> bool:
        """True when the callback declined to update anything.

        Dash answers ``204 No Content`` for a ``PreventUpdate`` or an
        all-``no_update`` return. That is a meaningful outcome, not a
        failure — it is how a validating callback rejects bad input — so it
        is surfaced as its own state rather than folded into ``ok``.
        """
        return self.status_code == 204

    def value(self, component_id: str, prop: str) -> Any:
        """Return one output value, or raise if the callback did not set it."""
        response = self.payload.get("response", {})
        if component_id not in response:
            raise AssertionError(
                f"callback returned no value for {component_id}.{prop}; it set: {sorted(response)}"
            )
        return response[component_id][prop]

    def text(self, component_id: str, prop: str = "children") -> str:
        """Return one output rendered down to its visible text."""
        return component_text(self.value(component_id, prop))


class DashDriver:
    """Executes registered callbacks through the WSGI stack, no browser."""

    def __init__(self, app: Any) -> None:
        self._app = app
        self._client = app.server.test_client()

    @property
    def client(self) -> Any:
        """The raw Flask test client, for non-callback routes."""
        return self._client

    # ── introspection ──────────────────────────────────────────

    def layout_ids(self) -> set[str]:
        """Every string ``id`` present in the initial layout tree.

        Pattern-matching (dict) ids are skipped: they identify a *family* of
        components, and Dash resolves them at dispatch time rather than
        against the static tree.
        """
        found: set[str] = set()

        def walk(node: Any) -> None:
            if node is None or isinstance(node, str):
                return
            if isinstance(node, (list, tuple)):
                for child in node:
                    walk(child)
                return
            ident = getattr(node, "id", None)
            if isinstance(ident, str):
                found.add(ident)
            walk(getattr(node, "children", None))

        layout = self._app.layout
        walk(layout() if callable(layout) else layout)
        return found

    def callback_output_ids(self) -> set[str]:
        """Every component id targeted by an ``Output`` in any callback.

        Read from ``app.callback_map`` — Dash's own registry — rather than by
        grepping source, so it stays true no matter how a callback is
        declared or which module registers it.
        """
        ids: set[str] = set()
        for key in self._app.callback_map:
            for target in _flatten_output_key(key):
                if target.startswith("{"):  # pattern-matching output
                    continue
                ids.add(target.rsplit(".", 1)[0])
        return ids

    # ── dispatch ───────────────────────────────────────────────

    @staticmethod
    def _dep_keys(declared: list[dict[str, Any]]) -> set[str]:
        """The ``"id"`` and ``"id.property"`` spellings a caller may use."""
        keys: set[str] = set()
        for dep in declared:
            if isinstance(dep["id"], str):
                keys.add(dep["id"])
                keys.add(f"{dep['id']}.{dep['property']}")
        return keys

    def _resolve(self, output: str, inputs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Find the one registered callback that produces ``output``.

        ``output`` is ``"component-id.property"``. When several callbacks
        target it (``allow_duplicate=True``), the supplied input keys pick the
        intended one — that ambiguity is real in this app: ``outlook-chart``
        is written by both the forecast callback and the replay overlay, and
        ``region-selector.value`` by three callbacks, two of which take the
        same ``url`` component and differ only in which property they watch.
        That second case is why input keys may be spelled ``"url.search"``.
        """
        candidates = [
            (key, spec)
            for key, spec in self._app.callback_map.items()
            if output in _flatten_output_key(key)
        ]
        if not candidates:
            raise KeyError(f"no registered callback outputs {output!r}")
        if len(candidates) > 1:
            wanted = set(inputs)
            narrowed = [
                (key, spec)
                for key, spec in candidates
                if wanted and wanted <= self._dep_keys(spec["inputs"])
            ]
            if len(narrowed) != 1:
                shapes = "; ".join(
                    str(sorted(f"{d['id']}.{d['property']}" for d in spec["inputs"]))
                    for _, spec in candidates
                )
                raise KeyError(
                    f"{output!r} is produced by {len(candidates)} callbacks and the supplied "
                    f"inputs {sorted(wanted)} do not select exactly one. Candidates: {shapes}"
                )
            candidates = narrowed
        return candidates[0]

    def dispatch(
        self,
        output: str,
        inputs: dict[str, Any],
        state: dict[str, Any] | None = None,
        triggered: list[str] | None = None,
    ) -> DispatchResult:
        """Invoke the callback that produces ``output`` and return its result.

        Args:
            output: ``"component-id.property"`` of one of the callback's outputs.
            inputs: value per input component id. Ids the callback declares but
                that are absent here are sent as ``None``, which is what Dash
                sends for a component that has not been touched yet.
            state: value per state component id, same convention.
            triggered: ids to report in ``changedPropIds``. Defaults to every
                supplied input, which matters for callbacks reading ``ctx``.

        Returns:
            A ``DispatchResult``. A non-200 status is returned rather than
            raised so tests can assert on failure modes deliberately.
        """
        key, spec = self._resolve(output, inputs)
        state = state or {}

        flat = _flatten_output_key(key)
        outputs: Any
        if key.startswith(".."):
            outputs = [
                {"id": target.rsplit(".", 1)[0], "property": target.rsplit(".", 1)[1]}
                for target in flat
            ]
        else:
            outputs = {"id": flat[0].rsplit(".", 1)[0], "property": flat[0].rsplit(".", 1)[1]}

        def deps(declared: list[dict[str, Any]], values: dict[str, Any]) -> list[dict[str, Any]]:
            resolved = []
            for d in declared:
                qualified = f"{d['id']}.{d['property']}"
                value = values.get(qualified, values.get(d["id"]))
                resolved.append({"id": d["id"], "property": d["property"], "value": value})
            return resolved

        def supplied(d: dict[str, Any]) -> bool:
            return d["id"] in inputs or f"{d['id']}.{d['property']}" in inputs

        body = {
            "output": key,
            "outputs": outputs,
            "inputs": deps(spec["inputs"], inputs),
            "state": deps(spec.get("state", []), state),
            "changedPropIds": triggered
            if triggered is not None
            else [f"{d['id']}.{d['property']}" for d in spec["inputs"] if supplied(d)],
        }
        response = self._client.post("/_dash-update-component", json=body)
        payload: dict[str, Any] = {}
        if response.status_code == 200:
            payload = response.get_json() or {}
        return DispatchResult(status_code=response.status_code, payload=payload)
