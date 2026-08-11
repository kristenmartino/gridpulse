"""One model, one name — pinned across every user-visible surface.

The same model was labelled three ways in one product: the Models tab's
compare-models checklist said **SARIMAX**, the Forecast tab's model
segmented control said **ARIMA**, and any label derived with
``model_name.title()`` rendered **Arima** (the Overview hero metrics card,
its ensemble caption, and the data-scientist spotlight bar chart). A user
switching tabs saw two or three names for one model.

The resolution is SARIMAX, for two reasons that point the same way:

* It is what the model *is*. ``models/arima_model.py`` fits a seasonal
  order with exogenous weather regressors — the **S** and the **X** are
  both load-bearing, and both are pinned by
  ``tests/unit/test_arima_model.py`` (``ARIMA_EXOG_COLS`` must carry the
  weather columns; ``d + D <= 1`` guards the integrated component).
* It is what the rest of the project already published: README.md,
  PRD.md, TECHNICAL_SPEC.md §5.3, CLAUDE.md's module map, and
  web/landing.html.

``"arima"`` remains the internal key — the Redis payload key, the config
key, the callback value — and none of those are user-visible. This module
pins the boundary between the two.

The tests below are about the LABELS being single-valued, not about the
model. They are cheap and pure: no Dash callback dispatch, no I/O.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from components.accessibility import MODEL_DISPLAY_NAMES, model_display_name

COMPONENTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "components"

#: Spellings that must never appear as a user-visible label literal in
#: ``components/``. Each is a real spelling this product shipped.
#:
#: Matched on WORD BOUNDARIES, not substrings — ``"ARIMA" in "SARIMAX"`` is
#: True, so a substring check flags the canonical label as a violation of
#: itself. That same trap made
#: ``test_insights_extended::test_model_comparison_insight`` pass under both
#: spellings before it was tightened; it is worth getting right twice.
BANNED_LABEL_LITERALS = ("ARIMA", "Arima", "Xgboost", "XGboost")


class TestCanonicalMap:
    """``MODEL_DISPLAY_NAMES`` is the single source of truth."""

    def test_arima_key_displays_as_sarimax(self):
        """The resolved spelling, asserted directly."""
        assert MODEL_DISPLAY_NAMES["arima"] == "SARIMAX"
        assert model_display_name("arima") == "SARIMAX"

    def test_every_selectable_model_has_a_name(self):
        """The 4 user-selectable forecasts (docs/CANONICAL_FACTS.md)."""
        for key in ("prophet", "arima", "xgboost", "ensemble"):
            assert key in MODEL_DISPLAY_NAMES, f"{key} has no display name"
            assert MODEL_DISPLAY_NAMES[key], f"{key} maps to an empty label"

    def test_names_are_distinct(self):
        """Two keys sharing a label would be a different flavour of the same bug."""
        labels = list(MODEL_DISPLAY_NAMES.values())
        assert len(labels) == len(set(labels)), f"duplicate labels: {labels}"

    def test_unknown_key_returns_the_key_not_a_blank(self):
        """An unmapped key is a missing entry, not a reason to render nothing.

        Notably it must NOT fall back to ``.title()`` — that is the helper
        that produced "Arima" in the first place.
        """
        assert model_display_name("lstm") == "lstm"

    def test_title_case_is_never_the_right_derivation(self):
        """Pins WHY the map exists: ``.title()`` is wrong for both models
        whose names are not plain words. If this ever passes for both, the
        map has stopped earning its keep."""
        assert "arima".title() != MODEL_DISPLAY_NAMES["arima"]  # Arima
        assert "xgboost".title() != MODEL_DISPLAY_NAMES["xgboost"]  # Xgboost


def _component_sources() -> list[pathlib.Path]:
    return sorted(COMPONENTS_DIR.glob("*.py"))


def _string_literals(path: pathlib.Path) -> list[str]:
    """Every string CONSTANT in a module, comments and docstrings excluded.

    Parsing rather than grepping is deliberate: ``accessibility.py``'s own
    comment block names all three historical spellings, and a grep-based
    check would either flag that or need an exception broad enough to let a
    real regression through.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    return [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
    ]


@pytest.mark.parametrize("path", _component_sources(), ids=lambda p: p.name)
def test_no_component_writes_a_banned_label_literal(path):
    """No file in ``components/`` may hand-write a superseded spelling.

    This is the test that would have caught the original report: two tabs,
    two literals, no shared constant between them.
    """
    offenders = [
        (lit, banned)
        for lit in _string_literals(path)
        for banned in BANNED_LABEL_LITERALS
        if re.search(rf"\b{banned}\b", lit)
    ]
    assert not offenders, (
        f"{path.name} writes a superseded model-label spelling {offenders!r}. "
        f"Read components.accessibility.model_display_name() instead — the "
        f"canonical label is {MODEL_DISPLAY_NAMES['arima']!r}."
    )


#: The only things ``components/`` may ``.title()``. Everything else is
#: presumed to be a model key until declared otherwise.
#:
#: An allowlist rather than a "looks like a model" heuristic, because the
#: heuristic does not hold. The three instances that shipped had receivers
#: named ``model_name``, ``primary_key``, ``best_key``, and ``m`` — only the
#: first contains "model", so a name-based rule catches one of four. Inverting
#: it costs one line per legitimate use and cannot silently miss.
TITLE_CASE_ALLOWED_RECEIVERS = {
    "BASELINE_SERIES_LABEL",  # "seasonal-naive baseline" — not one of the 4 models
    "fuel",  # generation fuel type
    "largest_fuel",  # ditto
    "grade",  # MAPE governance grade
    "source",  # data-source name
}


@pytest.mark.parametrize("path", _component_sources(), ids=lambda p: p.name)
def test_no_component_derives_a_model_label_with_title_case(path):
    """``<model_key>.title()`` is how "Arima" and "Xgboost" reached the UI.

    ``.title()`` on a model key is never correct: it renders "Arima" and
    "Xgboost". The Overview hero card and its ensemble caption already
    special-cased "xgboost" by hand for exactly this reason and still let
    "Arima" through — the same bug fixed once and not generalised.

    Adding a legitimate ``.title()`` means adding its receiver to
    ``TITLE_CASE_ALLOWED_RECEIVERS`` above, which is the point: the
    declaration is the review.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "title"
        ):
            continue
        receiver = node.func.value
        names = {n.id for n in ast.walk(receiver) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(receiver) if isinstance(n, ast.Attribute)}
        if not (names & TITLE_CASE_ALLOWED_RECEIVERS):
            offenders.append((ast.unparse(receiver), node.lineno))
    assert not offenders, (
        f"{path.name} calls .title() on an undeclared receiver at {offenders!r}. "
        f"If it is a model key, use model_display_name() — .title() renders "
        f"'Arima'/'Xgboost'. If it is not, add it to "
        f"TITLE_CASE_ALLOWED_RECEIVERS in {pathlib.Path(__file__).name}."
    )


class TestSurfacesAgree:
    """The two selectors from the original report, asserted together."""

    def test_forecast_and_models_selectors_use_the_same_label(self):
        """tab_demand_outlook's segmented control vs tab_models' checklist.

        These are the exact two widgets that disagreed. Reading the label out
        of the built layout (not the source) is what makes this a real check:
        it would still fail if someone re-hardcoded a literal that happened to
        pass the AST sweep above.
        """
        from components.tab_demand_outlook import _model_segmented
        from components.tab_models import _model_selector

        def _label_for_arima(control):
            radio = control.children[1]
            return next(o["label"] for o in radio.options if o["value"] == "arima")

        forecast_label = _label_for_arima(_model_segmented())
        models_label = _label_for_arima(_model_selector())

        assert forecast_label == models_label, (
            f"Forecast tab says {forecast_label!r}, Models tab says "
            f"{models_label!r} — same model, two names."
        )
        assert forecast_label == "SARIMAX"

    def test_both_selectors_cover_the_same_four_models(self):
        """A label fix that silently dropped an option would be worse."""
        from components.tab_demand_outlook import _model_segmented
        from components.tab_models import _model_selector

        def _values(control):
            return {o["value"] for o in control.children[1].options}

        assert _values(_model_segmented()) == _values(_model_selector())
        assert _values(_model_segmented()) == {"xgboost", "prophet", "arima", "ensemble"}
