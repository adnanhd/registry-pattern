"""Tests for the recursive factory pipeline (registry.factory.build)."""

from __future__ import annotations

from typing import Any

import pytest

from registry import BuildCfg, FunctionalRegistry, TypeRegistry, build, resolve
from registry.factory import _resolve_ref, parse_ref


class _SampleModelRegistry(TypeRegistry[Any]):
    pass


class _SampleFnRegistry(FunctionalRegistry):
    pass


@pytest.fixture(autouse=True)
def _clean_registries():
    _SampleModelRegistry.clear_artifacts()
    _SampleFnRegistry.clear_artifacts()
    yield
    _SampleModelRegistry.clear_artifacts()
    _SampleFnRegistry.clear_artifacts()


class _Adder:
    def __init__(self, base: int = 0, mult: int = 1):
        self.base = base
        self.mult = mult

    def __call__(self, x: int) -> int:
        return self.base + self.mult * x


def test_build_constructs_class() -> None:
    _SampleModelRegistry.register_artifact(_Adder)
    obj = build({"type": "_Adder", "data": {"base": 5, "mult": 3}})
    assert isinstance(obj, _Adder)
    assert obj.base == 5 and obj.mult == 3
    assert obj(10) == 35


def test_build_invokes_function() -> None:
    @_SampleFnRegistry.register_artifact
    def sum_two(a: int, b: int = 0) -> int:
        return a + b

    out = build({"type": "sum_two", "data": {"a": 7, "b": 8}})
    assert out == 15


def test_build_recurses_nested_envelope() -> None:
    class _Wrapper:
        def __init__(self, inner: object) -> None:
            self.inner = inner

    _SampleModelRegistry.register_artifact(_Adder)
    _SampleModelRegistry.register_artifact(_Wrapper)

    obj = build(
        {
            "type": "_Wrapper",
            "data": {"inner": {"type": "_Adder", "data": {"base": 2}}},
        }
    )
    assert isinstance(obj, _Wrapper)
    assert isinstance(obj.inner, _Adder)
    assert obj.inner.base == 2


def test_build_resolves_dollar_ref_from_ctx() -> None:
    @_SampleFnRegistry.register_artifact
    def takes_thing(thing: Any) -> Any:
        return thing

    sentinel = object()
    out = build(
        {"type": "takes_thing", "data": {"thing": "$external"}},
        ctx={"external": sentinel},
    )
    assert out is sentinel


def test_build_resolves_dollar_ref_from_sibling() -> None:
    @_SampleFnRegistry.register_artifact
    def with_siblings(left: Any, right: int) -> int:
        return left.base + right

    _SampleModelRegistry.register_artifact(_Adder)
    # First sibling is built (becomes _Adder), second references its .base.
    out = build(
        {
            "type": "with_siblings",
            "data": {
                "left": {"type": "_Adder", "data": {"base": 4}},
                "right": "$left.base",
            },
        }
    )
    assert out == 8


def test_build_call_method_ref() -> None:
    @_SampleFnRegistry.register_artifact
    def takes_value(adder: Any, value: int) -> int:
        return value

    _SampleModelRegistry.register_artifact(_Adder)
    out = build(
        {
            "type": "takes_value",
            "data": {
                "adder": {"type": "_Adder", "data": {"base": 10}},
                "value": "$adder.base",
            },
        }
    )
    assert out == 10


def test_resolve_raises_on_missing() -> None:
    with pytest.raises(KeyError, match="not registered"):
        resolve("NotARealType")


def test_resolve_disambiguates_with_repo() -> None:
    class _AltModelRegistry(TypeRegistry[Any]):
        pass

    _SampleModelRegistry.register_artifact(_Adder)
    _AltModelRegistry.register_artifact(_Adder)
    # Without repo: ambiguous
    with pytest.raises(KeyError, match="ambiguous"):
        resolve("_Adder")
    # With repo: disambiguated
    reg, art = resolve("_Adder", repo="_AltModelRegistry")
    assert reg is _AltModelRegistry
    _AltModelRegistry.clear_artifacts()


def test_build_envelope_meta_propagates() -> None:
    _SampleModelRegistry.register_artifact(_Adder)
    cfg = BuildCfg(type="_Adder", data={"base": 1}, meta={"label": "hello"})
    obj = build(cfg)
    assert obj.__meta__ == {"label": "hello"}
    assert cfg.meta == {"label": "hello"}


def test_build_writes_meta_back_to_input_dict() -> None:
    """When cfg is passed as a dict, the dict's meta key should be updated in place."""

    class _MetaWriter:
        @classmethod
        def post_init(cls, instance, meta):
            meta["sentinel"] = "written"

    class _MetaWriterRegistry(TypeRegistry[Any]):
        @classmethod
        def post_init(cls, instance, meta):
            meta["from_post_init"] = True

    _MetaWriterRegistry.register_artifact(_Adder)
    cfg = {"type": "_Adder", "data": {"base": 1}, "meta": {}}
    build(cfg)
    assert cfg["meta"] == {"from_post_init": True}
    _MetaWriterRegistry.clear_artifacts()


def test_build_uses_noop_validator_when_requested() -> None:
    _SampleModelRegistry.register_artifact(_Adder)
    # Pass an unknown extra key; pydantic validator strips it (not in schema),
    # noop validator passes everything through unchanged.
    cfg = {"type": "_Adder", "data": {"base": 9, "unused_kwarg": "x"}}
    # noop will then try _Adder(**{"base": 9, "unused_kwarg": "x"}) -> TypeError
    with pytest.raises(TypeError):
        build(cfg, validator="noop")
    # pydantic strips unknowns (extra="ignore" default for create_model fields)
    obj = build({"type": "_Adder", "data": {"base": 9}})
    assert obj.base == 9


# ---------------------------------------------------------------------------
# $ref grammar: escape, fail-loud on malformed input
# ---------------------------------------------------------------------------


class _Holder:
    """Target for method-call refs (``$holder.value()``)."""

    def __init__(self, base: int = 0):
        self.base = base

    def value(self) -> int:
        return self.base * 2


def test_ref_escape_strips_one_dollar() -> None:
    """``$$x`` is a literal ``$x`` -- the escape, never a lookup."""
    assert _resolve_ref("$$HOME/data", {}) == "$HOME/data"


def test_ref_escape_bare_double_dollar_yields_single() -> None:
    assert _resolve_ref("$$", {}) == "$"


def test_ref_escape_wins_over_a_resolvable_name() -> None:
    """The escape is checked first: ``$$external`` stays literal even when
    ``external`` IS in scope, so escaping is never silently overridden."""
    sentinel = object()
    assert _resolve_ref("$$external", {"external": sentinel}) == "$external"


def test_ref_escape_end_to_end_through_build() -> None:
    @_SampleFnRegistry.register_artifact
    def takes_thing(thing: Any) -> Any:
        return thing

    out = build(
        {"type": "takes_thing", "data": {"thing": "$$HOME/data"}},
        ctx={"HOME": "/should/not/be/used"},
    )
    assert out == "$HOME/data"


@pytest.mark.parametrize(
    "bad",
    [
        "$model.parameters(x)",  # args in the call form
        "$1abc",  # leading digit
        "$a-b",  # hyphen is not a name char
        "$",  # nothing to resolve
        "$.leading.dot",
        "$name()extra",
    ],
)
def test_ref_malformed_raises_value_error(bad: str) -> None:
    """A malformed $-string is a grammar error, not a literal kwarg."""
    with pytest.raises(ValueError, match="malformed reference") as exc:
        _resolve_ref(bad, {"model": _Holder(), "name": _Holder()})
    msg = str(exc.value)
    assert repr(bad) in msg  # quotes the offending string
    assert "$$literal" in msg  # states the accepted forms
    assert "$name.attr" in msg


def test_ref_malformed_raises_through_build() -> None:
    """The single call site propagates the grammar error instead of passing
    the string through as a kwarg."""

    @_SampleFnRegistry.register_artifact
    def takes_thing(thing: Any) -> Any:
        return thing

    with pytest.raises(ValueError, match="malformed reference"):
        build({"type": "takes_thing", "data": {"thing": "$a-b"}})


def test_ref_unknown_scheme_raises_value_error() -> None:
    """``$scheme://`` with no registered handler used to fall through the
    scheme lookup, miss the regex, and leak out as a literal. Now it raises
    and names the schemes that ARE registered."""
    with pytest.raises(ValueError, match="unknown scheme") as exc:
        _resolve_ref("$nosuchscheme://host/path", {})
    msg = str(exc.value)
    assert repr("$nosuchscheme://host/path") in msg
    assert "nosuchscheme" in msg
    assert "file" in msg  # lists the registered schemes
    assert "$$literal" in msg


def test_ref_registered_scheme_still_resolves() -> None:
    """The registered-scheme path is untouched by the fail-loud change."""
    from registry.factory import _REF_SCHEMES, register_ref_scheme

    register_ref_scheme("tmpscheme", lambda url: {"got": url})
    try:
        assert _resolve_ref("$tmpscheme://a/b", {}) == {"got": "tmpscheme://a/b"}
    finally:
        del _REF_SCHEMES["tmpscheme"]


def test_ref_wellformed_forms_unchanged() -> None:
    """Regression guard: every accepted form resolves exactly as before."""
    holder = _Holder(base=21)
    scope = {"holder": holder, "plain": 7}
    assert _resolve_ref("$plain", scope) == 7
    assert _resolve_ref("$holder", scope) is holder
    assert _resolve_ref("$holder.base", scope) == 21
    assert _resolve_ref("$holder.value()", scope) == 42


def test_ref_unknown_name_still_raises_key_error() -> None:
    """A well-formed ref naming something absent from scope stays a KeyError
    (a lookup miss), NOT the new ValueError (a grammar error)."""
    with pytest.raises(KeyError, match="not in scope"):
        _resolve_ref("$missing", {"present": 1})
    with pytest.raises(KeyError, match="not in scope"):
        _resolve_ref("$missing.attr", {"present": 1})


def test_ref_empty_dotted_segment_fails_at_attribute_lookup() -> None:
    """``_REF_RE`` accepts an empty dotted segment (``[\\w.]*`` allows a
    trailing/doubled dot), so ``$h.`` is NOT caught by the grammar check.
    It still fails loud -- at the attribute lookup -- and is never passed
    through as a literal, which is what the fail-loud contract requires.
    Documented so the export-side translator expects AttributeError here
    rather than ValueError.
    """
    holder = _Holder(base=1)
    with pytest.raises(AttributeError):
        _resolve_ref("$holder.", {"holder": holder})
    with pytest.raises(AttributeError):
        _resolve_ref("$holder..base", {"holder": holder})


def test_parse_ref_classifies_every_accepted_form() -> None:
    """The grammar parse needs no scope: it classifies, it does not resolve."""
    assert parse_ref("$$HOME/data") == ("escape", "$HOME/data")
    assert parse_ref("$file:///cfg.yaml") == ("scheme", ("file", "file:///cfg.yaml"))
    assert parse_ref("$model") == ("local", (["model"], None))
    assert parse_ref("$model.parameters()") == ("local", (["model", "parameters"], "()"))


def test_parse_ref_rejects_the_same_strings_resolve_rejects() -> None:
    """One grammar, two callers: whatever ``_resolve_ref`` calls malformed,
    ``parse_ref`` calls malformed -- with no scope and no I/O."""
    for bad in ("$model.parameters(x)", "$1abc", "$a-b", "$", "$.leading.dot"):
        with pytest.raises(ValueError, match="malformed reference"):
            parse_ref(bad)
    with pytest.raises(ValueError, match="unknown scheme"):
        parse_ref("$nosuchscheme://host/path")


def test_parse_ref_does_not_call_the_scheme_handler() -> None:
    """A static checker must be able to grammar-check ``$https://...``
    without fetching it, so the parse classifies and stops."""
    from registry.factory import _REF_SCHEMES, register_ref_scheme

    calls = []
    register_ref_scheme("tmpparse", lambda url: calls.append(url))
    try:
        assert parse_ref("$tmpparse://a/b") == ("scheme", ("tmpparse", "tmpparse://a/b"))
        assert calls == []
    finally:
        del _REF_SCHEMES["tmpparse"]
