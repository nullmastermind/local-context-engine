"""Tests for method graph improvements: typed signatures, call site extraction,
and get_callers_of_method for all languages (Python + JS/TS/Go/Java via tree-sitter).

Tests use stdlib ast path for Python fallback (always available).
Tree-sitter tests are marked to skip gracefully if grammars aren't installed.
"""

from __future__ import annotations

import textwrap

import pytest

from corbell.core.graph.schema import DependencyEdge, MethodNode
from corbell.core.graph.sqlite_store import SQLiteGraphStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_method(mid: str, name: str, svc: str = "svc") -> MethodNode:
    return MethodNode(
        id=mid,
        repo="/repo",
        file_path="f.py",
        class_name=None,
        method_name=name,
        signature=name,
        docstring=None,
        line_start=1,
        line_end=10,
        service_id=svc,
    )


# ---------------------------------------------------------------------------
# get_callers_of_method — SQLite reverse lookup
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_store(tmp_db):
    return SQLiteGraphStore(tmp_db)


def test_get_callers_of_method_basic(graph_store):
    """m1→m2 and m3→m2: get_callers_of_method('m2') returns both callers."""
    for mid, name in [("m1", "caller_a"), ("m2", "target"), ("m3", "caller_b")]:
        graph_store.upsert_node(_make_method(mid, name))

    graph_store.upsert_edge(DependencyEdge(source_id="m1", target_id="m2", kind="method_call"))
    graph_store.upsert_edge(DependencyEdge(source_id="m3", target_id="m2", kind="method_call"))

    callers = graph_store.get_callers_of_method("m2")
    caller_ids = {m.id for m in callers}
    assert "m1" in caller_ids
    assert "m3" in caller_ids
    assert "m2" not in caller_ids


def test_get_callers_of_method_no_callers(graph_store):
    """Method with no callers returns empty list."""
    graph_store.upsert_node(_make_method("m1", "lonely"))
    assert graph_store.get_callers_of_method("m1") == []


def test_get_callers_of_method_only_method_call_edges(graph_store):
    """Only method_call edges count, not http_call edges."""
    graph_store.upsert_node(_make_method("m1", "a"))
    graph_store.upsert_node(_make_method("m2", "b"))
    graph_store.upsert_edge(DependencyEdge(source_id="m1", target_id="m2", kind="http_call"))
    # http_call should NOT be returned by get_callers_of_method
    assert graph_store.get_callers_of_method("m2") == []


# ---------------------------------------------------------------------------
# get_flows_for_method
# ---------------------------------------------------------------------------


def test_get_flows_for_method(graph_store):
    """Flow step edge is correctly returned from get_flows_for_method."""
    from corbell.core.graph.schema import FlowNode

    flow = FlowNode(
        id="flow::svc::LoginFlow",
        name="LoginFlow",
        service_id="svc",
        entry_method_id="m1",
        step_count=2,
    )
    graph_store.upsert_node(_make_method("m1", "login_handler"))
    graph_store.upsert_node(_make_method("m2", "validate_token"))
    graph_store.upsert_node(flow)
    graph_store.upsert_edge(
        DependencyEdge(
            source_id="flow::svc::LoginFlow",
            target_id="m2",
            kind="flow_step",
            metadata={"step": 2, "flow_name": "LoginFlow"},
        )
    )

    flows = graph_store.get_flows_for_method("m2")
    assert len(flows) == 1
    assert flows[0]["flow_name"] == "LoginFlow"
    assert flows[0]["step"] == 2
    assert flows[0]["entry_method_id"] == "m1"


# ---------------------------------------------------------------------------
# Python call site extraction via stdlib ast
# ---------------------------------------------------------------------------


def test_python_call_sites_extracted(tmp_path, tmp_db):
    """Python call sites are extracted via stdlib ast when tree-sitter unavailable."""
    from corbell.core.graph.method_graph import MethodGraphBuilder

    src = tmp_path / "auth.py"
    src.write_text(textwrap.dedent("""
        def validate_token(token):
            return check_signature(token)

        def check_signature(token):
            return True
    """))

    store = SQLiteGraphStore(tmp_db)
    builder = MethodGraphBuilder(store)
    result = builder.build_for_service("svc", tmp_path)

    assert result["methods"] >= 2
    # Call edge should exist: validate_token -> check_signature
    methods = store.get_methods_for_service("svc")
    callee = next(m for m in methods if m.method_name == "check_signature")
    callers = store.get_callers_of_method(callee.id)
    callee_names = [m.method_name for m in callers]
    assert "validate_token" in callee_names


def test_python_builtins_filtered(tmp_path, tmp_db):
    """Builtin calls (len, print, etc.) are not included in call edges."""
    from corbell.core.graph.method_graph import MethodGraphBuilder

    src = tmp_path / "util.py"
    src.write_text(textwrap.dedent("""
        def process(items):
            n = len(items)
            print(n)
            return sorted(items)
    """))

    store = SQLiteGraphStore(tmp_db)
    builder = MethodGraphBuilder(store)
    builder.build_for_service("svc", tmp_path)

    methods = store.get_methods_for_service("svc")
    assert len(methods) == 1
    # No call edges to builtins
    callers = store.get_callers_of_method(methods[0].id)
    assert callers == []


def test_python_typed_signature_via_ast(tmp_path, tmp_db):
    """Python methods show typed signatures from stdlib ast."""
    from corbell.core.graph.method_graph import MethodGraphBuilder

    src = tmp_path / "typed.py"
    src.write_text(textwrap.dedent("""
        def validate(token: str, scope: list) -> bool:
            return True
    """))

    store = SQLiteGraphStore(tmp_db)
    builder = MethodGraphBuilder(store)
    builder.build_for_service("svc", tmp_path)

    methods = store.get_methods_for_service("svc")
    m = next(m for m in methods if m.method_name == "validate")
    # stdlib ast path builds signature with parameter names;
    # tree-sitter path puts params in typed_signature instead.
    assert "validate" in m.signature
    sig_with_params = m.signature if "token" in m.signature else (m.typed_signature or "")
    assert "token" in sig_with_params


# ---------------------------------------------------------------------------
# Tree-sitter tests (skip if grammars not installed)
# ---------------------------------------------------------------------------


def _ts_available(lang: str) -> bool:
    try:
        from corbell.core.graph.method_graph import _get_ts_parser
        return _get_ts_parser(lang) is not None
    except Exception:
        return False


@pytest.mark.skipif(not _ts_available("javascript"), reason="tree-sitter-javascript not installed")
def test_js_call_sites_extracted(tmp_path, tmp_db):
    """JavaScript call sites are extracted by the tree-sitter path."""
    from corbell.core.graph.method_graph import MethodGraphBuilder

    src = tmp_path / "auth.js"
    src.write_text(textwrap.dedent("""
        function validateUser(token) {
            return checkPermissions(token);
        }

        function checkPermissions(token) {
            return true;
        }
    """))

    store = SQLiteGraphStore(tmp_db)
    builder = MethodGraphBuilder(store)
    result = builder.build_for_service("svc", tmp_path)

    assert result["methods"] >= 2
    methods = store.get_methods_for_service("svc")
    callee = next((m for m in methods if m.method_name == "checkPermissions"), None)
    assert callee is not None, "checkPermissions not found"

    callers = store.get_callers_of_method(callee.id)
    caller_names = [m.method_name for m in callers]
    assert "validateUser" in caller_names


@pytest.mark.skipif(not _ts_available("typescript"), reason="tree-sitter-typescript not installed")
def test_ts_call_sites_extracted(tmp_path, tmp_db):
    """TypeScript call sites are extracted by tree-sitter."""
    from corbell.core.graph.method_graph import MethodGraphBuilder

    src = tmp_path / "service.ts"
    src.write_text(textwrap.dedent("""
        function processRequest(req: Request): Response {
            const user = getUser(req.userId);
            return buildResponse(user);
        }

        function getUser(id: string): User {
            return { id };
        }

        function buildResponse(user: User): Response {
            return { user };
        }
    """))

    store = SQLiteGraphStore(tmp_db)
    builder = MethodGraphBuilder(store)
    result = builder.build_for_service("svc", tmp_path)

    assert result["methods"] >= 3
    methods = store.get_methods_for_service("svc")
    entry = next((m for m in methods if m.method_name == "processRequest"), None)
    assert entry is not None

    # processRequest calls getUser and buildResponse
    callees_of_entry = []
    for m in methods:
        if m.id != entry.id:
            callers = store.get_callers_of_method(m.id)
            if any(c.id == entry.id for c in callers):
                callees_of_entry.append(m.method_name)

    assert len(callees_of_entry) > 0, "processRequest should have call edges"


@pytest.mark.skipif(not _ts_available("go"), reason="tree-sitter-go not installed")
def test_go_call_sites_extracted(tmp_path, tmp_db):
    """Go call sites are extracted by tree-sitter."""
    from corbell.core.graph.method_graph import MethodGraphBuilder

    src = tmp_path / "handlers.go"
    src.write_text(textwrap.dedent("""
        package main

        func LoginHandler(w http.ResponseWriter, r *http.Request) {
            user := validateCredentials(r)
            writeJSON(w, user)
        }

        func validateCredentials(r *http.Request) *User {
            return nil
        }

        func writeJSON(w http.ResponseWriter, data interface{}) {
        }
    """))

    store = SQLiteGraphStore(tmp_db)
    builder = MethodGraphBuilder(store)
    result = builder.build_for_service("svc", tmp_path)

    assert result["methods"] >= 3
    methods = store.get_methods_for_service("svc")
    callee = next((m for m in methods if m.method_name == "validateCredentials"), None)
    assert callee is not None

    callers = store.get_callers_of_method(callee.id)
    assert any(m.method_name == "LoginHandler" for m in callers)
