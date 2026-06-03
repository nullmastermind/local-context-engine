"""Tests for graph SQLite store."""

import pytest

from corbell.core.graph.schema import (
    DataStoreNode,
    DependencyEdge,
    MethodNode,
    QueueNode,
    ServiceNode,
)
from corbell.core.graph.sqlite_store import SQLiteGraphStore


@pytest.fixture
def store(tmp_db):
    return SQLiteGraphStore(tmp_db)


def test_upsert_and_get_service(store):
    node = ServiceNode(id="auth-service", name="Auth Service", repo="/repos/auth", language="python")
    store.upsert_node(node)
    retrieved = store.get_service("auth-service")
    assert retrieved is not None
    assert retrieved.id == "auth-service"
    assert retrieved.language == "python"


def test_upsert_datastore(store):
    node = DataStoreNode(id="ds:auth:postgres", kind="postgres", name="postgres-db")
    store.upsert_node(node)
    # Should not crash and should be in summary
    summary = store.get_all_nodes_summary()
    assert summary["nodes"].get("datastore", 0) == 1


def test_upsert_queue(store):
    node = QueueNode(id="q:payments:sqs", kind="sqs", name="payment-queue")
    store.upsert_node(node)
    summary = store.get_all_nodes_summary()
    assert summary["nodes"].get("queue", 0) == 1


def test_upsert_edge(store):
    s1 = ServiceNode(id="svc-a", name="A", repo="/a", language="python")
    s2 = ServiceNode(id="svc-b", name="B", repo="/b", language="python")
    store.upsert_node(s1)
    store.upsert_node(s2)
    edge = DependencyEdge(source_id="svc-a", target_id="svc-b", kind="http_call")
    store.upsert_edge(edge)
    deps = store.get_dependencies("svc-a")
    assert len(deps) == 1
    assert deps[0].target_id == "svc-b"
    assert deps[0].kind == "http_call"


def test_get_dependents(store):
    s1 = ServiceNode(id="svc-a", name="A", repo="/a", language="python")
    s2 = ServiceNode(id="svc-b", name="B", repo="/b", language="python")
    store.upsert_node(s1)
    store.upsert_node(s2)
    store.upsert_edge(DependencyEdge(source_id="svc-a", target_id="svc-b", kind="http_call"))
    dependents = store.get_dependents("svc-b")
    assert len(dependents) == 1
    assert dependents[0].source_id == "svc-a"


def test_get_all_services(store):
    for i in range(3):
        store.upsert_node(ServiceNode(id=f"svc-{i}", name=f"Service {i}", repo="/r", language="python"))
    svcs = store.get_all_services()
    assert len(svcs) == 3


def test_upsert_and_get_method(store):
    m = MethodNode(
        id="svc::file.py::MyClass.my_method",
        repo="/repos/svc",
        file_path="file.py",
        class_name="MyClass",
        method_name="my_method",
        signature="def my_method(self, x: int)",
        docstring="Does something.",
        line_start=10,
        line_end=25,
        service_id="svc",
    )
    store.upsert_node(m)
    retrieved = store.get_method(m.id)
    assert retrieved is not None
    assert retrieved.method_name == "my_method"
    assert retrieved.class_name == "MyClass"


def test_get_methods_for_service(store):
    for i in range(5):
        store.upsert_node(MethodNode(
            id=f"svc::f{i}.py::func",
            repo="/r",
            file_path=f"f{i}.py",
            class_name=None,
            method_name=f"func_{i}",
            signature=f"def func_{i}()",
            docstring=None,
            line_start=1,
            line_end=10,
            service_id="svc",
        ))
    methods = store.get_methods_for_service("svc")
    assert len(methods) == 5


def test_call_path_bfs(store):
    # Build a chain: m1 → m2 → m3
    for i in range(1, 4):
        store.upsert_node(MethodNode(
            id=f"m{i}", repo="/r", file_path="f.py", class_name=None,
            method_name=f"m{i}", signature=f"def m{i}()", docstring=None,
            line_start=i, line_end=i+5, service_id="svc",
        ))
    store.upsert_edge(DependencyEdge(source_id="m1", target_id="m2", kind="method_call"))
    store.upsert_edge(DependencyEdge(source_id="m2", target_id="m3", kind="method_call"))
    paths = store.get_call_path("m1", "m3")
    assert len(paths) >= 1
    assert paths[0] == ["m1", "m2", "m3"]


def test_clear(store):
    store.upsert_node(ServiceNode(id="svc", name="S", repo="/r", language="python"))
    store.clear()
    assert store.get_all_services() == []
    summary = store.get_all_nodes_summary()
    assert summary["edges"] == 0
