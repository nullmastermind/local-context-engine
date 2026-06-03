"""Tests for new language support: C#, Rust, Ruby, PHP.

Covers both tree-sitter (when grammar installed) and regex fallback paths.
Tests follow the same patterns as test_method_graph_improvements.py.
"""

from __future__ import annotations

import textwrap
from unittest.mock import patch

import pytest

from corbell.core.graph.sqlite_store import SQLiteGraphStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts_available(lang: str) -> bool:
    try:
        from corbell.core.graph.method_graph import _get_ts_parser
        return _get_ts_parser(lang) is not None
    except Exception:
        return False


def _build(tmp_path, tmp_db, filename: str, code: str, service_id: str = "svc"):
    """Write source file, build graph, return (store, result)."""
    from corbell.core.graph.method_graph import MethodGraphBuilder

    src = tmp_path / filename
    src.write_text(textwrap.dedent(code))
    store = SQLiteGraphStore(tmp_db)
    builder = MethodGraphBuilder(store)
    result = builder.build_for_service(service_id, tmp_path)
    return store, result


def _build_regex(tmp_path, tmp_db, filename: str, code: str, service_id: str = "svc"):
    """Build with tree-sitter disabled to test regex fallback."""
    from corbell.core.graph.method_graph import MethodGraphBuilder

    src = tmp_path / filename
    src.write_text(textwrap.dedent(code))
    store = SQLiteGraphStore(tmp_db)
    builder = MethodGraphBuilder(store)
    with patch("corbell.core.graph.method_graph._get_ts_parser", return_value=None):
        result = builder.build_for_service(service_id, tmp_path)
    return store, result


# ===================================================================
# C#
# ===================================================================

CSHARP_CODE = """\
using System;

namespace PaymentService
{
    public class PaymentProcessor
    {
        public bool ProcessPayment(string orderId, decimal amount)
        {
            var result = ValidateOrder(orderId);
            return result;
        }

        private bool ValidateOrder(string orderId)
        {
            return true;
        }

        public static int GetRetryCount()
        {
            return 3;
        }
    }
}
"""


@pytest.mark.skipif(not _ts_available("csharp"), reason="tree-sitter-c-sharp not installed")
def test_csharp_methods_extracted(tmp_path, tmp_db):
    """C# method declarations are extracted via tree-sitter."""
    store, result = _build(tmp_path, tmp_db, "Processor.cs", CSHARP_CODE)

    assert result["methods"] >= 3
    methods = store.get_methods_for_service("svc")
    names = {m.method_name for m in methods}
    assert "ProcessPayment" in names
    assert "ValidateOrder" in names
    assert "GetRetryCount" in names


@pytest.mark.skipif(not _ts_available("csharp"), reason="tree-sitter-c-sharp not installed")
def test_csharp_call_sites_extracted(tmp_path, tmp_db):
    """C# call edges are tracked between methods."""
    store, result = _build(tmp_path, tmp_db, "Processor.cs", CSHARP_CODE)

    methods = store.get_methods_for_service("svc")
    callee = next((m for m in methods if m.method_name == "ValidateOrder"), None)
    assert callee is not None, "ValidateOrder method not found"

    callers = store.get_callers_of_method(callee.id)
    caller_names = [m.method_name for m in callers]
    assert "ProcessPayment" in caller_names


def test_csharp_regex_fallback(tmp_path, tmp_db):
    """C# regex fallback finds method definitions."""
    store, result = _build_regex(tmp_path, tmp_db, "Processor.cs", CSHARP_CODE)

    assert result["methods"] >= 2
    methods = store.get_methods_for_service("svc")
    names = {m.method_name for m in methods}
    assert "ProcessPayment" in names or "ValidateOrder" in names


# ===================================================================
# Rust
# ===================================================================

RUST_CODE = """\
pub fn process_request(req: &Request) -> Response {
    let user = validate_token(&req.token);
    build_response(user)
}

fn validate_token(token: &str) -> User {
    User { id: 1 }
}

pub async fn build_response(user: User) -> Response {
    Response::ok(user)
}
"""


@pytest.mark.skipif(not _ts_available("rust"), reason="tree-sitter-rust not installed")
def test_rust_methods_extracted(tmp_path, tmp_db):
    """Rust function_item nodes are extracted via tree-sitter."""
    store, result = _build(tmp_path, tmp_db, "handler.rs", RUST_CODE)

    assert result["methods"] >= 3
    methods = store.get_methods_for_service("svc")
    names = {m.method_name for m in methods}
    assert "process_request" in names
    assert "validate_token" in names
    assert "build_response" in names


@pytest.mark.skipif(not _ts_available("rust"), reason="tree-sitter-rust not installed")
def test_rust_call_sites_extracted(tmp_path, tmp_db):
    """Rust call edges are tracked between functions."""
    store, result = _build(tmp_path, tmp_db, "handler.rs", RUST_CODE)

    methods = store.get_methods_for_service("svc")
    callee = next((m for m in methods if m.method_name == "validate_token"), None)
    assert callee is not None, "validate_token not found"

    callers = store.get_callers_of_method(callee.id)
    caller_names = [m.method_name for m in callers]
    assert "process_request" in caller_names


@pytest.mark.skipif(not _ts_available("rust"), reason="tree-sitter-rust not installed")
def test_rust_typed_signature(tmp_path, tmp_db):
    """Rust typed signatures with parameter types are extracted."""
    store, result = _build(tmp_path, tmp_db, "handler.rs", RUST_CODE)

    methods = store.get_methods_for_service("svc")
    m = next((m for m in methods if m.method_name == "validate_token"), None)
    assert m is not None
    # typed_signature should contain parameter types
    sig = m.typed_signature or m.signature
    assert "token" in sig


def test_rust_regex_fallback(tmp_path, tmp_db):
    """Rust regex fallback finds fn definitions."""
    store, result = _build_regex(tmp_path, tmp_db, "handler.rs", RUST_CODE)

    assert result["methods"] >= 3
    methods = store.get_methods_for_service("svc")
    names = {m.method_name for m in methods}
    assert "process_request" in names
    assert "validate_token" in names
    assert "build_response" in names


# ===================================================================
# Ruby
# ===================================================================

RUBY_CODE = """\
class AuthService
  def authenticate(username, password)
    user = find_user(username)
    verify_password(user, password)
  end

  def find_user(username)
    User.find_by(username: username)
  end

  def verify_password(user, password)
    BCrypt::Password.new(user.password_hash) == password
  end

  def self.default_timeout
    30
  end
end
"""


@pytest.mark.skipif(not _ts_available("ruby"), reason="tree-sitter-ruby not installed")
def test_ruby_methods_extracted(tmp_path, tmp_db):
    """Ruby method and singleton_method nodes are extracted via tree-sitter."""
    store, result = _build(tmp_path, tmp_db, "auth.rb", RUBY_CODE)

    assert result["methods"] >= 3
    methods = store.get_methods_for_service("svc")
    names = {m.method_name for m in methods}
    assert "authenticate" in names
    assert "find_user" in names
    assert "verify_password" in names


@pytest.mark.skipif(not _ts_available("ruby"), reason="tree-sitter-ruby not installed")
def test_ruby_call_sites_extracted(tmp_path, tmp_db):
    """Ruby call edges are tracked between methods."""
    store, result = _build(tmp_path, tmp_db, "auth.rb", RUBY_CODE)

    methods = store.get_methods_for_service("svc")
    callee = next((m for m in methods if m.method_name == "find_user"), None)
    assert callee is not None, "find_user not found"

    callers = store.get_callers_of_method(callee.id)
    caller_names = [m.method_name for m in callers]
    assert "authenticate" in caller_names


def test_ruby_regex_fallback(tmp_path, tmp_db):
    """Ruby regex fallback finds def and def self. methods."""
    store, result = _build_regex(tmp_path, tmp_db, "auth.rb", RUBY_CODE)

    assert result["methods"] >= 3
    methods = store.get_methods_for_service("svc")
    names = {m.method_name for m in methods}
    assert "authenticate" in names
    assert "find_user" in names
    assert "default_timeout" in names


# ===================================================================
# PHP
# ===================================================================

PHP_CODE = """\
<?php

class OrderService
{
    public function createOrder(string $customerId, array $items): Order
    {
        $validated = $this->validateItems($items);
        return $this->persistOrder($customerId, $validated);
    }

    private function validateItems(array $items): array
    {
        return array_filter($items, fn($item) => $item['qty'] > 0);
    }

    protected function persistOrder(string $customerId, array $items): Order
    {
        return new Order($customerId, $items);
    }
}

function calculateTotal(array $items): float
{
    $total = 0.0;
    foreach ($items as $item) {
        $total += $item['price'] * $item['qty'];
    }
    return $total;
}
"""


@pytest.mark.skipif(not _ts_available("php"), reason="tree-sitter-php not installed")
def test_php_methods_extracted(tmp_path, tmp_db):
    """PHP function_definition and method_declaration are extracted via tree-sitter."""
    store, result = _build(tmp_path, tmp_db, "OrderService.php", PHP_CODE)

    assert result["methods"] >= 3
    methods = store.get_methods_for_service("svc")
    names = {m.method_name for m in methods}
    assert "createOrder" in names
    assert "validateItems" in names
    assert "calculateTotal" in names


@pytest.mark.skipif(not _ts_available("php"), reason="tree-sitter-php not installed")
def test_php_call_sites_extracted(tmp_path, tmp_db):
    """PHP call edges are tracked between methods."""
    store, result = _build(tmp_path, tmp_db, "OrderService.php", PHP_CODE)

    methods = store.get_methods_for_service("svc")
    callee = next((m for m in methods if m.method_name == "validateItems"), None)
    assert callee is not None, "validateItems not found"

    callers = store.get_callers_of_method(callee.id)
    caller_names = [m.method_name for m in callers]
    assert "createOrder" in caller_names


def test_php_regex_fallback(tmp_path, tmp_db):
    """PHP regex fallback finds function and class method definitions."""
    store, result = _build_regex(tmp_path, tmp_db, "OrderService.php", PHP_CODE)

    assert result["methods"] >= 3
    methods = store.get_methods_for_service("svc")
    names = {m.method_name for m in methods}
    assert "createOrder" in names
    assert "calculateTotal" in names


# ===================================================================
# Cross-language: test/mock methods are skipped
# ===================================================================


@pytest.mark.skipif(not _ts_available("rust"), reason="tree-sitter-rust not installed")
def test_rust_skips_test_methods(tmp_path, tmp_db):
    """Rust functions starting with test_ or containing mock are skipped."""
    code = """\
    fn process_data() -> bool { true }

    fn test_process_data() { assert!(process_data()); }

    fn mock_database() -> Database { Database::new() }
    """
    store, result = _build(tmp_path, tmp_db, "lib.rs", code)

    methods = store.get_methods_for_service("svc")
    names = {m.method_name for m in methods}
    assert "process_data" in names
    assert "test_process_data" not in names
    assert "mock_database" not in names


def test_ruby_regex_skips_test_methods(tmp_path, tmp_db):
    """Ruby regex fallback skips test_ prefixed methods."""
    code = """\
    def process_order
      true
    end

    def test_process_order
      assert process_order
    end
    """
    store, result = _build_regex(tmp_path, tmp_db, "test_skip.rb", code)

    methods = store.get_methods_for_service("svc")
    names = {m.method_name for m in methods}
    assert "process_order" in names
    assert "test_process_order" not in names
