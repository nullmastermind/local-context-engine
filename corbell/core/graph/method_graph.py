"""Method-call AST graph builder.

Builds method-level call graphs from source code using tree-sitter for accurate
multi-language parsing. Falls back to Python ``ast`` for Python files when
tree-sitter is unavailable, and to lightweight regex for other languages.

Supported languages (via tree-sitter):
    Python, JavaScript, TypeScript, TSX, JSX, Go, Java

Install tree-sitter grammars:
    pip install "corbell[treesitter]"
    # or individually:
    pip install tree-sitter tree-sitter-python tree-sitter-javascript \\
                tree-sitter-typescript tree-sitter-go tree-sitter-java
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from corbell.core.graph.schema import DependencyEdge, GraphStore, MethodNode
from corbell.core.gitignore import load_gitignore

# ---------------------------------------------------------------------------
# Tree-sitter setup (optional dependency)
# ---------------------------------------------------------------------------

try:
    import tree_sitter  # noqa: F401
    from tree_sitter import Language, Parser as TSParser
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False

# Mapping: our language name -> (tree-sitter module name, language() callable attr)
_TS_MODULES: Dict[str, str] = {
    "python":     "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "tsx":        "tree_sitter_typescript",  # same package, different grammar fn
    "go":         "tree_sitter_go",
    "java":       "tree_sitter_java",
    "csharp":     "tree_sitter_c_sharp",
    "rust":       "tree_sitter_rust",
    "ruby":       "tree_sitter_ruby",
    "php":        "tree_sitter_php",
}

# Which AST node types to treat as function/method definitions per language
_TS_TARGET_NODES: Dict[str, Set[str]] = {
    "python": {
        "function_definition",
        "async_function_definition",
    },
    "javascript": {
        "function_declaration",
        "function_expression",
        "generator_function_declaration",
        "arrow_function",
        "method_definition",
    },
    "typescript": {
        "function_declaration",
        "function_expression",
        "generator_function_declaration",
        "arrow_function",
        "method_definition",
        "ambient_declaration",   # declare function ...
    },
    "tsx": {
        "function_declaration",
        "function_expression",
        "generator_function_declaration",
        "arrow_function",
        "method_definition",
    },
    "go": {
        "function_declaration",
        "method_declaration",
    },
    "java": {
        "method_declaration",
        "constructor_declaration",
    },
    "csharp": {
        "method_declaration",
        "constructor_declaration",
        "local_function_statement",
    },
    "rust": {
        "function_item",
    },
    "ruby": {
        "method",
        "singleton_method",
    },
    "php": {
        "function_definition",
        "method_declaration",
    },
}

# Child field names that hold the identifier for each language's function node
_TS_NAME_FIELDS: Dict[str, List[str]] = {
    "python":     ["name"],
    "javascript": ["name"],
    "typescript": ["name"],
    "go":         ["name"],
    "java":       ["name"],
    "csharp":     ["name"],
    "rust":       ["name"],
    "ruby":       ["name"],
    "php":        ["name"],
}

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", "venv", "env", ".venv", "tests", "__tests__",
    ".pytest_cache", "dist", "build", "coverage", ".next", ".nuxt",
    ".svelte-kit", ".cache", "out", "__tests__", ".turbo", ".vercel",
    "storybook-static", ".storybook",
}
_EXT_LANG = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".tsx":  "tsx",        # tsx uses a separate tree-sitter grammar (language_tsx)
    ".jsx":  "javascript",
    ".go":   "go",
    ".java": "java",
    ".cs":   "csharp",
    ".rs":   "rust",
    ".rb":   "ruby",
    ".php":  "php",
}

# ---------------------------------------------------------------------------
# Call site node types per language (for extracting function calls)
# ---------------------------------------------------------------------------

_TS_CALL_SITE_NODES: Dict[str, Set[str]] = {
    "python":     {"call"},
    "javascript": {"call_expression", "new_expression"},
    "typescript": {"call_expression", "new_expression"},
    "tsx":        {"call_expression", "new_expression"},
    "go":         {"call_expression"},
    "java":       {"method_invocation", "object_creation_expression"},
    "csharp":     {"invocation_expression", "object_creation_expression"},
    "rust":       {"call_expression", "macro_invocation"},
    "ruby":       {"call"},
    "php":        {"function_call_expression", "member_call_expression", "scoped_call_expression", "object_creation_expression"},
}

# ---------------------------------------------------------------------------
# Builtin blocklist — filter high-noise language builtins from call graph
# ---------------------------------------------------------------------------

_BUILTIN_BLOCKLIST: Dict[str, Set[str]] = {
    "python": {
        "print", "len", "range", "enumerate", "zip", "map", "filter",
        "sorted", "reversed", "list", "dict", "set", "tuple", "str",
        "int", "float", "bool", "bytes", "type", "isinstance", "issubclass",
        "hasattr", "getattr", "setattr", "delattr", "super", "object",
        "open", "repr", "hash", "id", "hex", "oct", "bin", "abs", "round",
        "min", "max", "sum", "all", "any", "next", "iter", "vars",
        "format", "input", "exec", "eval", "compile", "globals", "locals",
        "staticmethod", "classmethod", "property", "append", "extend",
        "items", "keys", "values", "get", "update", "pop", "copy", "join",
        "split", "strip", "replace", "startswith", "endswith", "decode",
        "encode", "lower", "upper", "format_map",
    },
    "javascript": {
        "console", "log", "error", "warn", "info", "debug", "assert",
        "setTimeout", "setInterval", "clearTimeout", "clearInterval",
        "setImmediate", "clearImmediate", "queueMicrotask",
        "Promise", "resolve", "reject", "then", "catch", "finally", "all",
        "fetch", "JSON", "parse", "stringify", "Math", "Date", "Array",
        "Object", "String", "Number", "Boolean", "Symbol", "BigInt",
        "parseInt", "parseFloat", "isNaN", "isFinite", "encodeURIComponent",
        "decodeURIComponent", "encodeURI", "decodeURI", "require",
        "map", "filter", "reduce", "forEach", "find", "findIndex",
        "push", "pop", "shift", "unshift", "splice", "slice", "join",
        "toString", "valueOf", "hasOwnProperty", "includes", "indexOf",
        "addEventListener", "removeEventListener", "emit", "on", "off",
        "next", "return", "throw", "keys", "values", "entries", "assign",
        "useState", "useEffect", "useContext", "useRef", "useMemo",
        "useCallback", "useReducer", "useLayoutEffect", "createContext",
        "createElement", "render", "it", "describe", "expect", "test",
        "beforeEach", "afterEach", "beforeAll", "afterAll", "jest",
    },
    "go": {
        "make", "len", "cap", "append", "copy", "delete", "close",
        "panic", "recover", "print", "println", "new", "real", "imag",
        "Errorf", "Sprintf", "Printf", "Println", "Fprintf", "Scanf",
        "Error", "String", "Format", "Marshal", "Unmarshal",
        "Fatal", "Fatalf", "Log", "Logf",
    },
    "java": {
        "println", "print", "printf", "format", "toString", "hashCode",
        "equals", "compareTo", "length", "size", "isEmpty", "contains",
        "add", "get", "put", "remove", "clear", "iterator", "next",
        "append", "insert", "delete", "substring", "charAt", "indexOf",
        "parseInt", "parseLong", "parseDouble", "parseFloat",
        "valueOf", "of", "ofNullable", "orElse", "isPresent", "get",
        "stream", "collect", "toList", "toMap", "filter", "map",
        "forEach", "anyMatch", "allMatch", "findFirst",
    },
    "csharp": {
        "WriteLine", "Write", "ToString", "Equals", "GetHashCode", "GetType",
        "ReferenceEquals", "Parse", "TryParse", "Format", "Join", "Concat",
        "IsNullOrEmpty", "IsNullOrWhiteSpace", "Select", "Where", "ToList",
        "ToArray", "FirstOrDefault", "Any", "All", "Count", "Max", "Min",
        "Sum", "Add", "Remove", "Clear", "Contains", "IndexOf", "Substring",
    },
    "rust": {
        "println", "print", "format", "panic", "unwrap", "expect",
        "clone", "to_string", "into", "from", "as_ref", "as_mut",
        "len", "is_empty", "push", "pop", "insert", "remove", "clear",
        "iter", "iter_mut", "into_iter", "map", "filter", "collect",
        "any", "all", "find", "Ok", "Err", "Some", "None",
    },
    "ruby": {
        "puts", "print", "p", "printf", "sprintf", "raise", "fail",
        "require", "require_relative", "include", "extend", "prepend",
        "to_s", "to_i", "to_f", "to_a", "to_h", "to_sym", "class",
        "is_a?", "kind_of?", "instance_of?", "respond_to?", "nil?",
        "empty?", "length", "size", "push", "pop", "shift", "unshift",
        "map", "select", "reject", "reduce", "inject", "each", "find",
    },
    "php": {
        "echo", "print", "print_r", "var_dump", "var_export", "printf",
        "sprintf", "die", "exit", "isset", "empty", "unset", "count",
        "sizeof", "array_push", "array_pop", "array_shift", "array_unshift",
        "array_map", "array_filter", "array_reduce", "array_keys", "array_values",
        "in_array", "explode", "implode", "str_replace", "substr", "strlen",
        "strpos", "strtolower", "strtoupper", "trim", "json_encode", "json_decode",
        "Exception", "RuntimeException", "InvalidArgumentException",
    },
}
# Add typescript as alias of javascript builtins
_BUILTIN_BLOCKLIST["typescript"] = _BUILTIN_BLOCKLIST["javascript"]
_BUILTIN_BLOCKLIST["tsx"] = _BUILTIN_BLOCKLIST["javascript"]


# ---------------------------------------------------------------------------
# Parser cache
# ---------------------------------------------------------------------------

_parser_cache: Dict[str, Any] = {}  # lang -> TSParser | None


def _get_ts_parser(lang: str) -> Optional[Any]:
    """Return a cached tree-sitter Parser for *lang*, or None if unavailable."""
    if not _TS_AVAILABLE:
        return None
    if lang in _parser_cache:
        return _parser_cache[lang]

    module_name = _TS_MODULES.get(lang)
    parser = None
    if module_name:
        try:
            mod = __import__(module_name)
            # tree_sitter_typescript exposes two grammars:
            #   language_typescript() for .ts files
            #   language_tsx()        for .tsx files (JSX-aware)
            if lang == "tsx" and hasattr(mod, "language_tsx"):
                lang_obj = Language(mod.language_tsx())
            elif lang == "typescript" and hasattr(mod, "language_typescript"):
                lang_obj = Language(mod.language_typescript())
            elif lang == "php" and hasattr(mod, "language_php"):
                lang_obj = Language(mod.language_php())
            elif hasattr(mod, "language"):
                lang_obj = Language(mod.language())
            else:
                raise AttributeError(f"No language() callable in {module_name}")
            p = TSParser(lang_obj)
            parser = p
        except Exception:
            parser = None

    _parser_cache[lang] = parser
    return parser


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


class MethodGraphBuilder:
    """Extract method nodes and call edges, store in GraphStore."""

    def __init__(self, graph_store: GraphStore):
        self.store = graph_store

    def build_for_service(
        self, service_id: str, repo_path: Path, file_list: Optional[List[Path]] = None
    ) -> Dict[str, Any]:
        """Scan *repo_path* and populate method nodes + call edges.

        Uses tree-sitter for all supported languages when the grammar packages
        are installed. Falls back to Python ``ast`` for Python files, and to
        lightweight regex for JS/TS/Go/Java when tree-sitter is unavailable.

        Args:
            service_id: Identifier for the owning service.
            repo_path: Root directory of the repository to scan.
            file_list: Optional pre-filtered list of Path objects to scan.
                When provided, skips rglob/gitignore/skip_dirs filtering —
                the caller is responsible for pre-filtering. Falls back to
                the original rglob behavior when None.

        Returns:
            Summary dict with ``methods``, ``calls``, ``files_scanned``, ``ts_available``.
        """
        all_methods: Dict[str, Dict] = {}
        all_calls: List[Dict] = []
        files_scanned = 0

        if file_list is not None:
            # Pre-filtered list from caller — skip all filtering
            for fp in file_list:
                lang = _EXT_LANG.get(fp.suffix)
                if not lang:
                    continue
                files_scanned += 1
                result = self._analyze_file(fp, service_id, lang)
                for m in result["methods"]:
                    all_methods[m["id"]] = m
                all_calls.extend(result["calls"])
        else:
            gitignore_spec = load_gitignore(Path(repo_path))

            for fp in Path(repo_path).rglob("*"):
                if not fp.is_file():
                    continue
                # Only skip if the immediate parent directory name is in SKIP_DIRS
                # (avoids false-positives from matching path segments like 'corbel')
                rel = fp.relative_to(repo_path)
                if any(part in _SKIP_DIRS for part in rel.parts):
                    continue
                if gitignore_spec.match_file(str(rel).replace("\\", "/")):
                    continue
                lang = _EXT_LANG.get(fp.suffix)
                if not lang:
                    continue
                files_scanned += 1
                result = self._analyze_file(fp, service_id, lang)
                for m in result["methods"]:
                    all_methods[m["id"]] = m
                all_calls.extend(result["calls"])

        # Persist method nodes (batched)
        method_nodes = []
        for method_id, info in all_methods.items():
            node = MethodNode(
                id=method_id,
                repo=str(repo_path),
                file_path=info["file_path"],
                class_name=info.get("class_name"),
                method_name=info["name"],
                signature=info.get("signature", info["name"]),
                docstring=info.get("docstring"),
                line_start=info.get("line_number", 0),
                line_end=info.get("line_end", info.get("line_number", 0)),
                service_id=service_id,
                typed_signature=info.get("typed_signature"),
            )
            method_nodes.append(node)
        self.store.upsert_nodes_batch(method_nodes)

        # Build and persist call graph edges (batched)
        call_graph = self._build_call_graph(all_methods, all_calls)
        edge_objects = []
        for caller_id, callee_id, meta in call_graph:
            edge_objects.append(
                DependencyEdge(
                    source_id=caller_id,
                    target_id=callee_id,
                    kind="method_call",
                    metadata=meta,
                )
            )
        self.store.upsert_edges_batch(edge_objects)

        return {
            "methods": len(all_methods),
            "calls": len(call_graph),
            "files_scanned": files_scanned,
            "ts_available": _TS_AVAILABLE,
        }


    # ------------------------------------------------------------------ #
    # Dispatch                                                             #
    # ------------------------------------------------------------------ #

    def _analyze_file(self, fp: Path, service_id: str, lang: str) -> Dict:
        try:
            content = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return {"methods": [], "calls": []}

        # 1. Try tree-sitter
        parser = _get_ts_parser(lang)
        if parser is not None:
            return self._analyze_with_tree_sitter(fp, content, service_id, lang, parser)

        # 2. Python-specific fallback: stdlib ast (accurate)
        if lang == "python":
            return self._analyze_python_ast(fp, content, service_id)

        # 3. Last resort: regex (JS/TS/Go/Java when tree-sitter is absent)
        return self._analyze_regex_fallback(fp, content, service_id, lang)

    def _make_method_id(self, service_id: str, fp: Path, full_name: str) -> str:
        return f"{service_id}::{fp.name}::{full_name}"

    # ------------------------------------------------------------------ #
    # Tree-sitter analyzer (all languages)                                 #
    # ------------------------------------------------------------------ #

    def _analyze_with_tree_sitter(
        self,
        fp: Path,
        content: str,
        service_id: str,
        lang: str,
        parser: Any,
    ) -> Dict:
        """Parse *content* with tree-sitter and extract method nodes + call sites."""
        methods: List[Dict] = []
        calls: List[Dict] = []

        try:
            tree = parser.parse(bytes(content, "utf-8"))
        except Exception:
            return {"methods": [], "calls": []}

        target_node_types = _TS_TARGET_NODES.get(lang, set())
        call_site_types = _TS_CALL_SITE_NODES.get(lang, set())
        builtins = _BUILTIN_BLOCKLIST.get(lang, set())
        def _node_name(node) -> Optional[str]:
            """Extract the identifier name from a function/method node."""
            # 1. Try matching identifier child that is exactly the "name" field
            for child in node.children:
                if child.type == "identifier" and child == node.child_by_field_name("name"):
                    return child.text.decode("utf-8", errors="ignore")
            # 2. Try via the "name" field directly (PHP uses node type "name")
            name_field = node.child_by_field_name("name")
            if name_field is not None:
                return name_field.text.decode("utf-8", errors="ignore")
            # 3. Fall back to first identifier child
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8", errors="ignore")
            return None

        def _receiver_or_class(node) -> Optional[str]:
            """For Go method_declaration, extract the receiver type name."""
            recv = node.child_by_field_name("receiver")
            if recv:
                for sub in recv.children:
                    if sub.type in ("type_identifier", "pointer_type", "qualified_type"):
                        return sub.text.decode("utf-8", errors="ignore").lstrip("*")
            return None

        def _extract_callee_name(node) -> Optional[str]:
            """Extract the called function/method name from a call site node."""
            if lang == "python":
                func = node.child_by_field_name("function")
                if func is None:
                    return None
                if func.type == "identifier":
                    return func.text.decode("utf-8", errors="ignore")
                if func.type == "attribute":
                    attr = func.child_by_field_name("attribute")
                    if attr:
                        return attr.text.decode("utf-8", errors="ignore")
            elif lang in ("javascript", "typescript", "tsx"):
                if node.type == "new_expression":
                    # new MyClass(...) — get the constructor name
                    ctor = node.child_by_field_name("constructor")
                    if ctor and ctor.type == "identifier":
                        return ctor.text.decode("utf-8", errors="ignore")
                    return None
                func = node.child_by_field_name("function")
                if func is None:
                    return None
                if func.type == "identifier":
                    return func.text.decode("utf-8", errors="ignore")
                if func.type in ("member_expression", "subscript_expression"):
                    prop = func.child_by_field_name("property")
                    if prop:
                        return prop.text.decode("utf-8", errors="ignore")
            elif lang == "go":
                func = node.child_by_field_name("function")
                if func is None:
                    return None
                if func.type == "identifier":
                    return func.text.decode("utf-8", errors="ignore")
                if func.type == "selector_expression":
                    field = func.child_by_field_name("field")
                    if field:
                        return field.text.decode("utf-8", errors="ignore")
            elif lang == "java":
                if node.type == "object_creation_expression":
                    type_node = node.child_by_field_name("type")
                    if type_node:
                        return type_node.text.decode("utf-8", errors="ignore")
                    return None
                name = node.child_by_field_name("name")
                if name:
                    return name.text.decode("utf-8", errors="ignore")
            elif lang == "csharp":
                if node.type == "object_creation_expression":
                    t = node.child_by_field_name("type")
                    if t:
                        return t.text.decode("utf-8", errors="ignore")
                    return None
                func = node.child_by_field_name("function")
                if func is None:
                    return None
                if func.type == "identifier":
                    return func.text.decode("utf-8", errors="ignore")
                if func.type == "member_access_expression":
                    name = func.child_by_field_name("name")
                    if name:
                        return name.text.decode("utf-8", errors="ignore")
            elif lang == "rust":
                func = node.child_by_field_name("function")
                if func:
                    if func.type in ("identifier", "scoped_identifier"):
                        return func.text.decode("utf-8", errors="ignore")
                    elif func.type == "field_expression":
                        field = func.child_by_field_name("field")
                        if field:
                            return field.text.decode("utf-8", errors="ignore")
                elif node.type == "macro_invocation":
                    mac = node.child_by_field_name("macro")
                    if mac:
                        return mac.text.decode("utf-8", errors="ignore")
            elif lang == "ruby":
                method = node.child_by_field_name("method")
                if method:
                    return method.text.decode("utf-8", errors="ignore")
            elif lang == "php":
                if node.type == "object_creation_expression":
                    cls = node.child_by_field_name("class")
                    if cls:
                        return cls.text.decode("utf-8", errors="ignore")
                    return None
                name_node = node.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode("utf-8", errors="ignore")
            return None

        def _extract_typed_signature(node) -> str:
            """Build a typed signature string like ``validate(token: str) -> bool``."""
            name = _node_name(node) or "?"
            params_node = node.child_by_field_name("parameters")
            param_strs: List[str] = []

            if params_node:
                for param in params_node.named_children:
                    if lang in ("javascript", "typescript", "tsx"):
                        pattern = (
                            param.child_by_field_name("pattern")
                            or param.child_by_field_name("name")
                        )
                        type_ann = param.child_by_field_name("type")
                        pname = pattern.text.decode("utf-8", "ignore") if pattern else ""
                        if type_ann:
                            raw_t = type_ann.text.decode("utf-8", "ignore").strip().lstrip(":").strip()
                            param_strs.append(f"{pname}: {raw_t}" if pname else raw_t)
                        elif pname:
                            param_strs.append(pname)

                    elif lang == "python":
                        if param.type in (
                            "typed_parameter", "typed_default_parameter"
                        ):
                            pname = ""
                            ptype = ""
                            for child in param.children:
                                if child.type == "identifier" and not pname:
                                    pname = child.text.decode("utf-8", "ignore")
                                elif child.type == "type":
                                    ptype = child.text.decode("utf-8", "ignore")
                            param_strs.append(f"{pname}: {ptype}" if ptype else pname)
                        elif param.type in ("identifier", "list_splat_pattern", "dictionary_splat_pattern"):
                            param_strs.append(param.text.decode("utf-8", "ignore"))
                        elif param.type == "default_parameter":
                            n = param.child_by_field_name("name")
                            if n:
                                param_strs.append(n.text.decode("utf-8", "ignore"))

                    elif lang == "go":
                        pnames: List[str] = []
                        ptype = ""
                        for child in param.children:
                            if child.type == "identifier":
                                pnames.append(child.text.decode("utf-8", "ignore"))
                            elif child.type in (
                                "type_identifier", "pointer_type", "qualified_type",
                                "slice_type", "array_type", "map_type", "interface_type",
                            ):
                                ptype = child.text.decode("utf-8", "ignore")
                        if pnames:
                            param_strs.append(
                                f"{' '.join(pnames)} {ptype}".strip() if ptype else " ".join(pnames)
                            )

                    elif lang == "java":
                        pname_node = param.child_by_field_name("name")
                        ptype_node = param.child_by_field_name("type")
                        if pname_node and ptype_node:
                            param_strs.append(
                                f"{ptype_node.text.decode('utf-8','ignore')} "
                                f"{pname_node.text.decode('utf-8','ignore')}"
                            )

                    elif lang == "csharp":
                        pname_node = param.child_by_field_name("name")
                        ptype_node = param.child_by_field_name("type")
                        if pname_node and ptype_node:
                            param_strs.append(
                                f"{ptype_node.text.decode('utf-8','ignore')} "
                                f"{pname_node.text.decode('utf-8','ignore')}"
                            )
                        elif param.type == "parameter":
                            param_strs.append(param.text.decode("utf-8", "ignore"))

                    elif lang == "rust":
                        pat = param.child_by_field_name("pattern")
                        typ = param.child_by_field_name("type")
                        if pat and typ:
                            param_strs.append(
                                f"{pat.text.decode('utf-8','ignore')}: {typ.text.decode('utf-8','ignore')}"
                            )
                        else:
                            param_strs.append(param.text.decode("utf-8", "ignore"))

                    elif lang == "ruby":
                        if param.type in ("identifier", "keyword_parameter", "optional_parameter"):
                            param_strs.append(param.text.decode("utf-8", "ignore"))

                    elif lang == "php":
                        pname_node = param.child_by_field_name("name")
                        ptype_node = param.child_by_field_name("type")
                        pstr = ""
                        if ptype_node:
                            pstr += ptype_node.text.decode("utf-8", "ignore") + " "
                        if pname_node:
                            pstr += pname_node.text.decode("utf-8", "ignore")
                        if pstr:
                            param_strs.append(pstr.strip())

            params_str = ", ".join(param_strs)

            # Return type
            ret_node = node.child_by_field_name("return_type")
            if ret_node:
                ret_raw = ret_node.text.decode("utf-8", "ignore").strip()
                # Strip leading ':' (TS) or '->' (Python ts node already has it stripped)
                ret_clean = ret_raw.lstrip(":->").strip().lstrip(">:").strip()
                if ret_clean:
                    return f"{name}({params_str}) -> {ret_clean}"
            return f"{name}({params_str})"

        def traverse(
            node,
            enclosing_class: Optional[str] = None,
            parent=None,
            enclosing_method_id: Optional[str] = None,
        ) -> None:
            # Track class/struct/interface context
            if node.type in {"class_declaration", "class_definition",
                              "struct_type", "type_declaration",
                              "interface_declaration"}:
                name_child = node.child_by_field_name("name")
                cls_name = (
                    name_child.text.decode("utf-8", errors="ignore")
                    if name_child else None
                )
                for child in node.children:
                    traverse(
                        child,
                        enclosing_class=cls_name or enclosing_class,
                        parent=node,
                        enclosing_method_id=enclosing_method_id,
                    )
                return

            current_method_id = enclosing_method_id  # inherited default

            if node.type in target_node_types:
                raw_name = _node_name(node)

                # For Go method_declaration, use receiver type as class
                eff_class = enclosing_class
                if lang == "go" and node.type == "method_declaration":
                    eff_class = _receiver_or_class(node) or eff_class

                # Arrow functions / function expressions without their own name
                if raw_name is None and node.type in {
                    "arrow_function", "function_expression", "generator_function",
                }:
                    if parent and parent.type == "variable_declarator":
                        name_child = parent.child_by_field_name("name")
                        if name_child:
                            raw_name = name_child.text.decode("utf-8", errors="ignore")

                if raw_name:
                    # Skip test and mock methods
                    lower_name = raw_name.lower()
                    if lower_name.startswith("test_") or "mock" in lower_name:
                        return

                    full = f"{eff_class}.{raw_name}" if eff_class else raw_name
                    mid = self._make_method_id(service_id, fp, full)
                    line_start = node.start_point[0] + 1
                    line_end = node.end_point[0] + 1

                    # Python docstring extraction
                    docstring: Optional[str] = None
                    if lang == "python" and node.children:
                        body = node.child_by_field_name("body")
                        if body and body.children:
                            first = body.children[0]
                            if first.type == "expression_statement":
                                ds_node = first.children[0] if first.children else None
                                if ds_node and ds_node.type == "string":
                                    docstring = ds_node.text.decode(
                                        "utf-8", errors="ignore"
                                    ).strip("\"'")

                    typed_sig = _extract_typed_signature(node)

                    methods.append({
                        "id": mid,
                        "name": raw_name,
                        "full_name": full,
                        "class_name": eff_class,
                        "file_path": str(fp),
                        "line_number": line_start,
                        "line_end": line_end,
                        "signature": raw_name,        # plain name (backward compat)
                        "typed_signature": typed_sig,  # NEW: full typed form
                        "docstring": docstring,
                        "service_id": service_id,
                    })
                    current_method_id = mid  # children see us as enclosing method

            elif call_site_types and node.type in call_site_types and enclosing_method_id:
                # Extract call site
                callee = _extract_callee_name(node)
                if callee and callee not in builtins:
                    calls.append({
                        "caller_id": enclosing_method_id,
                        "callee_name": callee,
                        "line_number": node.start_point[0] + 1,
                    })

            for child in node.children:
                traverse(
                    child,
                    enclosing_class=enclosing_class,
                    parent=node,
                    enclosing_method_id=current_method_id,
                )

        traverse(tree.root_node)
        return {"methods": methods, "calls": calls}

    # ------------------------------------------------------------------ #
    # Python ast fallback                                                  #
    # ------------------------------------------------------------------ #

    def _analyze_python_ast(self, fp: Path, content: str, service_id: str) -> Dict:
        """Use Python's stdlib ast for accurate extraction when tree-sitter is absent."""
        methods: List[Dict] = []
        calls: List[Dict] = []

        try:
            tree = ast.parse(content, filename=str(fp))
        except SyntaxError:
            return {"methods": [], "calls": []}

        class _Visitor(ast.NodeVisitor):
            def __init__(self_inner):
                self_inner.current_class: Optional[str] = None
                self_inner.current_method_id: Optional[str] = None

            def visit_ClassDef(self_inner, node):
                old = self_inner.current_class
                self_inner.current_class = node.name
                self_inner.generic_visit(node)
                self_inner.current_class = old

            def _visit_func(self_inner, node):
                mname = node.name
                # Skip test and mock methods
                lower_name = mname.lower()
                if lower_name.startswith("test_") or "mock" in lower_name:
                    return

                full = (
                    f"{self_inner.current_class}.{mname}"
                    if self_inner.current_class else mname
                )
                mid = self._make_method_id(service_id, fp, full)

                sig_parts = [a.arg for a in node.args.args]
                sig = f"def {mname}({', '.join(sig_parts)})"
                docstring = ast.get_docstring(node)

                line_end = max(
                    (getattr(n, "end_lineno", node.lineno) for n in ast.walk(node)),
                    default=node.lineno,
                )
                methods.append({
                    "id": mid,
                    "name": mname,
                    "full_name": full,
                    "class_name": self_inner.current_class,
                    "file_path": str(fp),
                    "line_number": node.lineno,
                    "line_end": line_end,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "signature": sig,
                    "docstring": docstring,
                    "service_id": service_id,
                })

                old_mid = self_inner.current_method_id
                self_inner.current_method_id = mid
                self_inner.generic_visit(node)
                self_inner.current_method_id = old_mid

            visit_FunctionDef = _visit_func
            visit_AsyncFunctionDef = _visit_func

            def visit_Call(self_inner, node):
                if not self_inner.current_method_id:
                    self_inner.generic_visit(node)
                    return
                callee: Optional[str] = None
                if isinstance(node.func, ast.Name):
                    callee = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    callee = node.func.attr
                if callee:
                    calls.append({
                        "caller_id": self_inner.current_method_id,
                        "callee_name": callee,
                        "line_number": node.lineno,
                    })
                self_inner.generic_visit(node)

        _Visitor().visit(tree)
        return {"methods": methods, "calls": calls}

    # ------------------------------------------------------------------ #
    # Regex fallback (JS/TS/Go/Java when tree-sitter absent)               #
    # ------------------------------------------------------------------ #

    def _analyze_regex_fallback(
        self, fp: Path, content: str, service_id: str, lang: str
    ) -> Dict:
        """Minimal regex extraction used only when tree-sitter grammars are missing."""
        if lang in ("javascript", "typescript", "tsx"):
            return self._regex_js(fp, content, service_id)
        if lang == "go":
            return self._regex_go(fp, content, service_id)
        if lang == "java":
            return self._regex_java(fp, content, service_id)
        if lang == "csharp":
            return self._regex_csharp(fp, content, service_id)
        if lang == "rust":
            return self._regex_rust(fp, content, service_id)
        if lang == "ruby":
            return self._regex_ruby(fp, content, service_id)
        if lang == "php":
            return self._regex_php(fp, content, service_id)
        return {"methods": [], "calls": []}

    # --- JS/TS regex (used only as last-resort fallback) ---

    def _regex_js(self, fp: Path, content: str, service_id: str) -> Dict:
        methods: List[Dict] = []
        lines = content.splitlines()
        current_class: Optional[str] = None
        KEYWORDS = {
            "if", "else", "for", "while", "switch", "catch", "try", "return",
            "new", "typeof", "instanceof", "import", "export", "from", "class",
            "extends", "implements", "interface", "type", "enum", "declare",
            "public", "private", "protected", "static", "async", "await",
        }
        PATTERNS: List[Tuple[re.Pattern, str]] = [
            (re.compile(r"^\s*export\s+default\s+(?:async\s+)?function\s*([\w$]*)\s*[<(]"), "default_fn"),
            (re.compile(r"^\s*export\s+(?:async\s+)?function\s+([\w$]+)\s*[<(]"), "exported_fn"),
            (re.compile(r"^\s*(?:export\s+)?async\s+function\s+([\w$]+)\s*[<(]"), "async_fn"),
            (re.compile(r"^\s*(?:export\s+)?function\s+([\w$]+)\s*[<(]"), "fn"),
            (re.compile(r"^\s*export\s+(?:const|let|var)\s+([\w$]+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[\w$]+)\s*(?::[^=]+)?=>"), "exported_arrow"),
            (re.compile(r"^\s*(?:const|let|var)\s+([\w$]+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[\w$]+)\s*(?::[^=>]+)?=>"), "arrow"),
            (re.compile(r"^\s*(?:(?:public|private|protected|static|abstract|override|async|readonly)\s+)*"
                        r"([\w$]+)\s*[<(][^)]*\)\s*(?::[^{]+)?\s*\{"), "class_method"),
        ]
        class_pat = re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+([\w$]+)")
        for lnum, line in enumerate(lines, 1):
            cm = class_pat.match(line)
            if cm:
                current_class = cm.group(1)
            for pat, kind in PATTERNS:
                m = pat.match(line)
                if not m:
                    continue
                raw = m.group(1) if m.lastindex and m.group(1) else None
                if raw is None:
                    raw = fp.stem if kind == "default_fn" else None
                if not raw or raw in KEYWORDS:
                    continue
                # Skip test and mock methods
                lower_name = raw.lower()
                if lower_name.startswith("test_") or "mock" in lower_name:
                    continue

                full = f"{current_class}.{raw}" if (current_class and kind == "class_method") else raw
                mid = self._make_method_id(service_id, fp, full)
                methods.append({
                    "id": mid, "name": raw, "full_name": full,
                    "class_name": current_class if kind == "class_method" else None,
                    "file_path": str(fp), "line_number": lnum, "line_end": lnum,
                    "signature": raw, "docstring": None, "service_id": service_id,
                })
                break
        return {"methods": methods, "calls": []}

    def _regex_go(self, fp: Path, content: str, service_id: str) -> Dict:
        methods: List[Dict] = []
        pat = re.compile(r"^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(")
        for lnum, line in enumerate(content.splitlines(), 1):
            m = pat.match(line)
            if m:
                mname = m.group(1)
                # Skip test and mock methods
                lower_name = mname.lower()
                if lower_name.startswith("test_") or "mock" in lower_name:
                    continue

                mid = self._make_method_id(service_id, fp, mname)
                methods.append({
                    "id": mid, "name": mname, "full_name": mname,
                    "class_name": None, "file_path": str(fp),
                    "line_number": lnum, "line_end": lnum,
                    "signature": mname, "docstring": None, "service_id": service_id,
                })
        return {"methods": methods, "calls": []}

    def _regex_java(self, fp: Path, content: str, service_id: str) -> Dict:
        methods: List[Dict] = []
        pat = re.compile(
            r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{?"
        )
        skip = {"if", "for", "while", "switch", "catch", "class"}
        for lnum, line in enumerate(content.splitlines(), 1):
            m = pat.search(line)
            if m and m.group(1) not in skip and "class " not in line:
                mname = m.group(1)
                # Skip test and mock methods
                lower_name = mname.lower()
                if lower_name.startswith("test_") or "mock" in lower_name:
                    continue

                mid = self._make_method_id(service_id, fp, mname)
                methods.append({
                    "id": mid, "name": mname, "full_name": mname,
                    "class_name": None, "file_path": str(fp),
                    "line_number": lnum, "line_end": lnum,
                    "signature": mname, "docstring": None, "service_id": service_id,
                })
        return {"methods": methods, "calls": []}

    def _regex_csharp(self, fp: Path, content: str, service_id: str) -> Dict:
        methods: List[Dict] = []
        pat = re.compile(
            r"(?:public|private|protected|internal|static|async|\s)+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{?"
        )
        skip = {"if", "for", "while", "switch", "catch", "class"}
        for lnum, line in enumerate(content.splitlines(), 1):
            m = pat.search(line)
            if m and m.group(1) not in skip and "class " not in line:
                mname = m.group(1)
                lower_name = mname.lower()
                if lower_name.startswith("test") or "mock" in lower_name:
                    continue
                mid = self._make_method_id(service_id, fp, mname)
                methods.append({
                    "id": mid, "name": mname, "full_name": mname,
                    "class_name": None, "file_path": str(fp),
                    "line_number": lnum, "line_end": lnum,
                    "signature": mname, "docstring": None, "service_id": service_id,
                })
        return {"methods": methods, "calls": []}

    def _regex_rust(self, fp: Path, content: str, service_id: str) -> Dict:
        methods: List[Dict] = []
        pat = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(")
        for lnum, line in enumerate(content.splitlines(), 1):
            m = pat.match(line)
            if m:
                mname = m.group(1)
                lower_name = mname.lower()
                if lower_name.startswith("test") or "mock" in lower_name:
                    continue
                mid = self._make_method_id(service_id, fp, mname)
                methods.append({
                    "id": mid, "name": mname, "full_name": mname,
                    "class_name": None, "file_path": str(fp),
                    "line_number": lnum, "line_end": lnum,
                    "signature": mname, "docstring": None, "service_id": service_id,
                })
        return {"methods": methods, "calls": []}

    def _regex_ruby(self, fp: Path, content: str, service_id: str) -> Dict:
        methods: List[Dict] = []
        pat = re.compile(r"^\s*def\s+(?:self\.)?(\w+)")
        for lnum, line in enumerate(content.splitlines(), 1):
            m = pat.match(line)
            if m:
                mname = m.group(1)
                lower_name = mname.lower()
                if lower_name.startswith("test_") or "mock" in lower_name:
                    continue
                mid = self._make_method_id(service_id, fp, mname)
                methods.append({
                    "id": mid, "name": mname, "full_name": mname,
                    "class_name": None, "file_path": str(fp),
                    "line_number": lnum, "line_end": lnum,
                    "signature": mname, "docstring": None, "service_id": service_id,
                })
        return {"methods": methods, "calls": []}

    def _regex_php(self, fp: Path, content: str, service_id: str) -> Dict:
        methods: List[Dict] = []
        pat = re.compile(r"^\s*(?:(?:public|private|protected|static|final)\s+)*function\s+(\w+)\s*\(")
        for lnum, line in enumerate(content.splitlines(), 1):
            m = pat.match(line)
            if m:
                mname = m.group(1)
                lower_name = mname.lower()
                if lower_name.startswith("test") or "mock" in lower_name:
                    continue
                mid = self._make_method_id(service_id, fp, mname)
                methods.append({
                    "id": mid, "name": mname, "full_name": mname,
                    "class_name": None, "file_path": str(fp),
                    "line_number": lnum, "line_end": lnum,
                    "signature": mname, "docstring": None, "service_id": service_id,
                })
        return {"methods": methods, "calls": []}

    # ------------------------------------------------------------------ #
    # Call graph resolution                                                #
    # ------------------------------------------------------------------ #

    def _build_call_graph(
        self, all_methods: Dict[str, Dict], all_calls: List[Dict]
    ) -> List[Tuple[str, str, Dict]]:
        """Match call names to method IDs → (caller, callee, meta) triples."""
        name_to_ids: Dict[str, Set[str]] = defaultdict(set)
        for mid, info in all_methods.items():
            name_to_ids[info["name"]].add(mid)
            if info.get("full_name") and info["full_name"] != info["name"]:
                name_to_ids[info["full_name"]].add(mid)

        seen: Set[Tuple[str, str]] = set()
        result = []
        skip = {"if", "for", "while", "return", "try", "except", "catch", "with", "else", "elif"}
        for call in all_calls:
            caller_id = call["caller_id"]
            callee_name = call.get("callee_name", "")
            if callee_name in skip:
                continue
            for callee_id in name_to_ids.get(callee_name, set()):
                if caller_id == callee_id:
                    continue
                key = (caller_id, callee_id)
                if key not in seen:
                    seen.add(key)
                    result.append(
                        (caller_id, callee_id, {"line": call.get("line_number")})
                    )
        return result
