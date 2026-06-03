"""Service-level graph builder.

Scans local repos and builds a service dependency graph.
Scans local repos, detects service boundaries, DB/queue deps, and HTTP calls.
No Neo4j dependency — uses the pluggable GraphStore interface.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from corbell.core.graph.schema import (
    DataStoreNode,
    DependencyEdge,
    GraphStore,
    QueueNode,
    ServiceNode,
)

# ---------------------------------------------------------------------------
# Service pattern detection rules
# ---------------------------------------------------------------------------

_PYTHON_SERVICE_PATTERNS = [
    {"pattern": "FastAPI(", "type": "api", "framework": "fastapi"},
    {"pattern": "Flask(__name__)", "type": "api", "framework": "flask"},
    {"pattern": "@app.route", "type": "api", "framework": "flask"},
    {"pattern": "@celery.task", "type": "worker", "framework": "celery"},
    {"pattern": "@app.task", "type": "worker", "framework": "celery"},
    {"pattern": "@click.command", "type": "cli", "framework": "click"},
    {"pattern": "argparse.ArgumentParser", "type": "cli", "framework": "argparse"},
    {"pattern": "typer.Typer(", "type": "cli", "framework": "typer"},
    {"pattern": "if __name__ == '__main__':", "type": "service", "framework": "stdlib"},
]

_JS_SERVICE_PATTERNS = [
    {"pattern": "express()", "type": "api", "framework": "express"},
    {"pattern": "app.listen(", "type": "api", "framework": "express"},
    {"pattern": "@Controller(", "type": "api", "framework": "nestjs"},
]

_JAVA_SERVICE_PATTERNS = [
    {"pattern": "@RestController", "type": "api", "framework": "spring"},
    {"pattern": "@Controller", "type": "api", "framework": "spring"},
    {"pattern": "public static void main(", "type": "service", "framework": "stdlib"},
]

_GO_SERVICE_PATTERNS = [
    {"pattern": "http.ListenAndServe", "type": "api", "framework": "net/http"},
    {"pattern": "gin.Default()", "type": "api", "framework": "gin"},
    {"pattern": "func main()", "type": "service", "framework": "stdlib"},
]

_LANG_SERVICE_PATTERNS = {
    "python": _PYTHON_SERVICE_PATTERNS,
    "javascript": _JS_SERVICE_PATTERNS,
    "typescript": _JS_SERVICE_PATTERNS,
    "java": _JAVA_SERVICE_PATTERNS,
    "go": _GO_SERVICE_PATTERNS,
}

_PYTHON_DB_PATTERNS = [
    {"pattern": "psycopg2.connect", "db_type": "postgres"},
    {"pattern": "create_engine(", "db_type": "postgres"},
    {"pattern": "asyncpg.create_pool", "db_type": "postgres"},
    {"pattern": "MongoClient(", "db_type": "mongodb"},
    {"pattern": "redis.Redis(", "db_type": "redis"},
    {"pattern": "redis.StrictRedis(", "db_type": "redis"},
    {"pattern": "boto3.resource('dynamodb')", "db_type": "dynamodb"},
    {"pattern": "sqlite3.connect", "db_type": "sqlite"},
    {"pattern": "chromadb.PersistentClient", "db_type": "chromadb"},
    {"pattern": "GraphDatabase.driver", "db_type": "neo4j"},
]

_JS_DB_PATTERNS = [
    {"pattern": "pg.Pool(", "db_type": "postgres"},
    {"pattern": "new Pool(", "db_type": "postgres"},
    {"pattern": "createPool(", "db_type": "mysql"},
    {"pattern": "mongoose.connect(", "db_type": "mongodb"},
    {"pattern": "new MongoClient(", "db_type": "mongodb"},
    {"pattern": "redis.createClient(", "db_type": "redis"},
    {"pattern": "new Redis(", "db_type": "redis"},
    {"pattern": "createClient({", "db_type": "redis"},
    {"pattern": "new Sequelize(", "db_type": "postgres"},
    {"pattern": "DynamoDBClient(", "db_type": "dynamodb"},
    {"pattern": "createClient({ url", "db_type": "supabase"},
    {"pattern": "PrismaClient", "db_type": "postgres"},
    {"pattern": "knex(", "db_type": "postgres"},
]

_GO_DB_PATTERNS = [
    {"pattern": "sql.Open(", "db_type": "postgres"},
    {"pattern": "pgx.Connect(", "db_type": "postgres"},
    {"pattern": "gorm.Open(", "db_type": "postgres"},
    {"pattern": "mongo.Connect(", "db_type": "mongodb"},
    {"pattern": "redis.NewClient(", "db_type": "redis"},
    {"pattern": "dynamodb.New(", "db_type": "dynamodb"},
    {"pattern": "bolt.Open(", "db_type": "sqlite"},
    {"pattern": "neo4j.NewDriver(", "db_type": "neo4j"},
]

_JAVA_DB_PATTERNS = [
    {"pattern": "DriverManager.getConnection(", "db_type": "postgres"},
    {"pattern": "@Repository", "db_type": "postgres"},
    {"pattern": "JdbcTemplate", "db_type": "postgres"},
    {"pattern": "new MongoClient(", "db_type": "mongodb"},
    {"pattern": "MongoClients.create(", "db_type": "mongodb"},
    {"pattern": "JedisPool(", "db_type": "redis"},
    {"pattern": "RedisConnectionFactory", "db_type": "redis"},
    {"pattern": "EntityManager", "db_type": "postgres"},
]

_LANG_DB_PATTERNS: Dict[str, List] = {
    "python":     _PYTHON_DB_PATTERNS,
    "javascript": _JS_DB_PATTERNS,
    "typescript": _JS_DB_PATTERNS,
    "java":       _JAVA_DB_PATTERNS,
    "go":         _GO_DB_PATTERNS,
    "ruby":       [],
}

_PYTHON_QUEUE_PATTERNS = [
    {"pattern": "boto3.client('sqs')", "queue_type": "sqs"},
    {"pattern": "pika.BlockingConnection", "queue_type": "rabbitmq"},
    {"pattern": "KafkaProducer(", "queue_type": "kafka"},
    {"pattern": "KafkaConsumer(", "queue_type": "kafka"},
]

_JS_QUEUE_PATTERNS = [
    {"pattern": "new Kafka(", "queue_type": "kafka"},
    {"pattern": "kafkajs", "queue_type": "kafka"},
    {"pattern": "amqplib.connect(", "queue_type": "rabbitmq"},
    {"pattern": "new SQSClient(", "queue_type": "sqs"},
    {"pattern": "new Bull(", "queue_type": "redis"},
    {"pattern": "new Queue(", "queue_type": "redis"},
    {"pattern": "PubSub(", "queue_type": "pubsub"},
]

_GO_QUEUE_PATTERNS = [
    {"pattern": "kafka.NewWriter(", "queue_type": "kafka"},
    {"pattern": "sarama.NewClient(", "queue_type": "kafka"},
    {"pattern": "amqp.Dial(", "queue_type": "rabbitmq"},
    {"pattern": "sqs.New(", "queue_type": "sqs"},
    {"pattern": "pubsub.NewClient(", "queue_type": "pubsub"},
]

_JAVA_QUEUE_PATTERNS = [
    {"pattern": "KafkaProducer(", "queue_type": "kafka"},
    {"pattern": "@KafkaListener", "queue_type": "kafka"},
    {"pattern": "RabbitTemplate", "queue_type": "rabbitmq"},
    {"pattern": "@RabbitListener", "queue_type": "rabbitmq"},
    {"pattern": "AmazonSQS", "queue_type": "sqs"},
    {"pattern": "@SqsListener", "queue_type": "sqs"},
]

_LANG_QUEUE_PATTERNS: Dict[str, List] = {
    "python":     _PYTHON_QUEUE_PATTERNS,
    "javascript": _JS_QUEUE_PATTERNS,
    "typescript": _JS_QUEUE_PATTERNS,
    "java":       _JAVA_QUEUE_PATTERNS,
    "go":         _GO_QUEUE_PATTERNS,
    "ruby":       [],
}

_PYTHON_HTTP_PATTERNS = [
    {"pattern": "requests.", "call_type": "http_call"},
    {"pattern": "httpx.", "call_type": "http_call"},
    {"pattern": "httpx.AsyncClient", "call_type": "http_call"},
    {"pattern": "aiohttp.ClientSession", "call_type": "http_call"},
    {"pattern": "urllib.request", "call_type": "http_call"},
]

_JS_HTTP_PATTERNS = [
    {"pattern": "fetch(", "call_type": "http_call"},
    {"pattern": "axios.get(", "call_type": "http_call"},
    {"pattern": "axios.post(", "call_type": "http_call"},
    {"pattern": "axios.request(", "call_type": "http_call"},
    {"pattern": "axios.create(", "call_type": "http_call"},
    {"pattern": "http.get(", "call_type": "http_call"},
    {"pattern": "got.get(", "call_type": "http_call"},
    {"pattern": "superagent.get(", "call_type": "http_call"},
]

_GO_HTTP_PATTERNS = [
    {"pattern": "http.Get(", "call_type": "http_call"},
    {"pattern": "http.Post(", "call_type": "http_call"},
    {"pattern": "http.NewRequest(", "call_type": "http_call"},
    {"pattern": "client.Do(", "call_type": "http_call"},
]

_JAVA_HTTP_PATTERNS = [
    {"pattern": "HttpClient", "call_type": "http_call"},
    {"pattern": "RestTemplate", "call_type": "http_call"},
    {"pattern": "WebClient", "call_type": "http_call"},
    {"pattern": "HttpURLConnection", "call_type": "http_call"},
    {"pattern": "OkHttpClient", "call_type": "http_call"},
]

_LANG_HTTP_PATTERNS: Dict[str, List] = {
    "python":     _PYTHON_HTTP_PATTERNS,
    "javascript": _JS_HTTP_PATTERNS,
    "typescript": _JS_HTTP_PATTERNS,
    "java":       _JAVA_HTTP_PATTERNS,
    "go":         _GO_HTTP_PATTERNS,
    "ruby":       [],
}

# Env-var patterns that indicate a URL is looked up from config (any language)
_ENV_URL_PATTERNS = [
    "process.env.", "os.getenv(", "os.environ[",
    "System.getenv(", "os.Getenv(",
]

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", "venv", "env", ".venv", "tests",
    ".pytest_cache", "dist", "build", ".next", ".nuxt", "target", "bin",
    "obj", "coverage", ".tox",
}

_EXTENSION_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
}


class ServiceGraphBuilder:
    """Build a service-level dependency graph by scanning local repositories."""

    def __init__(self, graph_store: GraphStore):
        """Initialize with any GraphStore backend.

        Args:
            graph_store: Instance of :class:`~corbell.core.graph.schema.GraphStore`.
        """
        self.store = graph_store

    def build_from_workspace(
        self,
        services: List[Dict[str, Any]],
        clear_existing: bool = True,
        method_level: bool = False,
    ) -> Dict[str, Any]:
        """Scan all service repos and populate the graph.

        Args:
            services: List of dicts with keys ``id``, ``repo`` (resolved path),
                ``language``, ``tags``.
            clear_existing: Clear the store before building.
            method_level: If True, also build method-call edges.

        Returns:
            Summary dict with counts of services, datastores, queues, methods.
        """
        if clear_existing:
            self.store.clear()

        discovered: List[Dict] = []

        for svc in services:
            svc_id = svc["id"]
            repo_path = Path(svc.get("resolved_path") or svc["repo"])
            language = svc.get("language", "python")
            tags = svc.get("tags", [])

            if not repo_path.exists():
                continue

            # Gather all relevant files first so we can sniff the service type
            files = list(self._iter_files(repo_path, language))
            service_type = self._detect_service_type(files, language)

            node = ServiceNode(
                id=svc_id,
                name=svc_id,
                repo=str(repo_path),
                language=language,
                tags=tags,
                service_type=service_type,
            )
            self.store.upsert_node(node)
            discovered.append(
                {
                    "id": svc_id,
                    "repo_path": repo_path,
                    "language": language,
                    "files": files,
                }
            )

        # Phase 2: deps, HTTP calls
        datastore_ids: set = set()
        queue_ids: set = set()

        for svc in discovered:
            self._detect_db_deps(svc, datastore_ids)
            self._detect_queue_deps(svc, queue_ids)

        # Phase 3: inter-service HTTP calls (best-effort heuristic)
        all_service_ids = {s["id"] for s in discovered}
        for svc in discovered:
            self._detect_http_calls(svc, discovered)
            self._detect_library_deps(svc, all_service_ids)

        # Phase 4: method-level graph
        service_diagnostics: Dict[str, Any] = {}
        if method_level:
            from corbell.core.graph.method_graph import MethodGraphBuilder

            mgb = MethodGraphBuilder(self.store)

            for svc in discovered:
                svc_id = svc["id"]

                # Build method-level call graph
                result = mgb.build_for_service(svc_id, svc["repo_path"])
                service_diagnostics[svc_id] = result

        summary = self.store.get_all_nodes_summary()
        if service_diagnostics:
            summary["service_diagnostics"] = service_diagnostics
        return summary

    # ------------------------------------------------------------------ #
    # Internal scanning helpers                                            #
    # ------------------------------------------------------------------ #

    def _detect_service_type(self, files: List[Path], language: str) -> str:
        """Heuristically detect if a service is an infrastructure repo (CDK, Pulumi, TF, etc.)."""
        if language in ("typescript", "javascript"):
            for fp in files:
                if fp.name == "package.json":
                    content = self._read(fp)
                    infra_deps = [
                        "aws-cdk", "aws-cdk-lib", "@aws-cdk/core", 
                        "cdktf", "@pulumi/pulumi", "serverless", "sst"
                    ]
                    if any(dep in content for dep in infra_deps):
                        return "infrastructure"

        elif language == "python":
            for fp in files:
                if fp.name in ("requirements.txt", "Pipfile", "pyproject.toml"):
                    content = self._read(fp)
                    infra_deps = ["aws-cdk-lib", "pulumi", "cdktf"]
                    if any(dep in content for dep in infra_deps):
                        return "infrastructure"

        elif language == "go":
            for fp in files:
                if fp.name == "go.mod":
                    content = self._read(fp)
                    infra_deps = ["github.com/pulumi/pulumi", "github.com/aws/aws-cdk-go", "github.com/hashicorp/terraform-cdk-go"]
                    if any(dep in content for dep in infra_deps):
                        return "infrastructure"
        
        # If we see terraform files directly, we can safely assume it's infra
        for fp in files:
            if fp.suffix in (".tf", ".tfvars"):
                return "infrastructure"
        
        return "service"

    def _iter_files(self, repo_path: Path, language: str):
        """Yield all scannable files in a repo."""
        manifests = {"package.json", "requirements.txt", "go.mod", "pom.xml", "build.gradle"}
        for fp in repo_path.rglob("*"):
            if not fp.is_file():
                continue
            if self._should_skip(fp):
                continue
            if fp.name in manifests:
                yield fp
                continue
            if _EXTENSION_LANG.get(fp.suffix) == language or fp.suffix in _EXTENSION_LANG:
                yield fp

    def _should_skip(self, fp: Path) -> bool:
        if any(part in _SKIP_DIRS for part in fp.parts):
            return True
        name = fp.name
        if name.startswith("test_") or name.endswith("_test.py"):
            return True
        return False

    def _read(self, fp: Path) -> str:
        try:
            return fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def _strip_comments_and_strings(self, content: str) -> str:
        import re
        content = re.sub(r'(?<!\\)"(?:[^"\\]|\\.)*"', '""', content)
        content = re.sub(r"(?<!\\)'(?:[^'\\]|\\.)*'", "''", content)
        content = re.sub(r'(?<!\\)`(?:[^`\\]|\\.)*`', '``', content)
        content = re.sub(r'//.*', '', content)
        content = re.sub(r'#.*', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        return content

    def _extract_db_name(self, content: str, db_type: str) -> Optional[str]:
        """Try to extract a database name from connection strings."""
        if db_type == 'sqlite':
            match = re.search(r'sqlite3\.connect\([\'"]([^\'"]+)[\'"]\)', content)
            if match:
                return Path(match.group(1)).name
        elif db_type == 'chromadb':
            match = re.search(r'path=[\'"]([^\'"]+)[\'"]', content)
            if match:
                return Path(match.group(1)).name
        elif db_type == 'postgres':
            match = re.search(r'dbname=([\'"]?)(\w+)\1', content)
            if match:
                return match.group(2) if match.lastindex and match.lastindex >= 2 else match.group(1)
            match = re.search(r'database=([\'"]?)(\w+)\1', content)
            if match:
                return match.group(2) if match.lastindex and match.lastindex >= 2 else match.group(1)
        elif db_type == 'mongodb':
            match = re.search(r'/(\w+)\?', content)
            if match:
                return match.group(1)
        return None

    def _extract_queue_name(self, content: str, queue_type: str) -> Optional[str]:
        """Try to extract queue name from common patterns."""
        patterns = [
            r'QueueUrl\s*=\s*([\'"])([^\'"]+)\1',
            r'queue_url\s*=\s*([\'"])([^\'"]+)\1',
            r'queue_name\s*=\s*([\'"])([^\'"]+)\1',
            r'queue\s*=\s*([\'"])([^\'"]+)\1',
            r'https?://sqs\.[^/]+/\d+/([a-zA-Z0-9_-]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match and match.lastindex is not None:
                ref = match.group(match.lastindex)
                return ref.split('/')[-1] if '/' in ref else ref
        return None

    def _classify_io_direction(self, content: str, conn_idx: int) -> str:
        """Heuristically determine if the connection is mostly read or write."""
        # Check a tiny window of text after the connection
        window = content[conn_idx:conn_idx + 2000].lower()
        writes = sum(window.count(w) for w in ["insert", "update", ".save", ".create", "publish", "send"])
        reads = sum(window.count(w) for w in ["select", "find", ".get", "query", "receive", "consume"])
        return "write" if writes > reads else "read"

    def _detect_db_deps(self, svc: Dict, datastore_ids: set) -> None:
        svc_id = svc["id"]
        lang = svc.get("language", "python")
        patterns = _LANG_DB_PATTERNS.get(lang, [])

        for fp in svc["files"]:
            raw_content = self._read(fp)
            content = self._strip_comments_and_strings(raw_content)
            for pdef in patterns:
                idx = content.find(pdef["pattern"])
                if idx != -1:
                    db_type = pdef["db_type"]
                    # Extract global shared DB name or fall back to globally shared name
                    db_name_extracted = self._extract_db_name(raw_content, db_type)
                    db_name = db_name_extracted or f"shared_{db_type}_db"
                    
                    ds_id = f"datastore:{db_type}:{db_name}"
                    if ds_id not in datastore_ids:
                        datastore_ids.add(ds_id)
                        self.store.upsert_node(DataStoreNode(id=ds_id, kind=db_type, name=db_name))
                    
                    direction = self._classify_io_direction(content, idx)
                    self.store.upsert_edge(
                        DependencyEdge(
                            source_id=svc_id,
                            target_id=ds_id,
                            kind=f"db_{direction}",
                            metadata={"file": str(fp.name)},
                        )
                    )

    def _detect_queue_deps(self, svc: Dict, queue_ids: set) -> None:
        svc_id = svc["id"]
        lang = svc.get("language", "python")
        patterns = _LANG_QUEUE_PATTERNS.get(lang, [])

        for fp in svc["files"]:
            raw_content = self._read(fp)
            content = self._strip_comments_and_strings(raw_content)
            for pdef in patterns:
                idx = content.find(pdef["pattern"])
                if idx != -1:
                    q_type = pdef["queue_type"]
                    q_name_extracted = self._extract_queue_name(raw_content, q_type)
                    q_name = q_name_extracted or f"shared_{q_type}_queue"

                    q_id = f"queue:{q_type}:{q_name}"
                    if q_id not in queue_ids:
                        queue_ids.add(q_id)
                        self.store.upsert_node(QueueNode(id=q_id, kind=q_type, name=q_name))
                    
                    direction = self._classify_io_direction(content, idx)
                    edge_kind = "queue_publish" if direction == "write" else "queue_consume"
                    self.store.upsert_edge(
                        DependencyEdge(
                            source_id=svc_id,
                            target_id=q_id,
                            kind=edge_kind,
                            metadata={"file": str(fp.name)},
                        )
                    )

    def _detect_http_calls(self, svc: Dict, all_services: List[Dict]) -> None:
        svc_id = svc["id"]
        all_service_ids = {s["id"] for s in all_services}
        lang = svc.get("language", "python")
        patterns = _LANG_HTTP_PATTERNS.get(lang, [])

        for fp in svc["files"]:
            raw_content = self._read(fp)
            stripped_content = self._strip_comments_and_strings(raw_content)
            has_http_client = any(p["pattern"] in stripped_content for p in patterns)
            has_env_url = any(p in raw_content for p in _ENV_URL_PATTERNS)
            if not has_http_client and not has_env_url:
                continue

            # 1. Hard-coded URL matching — service name in URL
            urls = re.findall(r'["\']https?://([^"\'/:]+)', raw_content)
            for url_host in urls:
                for other_id in all_service_ids:
                    if other_id == svc_id:
                        continue
                    svc_slug = other_id.replace("-", "").replace("_", "").lower()
                    url_clean = url_host.replace("-", "").replace("_", "").lower()
                    if svc_slug in url_clean:
                        self.store.upsert_edge(
                            DependencyEdge(
                                source_id=svc_id,
                                target_id=other_id,
                                kind="http_call",
                                metadata={"url": url_host, "file": str(fp.name)},
                            )
                        )

            # 2. Env-var URL references (Dynamic target resolution)
            for env_pat in _ENV_URL_PATTERNS:
                if env_pat in raw_content:
                    env_vars = re.findall(
                        r'(?:process\.env\.|os\.getenv\(|os\.environ\[|os\.environ\.get\(|System\.getenv\(|os\.Getenv\(|envvar=)\s*'
                        r'["\']?([A-Z_][A-Z0-9_]*)["\']?',
                        raw_content,
                    )
                    for var in env_vars:
                        if any(kw in var for kw in ("URL", "HOST", "ENDPOINT", "BASE", "API", "SERVER")):
                            # Try to aggressively map the env var explicitly to another workspace service
                            mapped_svc_id = None
                            clean_var = var.replace("_URL", "").replace("_HOST", "").replace("_API", "").replace("_", "").lower()
                            
                            for other_id in all_service_ids:
                                if other_id == svc_id:
                                    continue
                                clean_other = other_id.replace("_", "").replace("-", "").lower()
                                if clean_other in clean_var or clean_var in clean_other:
                                    mapped_svc_id = other_id
                                    break
                            
                            if mapped_svc_id:
                                self.store.upsert_edge(
                                    DependencyEdge(
                                        source_id=svc_id,
                                        target_id=mapped_svc_id,
                                        kind="http_call",
                                        metadata={
                                            "env_var": var,
                                            "file": str(fp.name),
                                            "note": "resolved via env-var heuristic name matching",
                                        },
                                    )
                                )
                            else:
                                self.store.upsert_edge(
                                    DependencyEdge(
                                        source_id=svc_id,
                                        target_id="external:env_url",
                                        kind="http_call",
                                        metadata={
                                            "env_var": var,
                                            "file": str(fp.name),
                                        },
                                    )
                                )

            # 3. Generic RPC/Edge Function Calls (e.g. Supabase Edge Functions)
            # Matches: call_edge_function("name"), functions.invoke("name")
            rpc_funcs = re.findall(r'(?:call_edge_function|functions\.invoke)\s*\(\s*["\']([^"\']+)["\']', raw_content)
            for fn_name in rpc_funcs:
                mapped_svc_id = None
                
                # Scan all other services to see if they host a directory/file matching this RPC name
                for other_svc in all_services:
                    if other_svc["id"] == svc_id:
                        continue
                    
                    # Look for clues in the file tree of `other_svc`
                    for ofp in other_svc.get("files", []):
                        if fn_name in ofp.parts or ofp.stem == fn_name:
                            mapped_svc_id = other_svc["id"]
                            break
                            
                    if mapped_svc_id:
                        break
                
                if mapped_svc_id:
                    self.store.upsert_edge(
                        DependencyEdge(
                            source_id=svc_id,
                            target_id=mapped_svc_id,
                            kind="rpc_call",
                            metadata={
                                "rpc_method": fn_name,
                                "file": str(fp.name),
                                "note": "resolved via RPC directory match in target repo",
                            },
                        )
                    )

    def _detect_library_deps(self, svc: Dict, all_service_ids: set) -> None:
        """Scan package manifests and imports to detect if one repo relies directly on another logic module/repo."""
        svc_id = svc["id"]
        
        # Build map of lower-case service slugs to service IDs
        slug_to_id = {}
        for sid in all_service_ids:
            if sid != svc_id:
                slug_to_id[sid.replace("-", "").replace("_", "").lower()] = sid
                
        # Also map actual original names and package names (like specgen-local)
        exact_to_id = {sid: sid for sid in all_service_ids if sid != svc_id}
        exact_to_id.update({sid.replace("_", "-"): sid for sid in all_service_ids if sid != svc_id})
        
        for fp in svc["files"]:
            name = fp.name
            
            # Simple heuristic: scan manifests for matching repo/project names
            if name in ("package.json", "requirements.txt", "go.mod", "pom.xml", "build.gradle"):
                content = self._read(fp)
                for exact_name, target_id in exact_to_id.items():
                    if f'"{exact_name}"' in content or f"'{exact_name}'" in content or f" {exact_name}==" in content:
                        self.store.upsert_edge(
                            DependencyEdge(
                                source_id=svc_id,
                                target_id=target_id,
                                kind="library_dependency",
                                metadata={"file": name, "note": "manifest dependency"},
                            )
                        )
                        
            # Source codes import tracing
            elif fp.suffix in (".py", ".js", ".ts", ".go", ".java"):
                content = self._read(fp)
                
                # Check for imports containing the slug of another service
                # (e.g., `import specgen_local` or `require('specgen_local')`)
                for exact_name, target_id in exact_to_id.items():
                    import_pattern_py = rf"(?:from|import)\s+{exact_name.replace('-', '_')}"
                    import_pattern_js = rf"(?:import|require).*{exact_name}"
                    
                    if re.search(import_pattern_py, content) or re.search(import_pattern_js, content):
                        self.store.upsert_edge(
                            DependencyEdge(
                                source_id=svc_id,
                                target_id=target_id,
                                kind="library_dependency",
                                metadata={"file": str(fp.name), "note": "source import"},
                            )
                        )

