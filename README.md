<div align="center">
  <h1>codebase-retrieval-context-engine</h1>
  <p><strong>Code retrieval engine for LLM context via MCP.</strong></p>
  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"/></a>
  </p>
</div>

---

## Add to Claude Code

```bash
claude mcp add codebase-retrieval -e CORBELL_LLM_PROVIDER=google -e GOOGLE_API_KEY=your-google-api-key -e GOOGLE_MODEL=gemini-3.1-flash-lite -e CORBELL_EMBEDDING_MODEL=voyage-4-lite -e VOYAGE_API_KEY=your-voyage-api-key -- uvx codebase-retrieval-context-engine
```

That's it. The AI agent passes workspace path and triggers index builds automatically.

---

## Build index manually (optional)

```bash
uvx codebase-retrieval-context-engine index build
```

Run from your project root. Env vars (`CORBELL_LLM_PROVIDER`, `GOOGLE_API_KEY`, etc.) must be set in your shell.

---

## Environment variables

| Variable | Description |
|---|---|
| `CORBELL_LLM_PROVIDER` | LLM provider for reranking (`google`, `anthropic`, `openai`) |
| `GOOGLE_API_KEY` | Google AI API key (supports multiple: `key1,key2,key3`) |
| `GOOGLE_MODEL` | e.g. `gemini-3.1-flash-lite` |
| `CORBELL_EMBEDDING_MODEL` | `voyage-4-lite`, `voyage-code-3`, or `gemini-embedding-001` |
| `VOYAGE_API_KEY` | Voyage AI API key (supports multiple: `key1,key2,key3`). Add a card to billing to unlock rate limits. |

---

## License

Apache 2.0
