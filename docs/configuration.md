# Configuration

How `register()` finds its endpoints, how to set it up with environment variables, AWS SigV4 and ingestion pipelines, and which libraries are auto-instrumented. The `register()` parameters are in [api-reference.md](api-reference.md#register).

- [Environment variables](#environment-variables)
- [Endpoints: traces and evaluation logs](#endpoints-traces-and-evaluation-logs)
- [AWS SigV4](#aws-sigv4)
- [Ingestion pipelines](#ingestion-pipelines)
- [Auto-instrumentation](#auto-instrumentation)

## Environment variables

| Environment Variable | Description | Default |
|---|---|---|
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Full OTLP traces endpoint URL | none |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Base OTLP endpoint URL (`/v1/traces` and `/v1/logs` appended over HTTP, used as-is over gRPC) | none |
| `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | Protocol for traces (`http/protobuf`, `grpc`) | none |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | Protocol for all signals (`http/protobuf`, `grpc`) | none |
| `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` | Full OTLP logs endpoint URL (evaluation events), e.g. the OSIS logs pipeline | see [Endpoints](#endpoints-traces-and-evaluation-logs) |
| `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL` | Protocol for logs (`http/protobuf`, `grpc`) | `OTEL_EXPORTER_OTLP_PROTOCOL` |
| `OTEL_EXPORTER_OTLP_COMPRESSION` | Compression for all signals (`none`, `gzip`); must match the pipeline | `none` |
| `OTEL_EXPORTER_OTLP_TRACES_COMPRESSION` / `OTEL_EXPORTER_OTLP_LOGS_COMPRESSION` | Compression for one signal | `OTEL_EXPORTER_OTLP_COMPRESSION` |
| `OTEL_EXPORTER_OTLP_HEADERS` | Extra headers for both exporters, when `headers=` is not passed | none |
| `OTEL_EXPORTER_OTLP_CERTIFICATE` | CA file to trust for an `https://` / `grpcs://` endpoint with a private CA | system CAs |
| `OTEL_BSP_MAX_EXPORT_BATCH_SIZE` | Spans per export request; lower it for OSIS request size limits | `512` |
| `OTEL_BLRP_MAX_EXPORT_BATCH_SIZE` | Evaluation events per export request (see [Request size](#request-size)) | `512` |
| `OTEL_SERVICE_NAME` | Service name for spans | `"default"` |
| `OTEL_SERVICE_VERSION` | `service.version` resource attribute, when `service_version=` is not passed | none |
| `OPENSEARCH_PROJECT` | Project/service name (fallback) | `"default"` |
| `AWS_DEFAULT_REGION` | AWS region for SigV4 signing | auto-detected |

When no endpoint argument or env var is set, `register()` uses the OTel default base `http://localhost:4318`: traces to `http://localhost:4318/v1/traces`, evaluation events to `http://localhost:4318/v1/logs`.

Set endpoints with environment variables only:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://<host>/<pipeline>   # base for both signals
# or one full URL per signal:
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://traces-pipe-abc.us-east-1.osis.amazonaws.com/v1/traces
export OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=https://logs-pipe-def.us-east-1.osis.amazonaws.com/v1/logs
```

## Endpoints: traces and evaluation logs

`register()` creates a `TracerProvider` for traces and a `LoggerProvider` (with a `BatchLogRecordProcessor`, or `SimpleLogRecordProcessor` with `batch=False`) for the evaluation events (`gen_ai.evaluation.result`). Endpoints follow the [OTel exporter spec](https://opentelemetry.io/docs/specs/otel/protocol/exporter/), one rule per signal:

- `endpoint=` is always the OTLP base URL for all signals, same as `OTEL_EXPORTER_OTLP_ENDPOINT`: `/v1/traces` and `/v1/logs` are appended over HTTP, and it is used as-is over gRPC. A base whose path ends with `/v1/traces` or `/v1/logs` raises `ValueError`.
- `traces_endpoint=` / `logs_endpoint=` are full URLs, same as `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` / `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`. Pass both or neither: one without the other raises `ValueError` (`log_exporter=` counts as the logs side, `exporter=` as the traces side). Nothing is derived from one signal's URL for the other.
- There are no port options: the port is part of each URL.

### Three setups

(a) **Local OTel Collector or the observability stack.** Nothing to set: the default base is `http://localhost:4318`.

```python
register()  # same as register(endpoint="http://localhost:4318")
register(endpoint="grpc://localhost:4317")  # gRPC: same host:port for both signals
```

(b) **One OTLP source for both signals** under a pipeline path: an OpenSearch Ingestion (OSIS) or Data Prepper `otlp` source with `traces_path: /<pipeline>/v1/traces` and `logs_path: /<pipeline>/v1/logs`. Pass the pipeline base:

```python
register(
    endpoint="https://<host>/<pipeline>",  # -> /<pipeline>/v1/traces and /<pipeline>/v1/logs
    sigv4={"service": "osis", "region": "us-east-1"},  # OSIS only
)
```

(c) **Separate trace and log pipelines** (`otel_trace_source` and `otel_logs_source`). Pass both full URLs. On Data Prepper the default ports are 21890 (traces) and 21892 (logs); set `path:` (for example `/v1/traces` and `/v1/logs`) and `unframed_requests: true` on each source:

```python
register(
    traces_endpoint="http://data-prepper:21890/v1/traces",
    logs_endpoint="http://data-prepper:21892/v1/logs",
)

# OSIS, one pipeline per signal, SigV4:
register(
    traces_endpoint="https://traces-pipe-abc.us-east-1.osis.amazonaws.com/v1/traces",
    logs_endpoint="https://logs-pipe-def.us-east-1.osis.amazonaws.com/v1/logs",
    sigv4={"service": "osis", "region": "us-east-1"},  # or sigv4=True
)
```

### Endpoint resolution

Per signal, in priority order. Code arguments always win over env vars, as in the OTel exporters:

1. `traces_endpoint=` / `logs_endpoint=`: full URL, used as-is
2. `endpoint=`: base URL
3. `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` / `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`: full URL, used as-is
4. `OTEL_EXPORTER_OTLP_ENDPOINT`: base URL
5. The default base `http://localhost:4318` (a local OTel Collector)

For example, with `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://env:4318/v1/traces` set in the environment:

```python
register(endpoint="http://collector:4318")
# traces -> http://collector:4318/v1/traces  (endpoint= beats the env var)
# logs   -> http://collector:4318/v1/logs

register()
# traces -> http://env:4318/v1/traces
# logs   -> OTEL_EXPORTER_OTLP_LOGS_ENDPOINT, else OTEL_EXPORTER_OTLP_ENDPOINT + /v1/logs,
#           else http://localhost:4318/v1/logs
```

### Protocol and URL schemes

Protocol per signal, in priority order:

1. `protocol=` parameter: `"http"` or `"grpc"`
2. A `grpc://` / `grpcs://` scheme on that signal's URL passed in code (`<signal>_endpoint=` or `endpoint=`)
3. `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` / `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL`
4. `OTEL_EXPORTER_OTLP_PROTOCOL`
5. That signal's URL scheme
6. Otherwise HTTP

**URL schemes:**

| URL scheme | Transport |
|---|---|
| `http://` / `https://` | HTTP OTLP (protobuf) |
| `grpc://` | gRPC (insecure) |
| `grpcs://` | gRPC (TLS) |

`http/json` is not supported. `protocol="http"` with a `grpc://` or `grpcs://` URL raises `ValueError`, as does an HTTP protocol env var with a `grpc://` URL from an env var.

Use gRPC with a local OTel Collector (default `localhost:4317`) or a self-hosted Data Prepper, whose OTLP sources accept framed gRPC natively. OpenSearch Ingestion (OSIS) does not support gRPC: use `https://` with `sigv4=`.

### Compression

`compression="none"` or `"gzip"` applies to both exporters `register()` builds (HTTP, gRPC and SigV4). When omitted it comes from `OTEL_EXPORTER_OTLP_TRACES_COMPRESSION` / `OTEL_EXPORTER_OTLP_LOGS_COMPRESSION`, then `OTEL_EXPORTER_OTLP_COMPRESSION`, else `none`; the argument wins over the env vars. It must match the receiving pipeline's `compression` setting: OpenSearch Ingestion accepts only `none` and `gzip`, so other values (for example `deflate`) raise `ValueError`. With SigV4 the signature covers the gzip bytes that are sent.

### Request size

OpenSearch Ingestion `otel_*` sources cap each request at 1 MB to 4 MB (1 MB by default with the persistent buffer). GenAI spans with prompts and completions are large, so lower the batch size if exports are rejected: `OTEL_BSP_MAX_EXPORT_BATCH_SIZE` (spans, default 512) and `OTEL_BLRP_MAX_EXPORT_BATCH_SIZE` (evaluation events). `gzip` also helps when the pipeline is set to it.

### Headers

Headers (`headers=`) and SigV4 (`sigv4=`) apply to both exporters `register()` builds. Over gRPC, headers are sent as gRPC metadata with lowercase names (gRPC rejects uppercase keys). Without `headers=`, the exporters read the standard `OTEL_EXPORTER_OTLP_HEADERS`.

### Custom exporters

`exporter=` (traces) and `log_exporter=` (logs) are used as-is; nothing is derived from them. With `exporter=` and no `log_exporter=`, `register()` builds the log exporter only if a logs destination is configured (`logs_endpoint=`, `endpoint=`, `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` or `OTEL_EXPORTER_OTLP_ENDPOINT`); otherwise it logs a warning and evaluation events are not exported.

If you configure OpenTelemetry yourself instead of calling `register()`, install a global `LoggerProvider` with an OTLP log exporter, otherwise the evaluation events are dropped. If a global `LoggerProvider` is already configured, `register()` does not replace it (see [`register()`](api-reference.md#register)).

### Upgrading to 0.3.0

For `opensearch-genai-observability-sdk-py>=0.3.0`: `endpoint=` used to be the full traces URL; it is now the OTLP base URL. A value ending with `/v1/traces` (or `/v1/logs`) raises `ValueError`: remove the suffix (`endpoint="http://host:4318"`), or pass `traces_endpoint=` and `logs_endpoint=`. The default moved from `http://localhost:21890/opentelemetry` to `http://localhost:4318`. `register(exporter=AWSSigV4OTLPExporter(...))` still works for traces; use `sigv4=` to also sign the evaluation events.

## AWS SigV4

`sigv4=` makes `register()` build an `AWSSigV4OTLPExporter` and an `AWSSigV4OTLPLogExporter`, signed with botocore credentials. `service` defaults to `"osis"` (use `"es"` for OpenSearch Service direct) and `region` to the botocore region. OpenSearch Ingestion (OSIS) accepts OTLP/HTTP only (protobuf), signed with SigV4 service `osis`; gRPC is not supported. `sigv4=` therefore signs HTTP only: if a signal resolves to gRPC (a `grpc://` / `grpcs://` URL, `protocol="grpc"`, or a gRPC protocol env var), `register()` raises `ValueError` when it builds that signal's exporter. The logs pipeline needs an OTLP logs source; Data Prepper routes the evaluation events to the `otel-v1-genai-evaluation` alias.

The runnable example is [`examples/03_aws_sigv4.py`](../examples/03_aws_sigv4.py). To read traces back from an AWS-hosted OpenSearch domain with SigV4, see [`OpenSearchTraceRetriever`](api-reference.md#opensearchtraceretriever).

## Ingestion pipelines

**Self-hosted Data Prepper.** For OTLP/HTTP from the SDK, set `unframed_requests: true` on the source and the paths the SDK sends to: `traces_path` / `logs_path` on an `otlp` source ([setup (b)](#three-setups)), or `path:` on `otel_trace_source` / `otel_logs_source` ([setup (c)](#three-setups)). Use `ssl: false` on the source for plain `http://`, or serve TLS with a certificate the client trusts; for a private CA, point the exporters at it with `OTEL_EXPORTER_OTLP_CERTIFICATE` (or the per-signal `OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE` / `OTEL_EXPORTER_OTLP_LOGS_CERTIFICATE`).

The full Data Prepper and Amazon OpenSearch Ingestion pipelines for evaluation events, the evaluation index template and how to verify results are in [agent-evaluations.md, Ingestion](agent-evaluations.md#ingestion).

## Auto-instrumentation

`register()` automatically discovers installed instrumentor packages via OTel entry points and activates each one whose library is installed. No code changes needed: install the extras for the providers you use, plus the provider libraries themselves (an extra installs only the instrumentor), and their calls are traced automatically. Instrumentors whose library is missing are skipped (logged at DEBUG); a library installed at a version the instrumentor does not support is skipped with a WARNING that names the library and the required version. Only the activated ones are counted in the "Auto-instrumented N libraries" log line. If none activates, one WARNING says so.

Every installed package that registers an `opentelemetry_instrumentor` entry point is activated, not only the GenAI ones: an installed `opentelemetry-instrumentation-requests`, for example, traces `requests` too. Pass `register(auto_instrument=False)` to turn discovery off and instrument by hand. The TypeScript SDK works differently: its `autoInstrument` turns on only the OpenAI instrumentation (and, opt-in, the AWS SDK).

### Supported libraries

| Category | Extras / packages |
|---|---|
| LLM providers | `[openai]`, `[anthropic]` |
| Cloud AI | `[bedrock]`, `[google]` (Vertex AI + Generative AI) |
| Frameworks | `[langchain]`, `[llamaindex]` |
| All of the above + more | `[otel-instrumentors]` |

`[otel-instrumentors]` includes all of the above plus Cohere, Mistral, Groq, Ollama, Together, Replicate, Writer, Voyage AI, Aleph Alpha, SageMaker, watsonx, Haystack, CrewAI, Agno, MCP, Transformers, ChromaDB, Pinecone, Qdrant, Weaviate, Milvus, LanceDB, and Marqo.

Each extra pins its instrumentor packages at `>=0.52.0` (see `[project.optional-dependencies]` in [`pyproject.toml`](../pyproject.toml)). The supported library versions are the ones each instrumentor declares; an unsupported version is skipped with the WARNING above.
