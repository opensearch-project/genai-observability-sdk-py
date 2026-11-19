# Shared travel assistant data

Data used by every framework example in `examples/tracing/` and `examples/evals/`, so
results are comparable across frameworks.

- `travel_data.json`: the system prompt and 6 cities, each with weather and 2 to 3 events.
- `test_suite.jsonl`: 5 test cases, one JSON object per line: `id`, `name`, `input`, `expected`.

## Tool contract

Each example implements the two tools locally (copy-paste friendly, no shared code):

- `get_weather(city)` looks up `cities[city.strip().lower()]` and returns
  `{"city": "Paris", "condition": "sunny", "high_c": 24, "low_c": 15}`.
- `find_events(city)` returns `{"city": "Paris", "events": [{"name": ..., "time": ...}]}`.
- For an unknown city both return `{"error": "no data for <city>"}`.

The data is static, so the only non-deterministic part of a run is the model.
