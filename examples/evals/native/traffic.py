# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Send a few live requests, as production traffic, for online_eval.py to judge.

python traffic.py
"""

from __future__ import annotations

from agent import AGENT_NAME, PROMPTS, run_agent

from opensearch_genai_observability_sdk_py import register

# Strands emits its own chat spans; auto_instrument=False keeps an installed Bedrock
# (botocore) instrumentor from tracing each model call a second time.
provider = register(
    auto_instrument=False,
    service_name="travel-assistant-evals-native-py",
    agent_name=AGENT_NAME,
    agent_version="v2",
)

for question in ("What should I do in Sydney this weekend?", "Plan a day in New York."):
    # The v2 prompt, so results filed under version v2 belong to v2 answers
    run = run_agent(question, system_prompt=PROMPTS["v2"], version="v2")
    print(f"trace {run.trace_id}: {run.answer[:80]}...")

provider.force_flush()
