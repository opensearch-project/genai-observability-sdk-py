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

from agent import AGENT_NAME, run_agent

from opensearch_genai_observability_sdk_py import register

provider = register(
    service_name="travel-assistant-evals-native", agent_name=AGENT_NAME, agent_version="v2"
)

for question in ("What should I do in Sydney this weekend?", "Plan a day in New York."):
    run = run_agent(question, version="v2")
    print(f"trace {run.trace_id}: {run.answer[:80]}...")

provider.force_flush()
