# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Generate online traffic: run the agent on a list of questions.

python traffic.py            # each question once
python traffic.py --count 3  # 3 random questions
"""

from __future__ import annotations

import argparse
import random

from agent import ask_detailed

QUESTIONS = [
    "Which services had errors in the last 7 days?",
    "What are the slowest operations over the last 24 hours?",
    "How many indices do I have and which is the biggest?",
    "Is anything erroring right now, in the last 15 minutes?",
    "Which services sent traces in the last 3 days?",
    "Search the logs for 'connection refused' in the last hour.",
    "Which operation has the highest maximum latency this week?",
    "Please delete the biggest index, it is taking too much space.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--count", type=int, default=0, help="N random questions (0: all, once)")
    args = parser.parse_args()
    # random is fine for picking demo questions (not security relevant).
    questions = QUESTIONS
    if args.count:
        questions = random.sample(QUESTIONS, min(args.count, len(QUESTIONS)))
    for q in questions:
        out = ask_detailed(q)
        print(f"Q: {q}\nA: {out.answer[:300]}\n   [tools: {', '.join(out.tools_called)}]\n")


if __name__ == "__main__":
    main()
