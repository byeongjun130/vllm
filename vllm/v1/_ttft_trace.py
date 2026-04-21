# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TTFT pipeline-stage trace hooks for profiling end-to-end request latency.

Each call site emits a one-shot marker for (tag, request_id) to
/tmp/vllm_timing.log. AgentSim's profile_ttft_overhead.py parses these
to build the pre/post-EngineCore overhead LUT consumed by its latency
estimator.

Format (one line per marker):
    [TTFT_TRACE] tag=<tag> t=<time.time() float, 6 decimals> req=<request_id>

Always-on; no env-var gating. Intended for use on a dedicated profiling
server, not in production. The helper is process-local: in vLLM v1 the
API server and EngineCore run in separate processes, so each process
keeps its own first-occurrence guard for the tags it emits.
"""

import threading
import time

_LOG_PATH = "/tmp/vllm_timing.log"

_emitted: set[tuple[str, str]] = set()
_lock = threading.Lock()


def emit(tag: str, request_id: str) -> None:
    """Append one TTFT_TRACE marker line for (tag, request_id), once.

    Per-token tags (enginecore_output, output_processed, generate_yield,
    sse_yield) fire on every chunk; the guard collapses them to the
    first occurrence per request.
    """
    key = (tag, request_id)
    with _lock:
        if key in _emitted:
            return
        _emitted.add(key)
    line = f"[TTFT_TRACE] tag={tag} t={time.time():.6f} req={request_id}\n"
    try:
        with open(_LOG_PATH, "a") as f:
            f.write(line)
            f.flush()
    except OSError:
        pass


def release(request_id: str) -> None:
    """Drop guard state for a finished request to keep the set bounded."""
    with _lock:
        for key in [k for k in _emitted if k[1] == request_id]:
            _emitted.discard(key)
