"""Per-iteration attention/other latency tracing.

Gated entirely by the env var ``VLLM_ITER_TRACING=1``; when unset,
every hook is a no-op at nanosecond cost.

When enabled, records CUDA events around each attention-layer call
and around the whole model forward. Uses
``cudaEventRecordWithFlags(..., cudaEventRecordExternal)`` when the
active stream is in CUDA-graph capture so the event becomes a real
external node that fires per replay rather than a structural graph
marker. Drains ``elapsed_time()`` after an `event.synchronize()` on
the step's `total_end` event and appends one NDJSON record per
iteration to ``VLLM_ITER_TRACING_PATH``
(default ``/tmp/vllm_iter_tracing.ndjson``).

File layout: one NDJSON line per iteration.

    {"iter_idx": 0, "total_us": 10234.5, "attn_us": 3120.8,
     "attn_us_mean": 48.76, "other_us": 7113.7, "num_attn_layers": 64,
     "request_shapes": [[2045, 2045], [1, 1024], [1, 1024]]}
"""

from __future__ import annotations

import ctypes
import json
import os
import threading
from typing import Optional

import torch


_ENV_FLAG = "VLLM_ITER_TRACING"
_ENV_PATH = "VLLM_ITER_TRACING_PATH"
_DEFAULT_PATH = "/tmp/vllm_iter_tracing.ndjson"
_CUDA_EVENT_RECORD_EXTERNAL = 0x1


def is_enabled() -> bool:
    return os.getenv(_ENV_FLAG, "0") == "1"


_CUDART: Optional[ctypes.CDLL] = None


def _load_cudart() -> ctypes.CDLL:
    global _CUDART
    if _CUDART is None:
        lib = ctypes.CDLL("libcudart.so")
        lib.cudaEventRecordWithFlags.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
        ]
        lib.cudaEventRecordWithFlags.restype = ctypes.c_int
        _CUDART = lib
    return _CUDART


def _warmed_event() -> torch.cuda.Event:
    """Create a timing event and lazy-init its underlying cudaEvent_t.

    Must be called OUTSIDE CUDA-graph capture — PyTorch's ``record()``
    allocates the event handle on first call, and allocation during
    capture would invalidate the graph.
    """
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _record(event: torch.cuda.Event, stream: torch.cuda.Stream) -> None:
    """Record ``event`` on ``stream``.

    Inside capture: use ``cudaEventRecordWithFlags(..., External)``
    so the event becomes a real external node whose timestamp is
    usable after replay. Outside capture: use PyTorch's native
    ``record()`` which handles lazy init.
    """
    if torch.cuda.is_current_stream_capturing():
        cudart = _load_cudart()
        err = cudart.cudaEventRecordWithFlags(
            ctypes.c_void_p(event.cuda_event),
            ctypes.c_void_p(stream.cuda_stream),
            _CUDA_EVENT_RECORD_EXTERNAL,
        )
        if err != 0:
            raise RuntimeError(
                f"cudaEventRecordWithFlags(External) failed: err={err}. "
                f"Event must be pre-warmed outside capture."
            )
    else:
        event.record(stream)


class IterationTracer:
    """Per-worker tracer. Accumulates one record per model step.

    Lifecycle per step:
        step_begin()  → records total_start
        attn_begin()  → per-attention-layer pair start
        attn_end()    → per-attention-layer pair end
        step_end(shapes) → records total_end, captures request shapes
        drain()       → blocks on total_end, reads elapsed_times,
                        writes one NDJSON line
    """

    def __init__(
        self,
        output_path: str,
        num_hidden_layers: int,
        max_attn_layers: int,
    ):
        self.output_path = output_path
        self.num_hidden_layers = num_hidden_layers
        self._file = open(output_path, "w", buffering=1)  # line-buffered
        self._lock = threading.Lock()

        self._total_start = _warmed_event()
        self._total_end = _warmed_event()

        # Reusable pool; grown lazily if the model has more than
        # max_attn_layers attention calls per step (rare).
        self._attn_start_pool = [_warmed_event() for _ in range(max_attn_layers)]
        self._attn_end_pool = [_warmed_event() for _ in range(max_attn_layers)]
        torch.cuda.synchronize()  # ensure all warming records complete

        self._step_idx = -1
        self._active = False
        self._attn_count = 0   # number of layers this step
        self._pending_attn_start: Optional[torch.cuda.Event] = None
        self._request_shapes: list[tuple[int, int]] = []
        self._counter = 0

    # -- hooks called from gpu_model_runner --------------------------------

    def step_begin(self, step_idx: int) -> None:
        self._step_idx = step_idx
        self._active = True
        self._attn_count = 0
        self._pending_attn_start = None
        self._request_shapes = []
        stream = torch.cuda.current_stream()
        _record(self._total_start, stream)

    def step_end(self, request_shapes: list[tuple[int, int]]) -> None:
        if not self._active:
            return
        stream = torch.cuda.current_stream()
        _record(self._total_end, stream)
        self._request_shapes = list(request_shapes)

    # -- hooks called from FlashAttentionImpl.forward ----------------------
    #
    # These fire ALWAYS — even during vLLM's warmup/capture path (which
    # runs via `_dummy_run`, not `execute_model`, so `step_begin` is
    # never called then). Recording events inside a captured region is
    # exactly what makes them fire on replay — so we need these hooks
    # active during capture to get any decode-replay attention timing
    # at all.
    #
    # Auto-wrap: each model forward calls attn_begin/end exactly
    # `num_hidden_layers` times. When we'd overflow the pool, we wrap
    # back to slot 0 — that means a new forward pass has started and
    # pool slots are being reused (e.g., different capture graphs
    # share the same pool). Each graph replay fires its own external
    # event nodes, updating slots 0..num_hidden_layers-1 to that
    # replay's timestamps.

    def attn_begin(self) -> None:
        if self._attn_count >= self.num_hidden_layers:
            # New forward started without a step_begin (e.g., warmup
            # capturing the next graph size). Wrap the pool.
            self._attn_count = 0
            self._pending_attn_start = None
        idx = self._attn_count
        if idx >= len(self._attn_start_pool):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"IterationTracer attn pool exhausted during capture "
                    f"(have {len(self._attn_start_pool)}, need >{idx}). "
                    "Increase max_attn_layers at tracer init."
                )
            self._attn_start_pool.append(_warmed_event())
            self._attn_end_pool.append(_warmed_event())
        stream = torch.cuda.current_stream()
        self._pending_attn_start = self._attn_start_pool[idx]
        _record(self._pending_attn_start, stream)

    def attn_end(self) -> None:
        if self._pending_attn_start is None:
            return
        idx = self._attn_count
        stream = torch.cuda.current_stream()
        _record(self._attn_end_pool[idx], stream)
        self._pending_attn_start = None
        self._attn_count += 1

    # -- called after the step's end-of-iteration CPU sync -----------------

    def drain(self) -> None:
        if not self._active:
            return
        # Block CPU until the GPU has recorded total_end. All earlier
        # events are guaranteed complete by then since they were
        # recorded earlier in program order on the same stream.
        self._total_end.synchronize()

        total_us = self._total_start.elapsed_time(self._total_end) * 1000.0

        # Two regimes produce attention events on this step:
        #  1. Eager path (large prefill / mixed batches that exceed the
        #     CUDA-graph capture sizes). FlashAttentionImpl.forward
        #     runs in Python, attn_begin/end fired per layer, and
        #     _attn_count == num_hidden_layers.
        #  2. FULL CUDA-graph replay (small decode batches). Python
        #     doesn't run during replay, so attn_begin/end wasn't
        #     called and _attn_count stayed 0. However, the pool's
        #     External events were captured into the graph during
        #     warmup and fire on every replay — their timestamps are
        #     the replay's. Read them based on known layer count.
        if self._attn_count > 0:
            num_attn_layers = self._attn_count
            source = "eager"
        else:
            num_attn_layers = self.num_hidden_layers
            source = "graph_replay"

        attn_us_total = 0.0
        for i in range(num_attn_layers):
            attn_us_total += (
                self._attn_start_pool[i].elapsed_time(self._attn_end_pool[i])
                * 1000.0
            )
        attn_us_mean = attn_us_total / max(num_attn_layers, 1)
        other_us = max(0.0, total_us - attn_us_total)

        record = {
            "iter_idx": self._step_idx,
            "total_us": total_us,
            "attn_us": attn_us_total,
            "attn_us_mean": attn_us_mean,
            "other_us": other_us,
            "num_attn_layers": num_attn_layers,
            "attn_source": source,
            "request_shapes": self._request_shapes,
        }
        with self._lock:
            self._file.write(json.dumps(record) + "\n")
            self._counter += 1

        self._active = False
        self._attn_count = 0
        self._request_shapes = []
        self._pending_attn_start = None

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()


# -- module-global tracer (one per worker process) -------------------------

_TRACER: Optional[IterationTracer] = None
_TRACER_LOCK = threading.Lock()


def init_tracer(num_hidden_layers: int) -> Optional[IterationTracer]:
    """Initialize the per-worker tracer. No-op when env var is unset or
    when this worker is not rank 0. Safe to call multiple times."""
    global _TRACER
    if not is_enabled():
        return None
    # Only rank 0 writes. In multi-worker (TP>1) setups non-zero
    # ranks see identical iteration shapes; rank-0 data is sufficient
    # for the validation we care about.
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return None
    except Exception:
        pass
    with _TRACER_LOCK:
        if _TRACER is None:
            path = os.getenv(_ENV_PATH, _DEFAULT_PATH)
            _TRACER = IterationTracer(
                output_path=path,
                num_hidden_layers=num_hidden_layers,
                max_attn_layers=max(num_hidden_layers, 1) + 8,  # safety margin
            )
        return _TRACER


def get_tracer() -> Optional[IterationTracer]:
    return _TRACER
