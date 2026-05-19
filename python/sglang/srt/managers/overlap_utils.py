from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_is_cuda = is_cuda()
_is_hip = is_hip()


def _resolve_future_token_ids_native(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


if _is_cuda or _is_hip:
    from sglang.jit_kernel.resolve_future_token_ids import (
        resolve_future_token_ids_cuda,
    )

    _resolve_future_token_ids = resolve_future_token_ids_cuda
else:
    _resolve_future_token_ids = _resolve_future_token_ids_native


@dataclass
class RelayerHandle:
    indices: torch.Tensor
    interval: Optional[slice] = None


class Relayer:
    def __init__(
        self,
        max_running_requests: int,
        chunked_prefill_size: int,
        context_len: int,
        device: torch.device,
        spec_algo: SpeculativeAlgorithm,
    ):
        # FIXME: the calculation of future_limit and future_buffer_len maybe too conservative
        self.future_ct = 0

        # Circular buffer layout (wraps in this order):
        # Running decode batch -> Prefill chunk 1 -> ... -> Prefill chunk N
        # A running decode batch's result will be resolved after all prefill chunks are done.
        # reserve `max_num_chunks` extra future slots on top of `max_running_requests * 3`.
        max_num_chunks = (
            (context_len + chunked_prefill_size - 1) // chunked_prefill_size
            if chunked_prefill_size
            else 0
        )
        self.future_limit = max_running_requests * (3 + max_num_chunks)
        # Adding 2 * max_running_requests to future_limit ensures the buffer is sufficiently large.
        self.future_buffer_len = self.future_limit + 2 * max_running_requests
        self.device = device
        self.spec_algo = spec_algo

        if self.spec_algo.is_none():
            # For non-speculative decoding, we only need to store the token ids.
            self.buf_initialized = True
            self.token_ids_buf = torch.empty(
                (self.future_buffer_len,), dtype=torch.int64, device=self.device
            )
        else:
            # For speculative decoding, we lazily initialize the buffers
            # This is to make the shape derivation easier.
            self.buf_initialized = False

        # Producer event for spec V2 forward outputs. Recorded by
        # store_for_new_batch on the producer (forward) stream; waited on by
        # resolve_draft_input_for_handoff on the consumer (schedule) stream.
        # Replaces the global verify_done barrier.
        self._spec_v2_producer_event: Optional[torch.cuda.Event] = None

    def _lazy_init_buf(self, draft_input: EagleDraftInput):
        self.buf_initialized = True

        # Get a reference for each tensor
        topk_p0 = draft_input.topk_p[0]
        topk_index0 = draft_input.topk_index[0]
        bonus_token0 = draft_input.bonus_tokens[0]
        new_seq_lens0 = draft_input.new_seq_lens[0]

        self.topk_p_buf = torch.empty(
            (self.future_buffer_len, *topk_p0.shape),
            dtype=topk_p0.dtype,
            device=self.device,
        )
        self.topk_index_buf = torch.empty(
            (self.future_buffer_len, *topk_index0.shape),
            dtype=topk_index0.dtype,
            device=self.device,
        )
        self.bonus_tokens_buf = torch.empty(
            (self.future_buffer_len, *bonus_token0.shape),
            dtype=bonus_token0.dtype,
            device=self.device,
        )
        self.new_seq_lens_buf = torch.empty(
            (self.future_buffer_len, *new_seq_lens0.shape),
            dtype=new_seq_lens0.dtype,
            device=self.device,
        )

        if spec_need_hidden_states():
            hidden_states0 = draft_input.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.future_buffer_len, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

    def alloc_handle(self, bs: int) -> RelayerHandle:
        """Update the circular buffer pointer and allocate a relayer handle."""
        cur_future_ct = self.future_ct
        self.future_ct = (cur_future_ct + bs) % self.future_limit
        start = cur_future_ct + 1
        end = cur_future_ct + 1 + bs
        indices = torch.arange(start, end, dtype=torch.int64, device=self.device)
        return RelayerHandle(indices=indices, interval=slice(start, end))

    def resolve_future(self, batch: ScheduleBatch):
        if self.spec_algo.is_none():
            _resolve_future_token_ids(batch.input_ids, self.token_ids_buf)
        else:
            # TODO(lsyin): write relayer handle into spec_info.relayer_handle
            draft_input: EagleDraftInput = batch.spec_info
            if draft_input is None:
                # FIXME(lsyin): No future exists, only for prefill batch, not compatible with mixed mode
                return
            indices = draft_input.relayer_handle.indices
            # The indices tensor was allocated on the default stream but is
            # used here on the forward stream. Meanwhile, the old spec_info
            # holding this tensor will lose all Python references (replaced at
            # batch.spec_info), so the caching allocator (torch GC) could
            # reclaim the memory before the GPU finishes reading it.
            indices.record_stream(torch.get_device_module(self.device).current_stream())
            draft_input.topk_p = self.topk_p_buf[indices]
            draft_input.topk_index = self.topk_index_buf[indices]
            draft_input.bonus_tokens = self.bonus_tokens_buf[indices]
            draft_input.new_seq_lens = self.new_seq_lens_buf[indices]
            if spec_need_hidden_states():
                draft_input.hidden_states = self.hidden_states_buf[indices]

    def is_empty_slice(self, s: slice) -> bool:
        start, stop, step = s.indices(self.future_buffer_len)
        if step > 0:
            return start >= stop
        else:
            return start <= stop

    def store(self, handle: RelayerHandle, batch_result: GenerationBatchResult):
        if self.spec_algo.is_none():
            intv = handle.interval
            if self.is_empty_slice(intv):
                # idle indices in dp attention do not need store info
                return
            self.token_ids_buf[intv] = batch_result.next_token_ids
        else:
            draft_input: EagleDraftInput = batch_result.next_draft_input
            self.store_for_new_batch(handle, draft_input)

    def store_for_new_batch(self, handle: RelayerHandle, draft_input: EagleDraftInput):
        intv = handle.interval
        if self.is_empty_slice(intv):
            # idle indices in dp attention do not need store info
            return

        if not self.buf_initialized:
            self._lazy_init_buf(draft_input)

        self.topk_p_buf[intv] = draft_input.topk_p
        self.topk_index_buf[intv] = draft_input.topk_index
        self.bonus_tokens_buf[intv] = draft_input.bonus_tokens
        self.new_seq_lens_buf[intv] = draft_input.new_seq_lens
        if spec_need_hidden_states():
            self.hidden_states_buf[intv] = draft_input.hidden_states

        # Record producer event on the current (forward) stream after the
        # buf writes. Consumers wait this event before reading channel views.
        event = torch.get_device_module(self.device).Event()
        event.record()
        self._spec_v2_producer_event = event

    def _wait_spec_v2_producer(
        self, consumer_stream: Optional[torch.cuda.Stream] = None
    ):
        event = self._spec_v2_producer_event
        if event is None:
            return
        if consumer_stream is None:
            consumer_stream = torch.get_device_module(self.device).current_stream()
        event.wait(consumer_stream)

    def resolve_draft_input_for_handoff(self, draft_input: EagleDraftInput):
        """Rebind a spec V2 draft input's tensor fields to channel slot views
        on the consumer (schedule) stream. The cross-stream sync against the
        forward-stream producer is folded into the wait here, so downstream
        SB.seq_lens / SB.spec_info reads on the schedule stream are properly
        ordered without a global verify_done barrier.
        """
        if draft_input is None or draft_input.relayer_handle is None:
            return
        indices = draft_input.relayer_handle.indices
        # Idle / bs=0 batch: producer side (store_for_new_batch) skipped the
        # write on the empty interval and did not record an event for the
        # newly alloc'd slot. Worker-side idle tensors stay attached.
        if indices.numel() == 0:
            return
        self._wait_spec_v2_producer()
        # The old spec_info holding `indices` loses its only Python ref when
        # caller does `batch.spec_info = draft_input` after this returns; the
        # caching allocator could reclaim the memory before the GPU finishes
        # reading it on the current stream. Defer reclaim via record_stream
        # until the RelayerHandle becomes a Relayer-owned ref (future work).
        indices.record_stream(torch.get_device_module(self.device).current_stream())
        draft_input.topk_p = self.topk_p_buf[indices]
        draft_input.topk_index = self.topk_index_buf[indices]
        draft_input.bonus_tokens = self.bonus_tokens_buf[indices]
        draft_input.new_seq_lens = self.new_seq_lens_buf[indices]
        if spec_need_hidden_states():
            draft_input.hidden_states = self.hidden_states_buf[indices]

    def handoff_to_next_iter(self, batch: ScheduleBatch, handle: RelayerHandle) -> None:
        """Install output_ids placeholder for next iter's scheduling prep.
        Negated handle indices serve as the placeholder; resolve_future fills
        in real tokens later on the forward stream. The spec V2 SB install
        is done separately by apply_spec_v2_relay_outputs after the channel
        store completes.
        """
        batch.output_ids = -handle.indices

    def apply_spec_v2_relay_outputs(
        self,
        batch: ScheduleBatch,
        handle: RelayerHandle,
        batch_result: GenerationBatchResult,
    ) -> None:
        """Install spec V2 worker output onto SB as channel-view-backed refs.
        Caller must ensure store_for_new_batch has populated the channel
        buffers; the cross-stream wait is folded into resolve_draft_input_for_handoff
        so any subsequent SB-side read carries the producer event.wait inline.
        """
        draft_input: EagleDraftInput = batch_result.next_draft_input
        draft_input.relayer_handle = handle
        self.resolve_draft_input_for_handoff(draft_input)
        batch.spec_info = draft_input
        batch.seq_lens = draft_input.new_seq_lens
