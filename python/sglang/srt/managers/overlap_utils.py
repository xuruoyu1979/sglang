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
    # Contiguous slot range from alloc_handle; consumed by Relayer.store* via
    # is_empty_slice. None after filter / merge (subset is not contiguous).
    interval: Optional[slice] = None

    def filter(self, keep_indices: torch.Tensor) -> RelayerHandle:
        return RelayerHandle(indices=self.indices[keep_indices])

    def merge(self, other: RelayerHandle) -> RelayerHandle:
        return RelayerHandle(indices=torch.cat([self.indices, other.indices]))


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
            # Spec V2: bufs are lazily inited per phase from first store.
            self._verify_bufs_initialized = False
            self._draft_extend_bufs_initialized = False

        # Recorded on forward stream by store_post_verify; awaited on
        # schedule stream in resolve_draft_input_for_handoff so
        # batch.seq_lens.cpu().item() in next iter's prepare_for_decode can
        # overlap with the draft_extend that follows on forward stream.
        # No event for draft_extend outputs: their channel views are
        # consumed only on forward stream (resolve_future at next forward,
        # same stream as the producer), so same-stream ordering suffices.
        self._event_post_verify: Optional[torch.cuda.Event] = None

    def _lazy_init_verify_bufs(self, draft_input: EagleDraftInput):
        self._verify_bufs_initialized = True
        new_seq_lens0 = draft_input.new_seq_lens[0]
        bonus_token0 = draft_input.bonus_tokens[0]
        self.new_seq_lens_buf = torch.empty(
            (self.future_buffer_len, *new_seq_lens0.shape),
            dtype=new_seq_lens0.dtype,
            device=self.device,
        )
        self.bonus_tokens_buf = torch.empty(
            (self.future_buffer_len, *bonus_token0.shape),
            dtype=bonus_token0.dtype,
            device=self.device,
        )

    def _lazy_init_draft_extend_bufs(self, draft_input: EagleDraftInput):
        self._draft_extend_bufs_initialized = True
        topk_p0 = draft_input.topk_p[0]
        topk_index0 = draft_input.topk_index[0]
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
            draft_input: EagleDraftInput = batch.spec_info
            if draft_input is None or batch.relayer_handle is None:
                # FIXME(lsyin): No future exists, only for prefill batch, not compatible with mixed mode
                return
            indices = batch.relayer_handle.indices
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
        """Non-spec: write token_ids_buf. Spec V2: write the draft-extend
        portion (topk_p / topk_index / hidden_states); the verify portion
        (new_seq_lens / bonus_tokens) is written earlier by the worker via
        store_post_verify so schedule-stream consumers can read it without
        waiting for draft_extend.
        """
        if self.spec_algo.is_none():
            intv = handle.interval
            if self.is_empty_slice(intv):
                return
            self.token_ids_buf[intv] = batch_result.next_token_ids
        else:
            self.store_post_draft_extend(handle, batch_result.next_draft_input)

    def store_post_verify(self, handle: RelayerHandle, draft_input: EagleDraftInput):
        """Called from inside worker.verify between sample and
        _draft_extend_for_decode. Writes verify-phase outputs to channel and
        records event_post_verify so schedule-stream consumers (the
        .cpu()/.item() hot path on seq_lens) can overlap with draft_extend.
        """
        intv = handle.interval
        if self.is_empty_slice(intv):
            return
        if not self._verify_bufs_initialized:
            self._lazy_init_verify_bufs(draft_input)
        self.new_seq_lens_buf[intv] = draft_input.new_seq_lens
        self.bonus_tokens_buf[intv] = draft_input.bonus_tokens
        event = torch.get_device_module(self.device).Event()
        event.record()
        self._event_post_verify = event

    def store_post_draft_extend(
        self, handle: RelayerHandle, draft_input: EagleDraftInput
    ):
        """Writes draft-extend outputs to channel. No event needed: the
        channel views are consumed on forward stream at next iter's
        resolve_future, same stream as this write."""
        intv = handle.interval
        if self.is_empty_slice(intv):
            return
        if not self._draft_extend_bufs_initialized:
            self._lazy_init_draft_extend_bufs(draft_input)
        self.topk_p_buf[intv] = draft_input.topk_p
        self.topk_index_buf[intv] = draft_input.topk_index
        if spec_need_hidden_states():
            self.hidden_states_buf[intv] = draft_input.hidden_states

    def resolve_draft_input_for_handoff(
        self, handle: RelayerHandle, draft_input: EagleDraftInput
    ):
        """Rebind new_seq_lens to its channel slot on the consumer (schedule)
        stream, waiting event_post_verify. Other fields (bonus_tokens /
        topk_p / topk_index / hidden_states) are consumed only on forward
        stream — resolve_future at next forward rebinds them in-place there
        (same stream as the producer, no event needed).
        """
        if handle is None or draft_input is None:
            return
        indices = handle.indices
        # Empty interval: producer skipped store and recorded no event;
        # worker-side idle tensors stay attached.
        if indices.numel() == 0:
            return
        stream = torch.get_device_module(self.device).current_stream()
        # Old SB.relayer_handle's only Python ref is dropped when caller
        # rebinds; record_stream defers allocator reclaim until current
        # stream's reads complete.
        indices.record_stream(stream)
        if self._event_post_verify is not None:
            self._event_post_verify.wait(stream)
        draft_input.new_seq_lens = self.new_seq_lens_buf[indices]

    def apply_outputs(
        self,
        batch: ScheduleBatch,
        handle: RelayerHandle,
        batch_result: GenerationBatchResult,
    ) -> None:
        """Post-forward SB install: input_ids = -handle.indices placeholder
        (resolve_future fills next iter); spec V2 non-delay also rebinds
        spec_info / seq_lens to channel views. Delay-sample defers the
        spec V2 portion to launch_batch_sample_if_needed after store().
        """
        batch.input_ids = -handle.indices
        if batch.is_spec_v2 and batch_result.delay_sample_func is None:
            self.apply_spec_v2_relay_outputs(batch, handle, batch_result)

    def apply_pre_forward_decode_delta(self, batch: ScheduleBatch) -> None:
        """Overlap-mode non-spec pre-forward seq_lens bump + post-+1 readers.
        Noop for spec algos (V2 updates seq_lens post-forward; V1 in worker)."""
        if not self.spec_algo.is_none():
            return
        batch.apply_pre_forward_decode_delta()

    def apply_spec_v2_relay_outputs(
        self,
        batch: ScheduleBatch,
        handle: RelayerHandle,
        batch_result: GenerationBatchResult,
    ) -> None:
        """Install spec V2 outputs onto SB as channel-view-backed refs.
        Caller must run store_post_verify (worker side) + store_post_draft_extend
        first; per-event waits are folded into resolve_draft_input_for_handoff.
        batch.relayer_handle is set by scheduler before forward."""
        draft_input: EagleDraftInput = batch_result.next_draft_input
        self.resolve_draft_input_for_handoff(handle, draft_input)
        batch.spec_info = draft_input
        batch.seq_lens = draft_input.new_seq_lens
