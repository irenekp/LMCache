# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Optional, Union
import os
import threading
import uuid

# Third Party
from vllm.config import (
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tp_group,
)
from vllm.sampling_params import SamplingParams
from vllm.utils import cdiv, get_kv_cache_torch_dtype
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash
from vllm.version import __version__ as VLLM_VERSION
import torch

# First Party
from lmcache import utils
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    extract_mm_features,
    lmcache_get_or_create_config,
    mla_enabled,
)
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEvictEvent, CacheStoreEvent, _lmcache_nvtx_annotate
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.compute.blend import LMCBlenderBuilder
from lmcache.v1.config import LMCacheEngineConfig, _validate_and_set_config_value
from lmcache.v1.gpu_connector import (
    GPUConnectorTimingSink,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.lookup_client import LookupClientFactory
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupServer,
)
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.v1.plugin.plugin_launcher import PluginLauncher

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _split_tier_stats(tier_stats: Any) -> dict[str, int]:
    if not isinstance(tier_stats, dict):
        return {}
    min_payload = tier_stats.get("min")
    if isinstance(min_payload, dict):
        return dict(min_payload)
    return dict(tier_stats)


def _normalize_tier_segments(tier_segments: Any) -> list[tuple[str, int]]:
    if not isinstance(tier_segments, list):
        return []
    out: list[tuple[str, int]] = []
    for item in tier_segments:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        backend_name, token_count = item
        if not isinstance(backend_name, str):
            continue
        if not isinstance(token_count, int) or token_count <= 0:
            continue
        out.append((backend_name, int(token_count)))
    return out


def _clip_tier_segments_to_interval(
    tier_segments: list[tuple[str, int]],
    *,
    start_offset: int,
    end_offset: int,
) -> dict[str, int]:
    if end_offset <= start_offset:
        return {}
    by_tier: dict[str, int] = {}
    cursor = 0
    for backend_name, token_count in tier_segments:
        seg_start = cursor
        seg_end = cursor + int(token_count)
        overlap_start = max(seg_start, int(start_offset))
        overlap_end = min(seg_end, int(end_offset))
        if overlap_end > overlap_start:
            by_tier[backend_name] = (
                by_tier.get(backend_name, 0) + (overlap_end - overlap_start)
            )
        cursor = seg_end
        if cursor >= end_offset:
            break
    return {str(k): int(v) for k, v in by_tier.items() if int(v) > 0}


def _fallback_tier_accounting(
    *,
    aggregate_tiers: Optional[dict[str, int]],
    host_fetched_tokens: int,
) -> dict[str, int]:
    host_fetched_tokens = int(host_fetched_tokens)
    if host_fetched_tokens <= 0:
        return {}

    normalized = {
        str(k): int(v)
        for k, v in (aggregate_tiers or {}).items()
        if int(v) > 0
    }
    if not normalized:
        return {"external": host_fetched_tokens}
    if len(normalized) == 1:
        tier_name = next(iter(normalized))
        return {tier_name: host_fetched_tokens}

    total = sum(normalized.values())
    if total <= 0:
        return {"external": host_fetched_tokens}

    out: dict[str, int] = {}
    remaining = host_fetched_tokens
    items = list(normalized.items())
    for idx, (tier_name, token_count) in enumerate(items):
        if idx == len(items) - 1:
            alloc = remaining
        else:
            alloc = min(
                remaining,
                (host_fetched_tokens * int(token_count)) // int(total),
            )
        if alloc > 0:
            out[tier_name] = alloc
            remaining -= alloc
    if remaining > 0:
        last_tier = items[-1][0]
        out[last_tier] = out.get(last_tier, 0) + remaining
    return {k: v for k, v in out.items() if v > 0}


def _build_cache_accounting(
    *,
    load_spec: "LoadSpec",
    host_fetched_tokens: int,
) -> dict[str, Any]:
    gpu_resident_tokens = max(0, int(load_spec.vllm_cached_tokens))
    host_fetched_tokens = max(0, int(host_fetched_tokens))
    clipped = _clip_tier_segments_to_interval(
        list(load_spec.lmcache_tier_hit_segments or []),
        start_offset=gpu_resident_tokens,
        end_offset=gpu_resident_tokens + host_fetched_tokens,
    )
    clipped_total = sum(clipped.values())
    if clipped_total != host_fetched_tokens:
        if clipped and clipped_total < host_fetched_tokens:
            last_tier = next(reversed(clipped))
            clipped[last_tier] += host_fetched_tokens - clipped_total
        else:
            clipped = _fallback_tier_accounting(
                aggregate_tiers=load_spec.lmcache_tier_hit_tokens,
                host_fetched_tokens=host_fetched_tokens,
            )

    total_cached_tokens = gpu_resident_tokens + host_fetched_tokens
    return {
        "gpu_resident_tokens": gpu_resident_tokens,
        "host_fetched_tokens": host_fetched_tokens,
        "host_fetched_tokens_by_tier": clipped,
        "total_cached_tokens": total_cached_tokens,
    }


def _normalize_block_hash_for_kv_events(block_hash: Any) -> Any:
    # vLLM KV cache manager emits hashes via maybe_convert_block_hash(...)
    # (int by default). Keep connector-side hashes in the same representation.
    if isinstance(block_hash, (bytes, bytearray)):
        return maybe_convert_block_hash(bytes(block_hash))
    return block_hash


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool
    # Effective prompt length used for LMCache lookup (after skip_last_n_tokens etc.)
    lookup_prompt_len: int
    # True if vLLM forces recomputing last token in the full-hit case.
    recalc_last_token: bool = False
    lmcache_tier_hit_tokens: Optional[dict[str, int]] = None
    lmcache_tier_hit_segments: Optional[list[tuple[str, int]]] = None


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str
    receiver_host: str
    receiver_init_port: int
    receiver_alloc_port: int
    is_last_prefill: bool = False
    num_transferred_tokens: int = 0


tmp_disagg_tracker: dict[str, DisaggSpec] = {}


def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    request_configs = None
    if sampling_params.extra_args is not None:
        if kv_transfer_params := sampling_params.extra_args.get("kv_transfer_params"):
            for k, v in kv_transfer_params.items():
                if k.startswith("lmcache."):
                    if request_configs is None:
                        request_configs = {}
                    request_configs[k] = v
    return request_configs


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    # FIXME: need to check whether the block ids will be changed after
    #        preemption
    allocated_block_ids: list[int]
    # The vLLM block hashes corresponding to full blocks.
    block_hashes: list[Any] = field(default_factory=list)

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # Multimodal hashes and positions
    mm_hashes: Optional[list[str]] = None
    mm_positions: Optional[list["PlaceholderRange"]] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
        block_hashes: Optional[list[Any]] = None,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            lmcache_config (LMCacheEngineConfig): the LMCache engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lmcache_cached_tokens (int): the number of tokens that are
                cached in LMCache.
            request_priority (int): the priority of the request
            skip_save (bool): whether the request cache should be saved
        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # list[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            # According to the vLLM code
            # (https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/
            # sched/scheduler.py#L943),
            # only one KVCacheGroup is supported in connector for now.

            # TODO: Please support multiple KVCacheGroup in connector.
            # NOTE: Also, `update` method in RequestTracker should be
            # updated accordingly.
            unfolded_block_ids = new_request.block_ids[0].copy()

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)

        request_configs = extract_request_configs(new_request.sampling_params)

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            block_hashes=list(block_hashes or []),
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            skip_save=skip_save,
            request_configs=request_configs,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again
        """

        self.token_ids.extend(new_token_ids)

        if new_block_ids is None:
            # https://github.com/vllm-project/vllm/commit/
            # b029de9902aa3ac58806c8c17776c7074175b6db#
            # diff-cafd89ce8a698a56acb24ada62831cbc7a980782f78a52d1742ba238031f296cL94
            new_block_ids = []
        elif len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            pass
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")
        self.allocated_block_ids.extend(new_block_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        # TODO: Need to further exclude the case of chunked prefill with 1 token.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    # disagg spec
    disagg_spec: Optional[DisaggSpec] = None
    # the configs of the request
    request_configs: Optional[dict] = None
    # vLLM block hashes aligned with chunk indices (full blocks only)
    block_hashes: Optional[list[Any]] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        is_last_prefill = False
        if input_token_len == tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a transfer
        # Check if request_configs has lmcache.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)

        skip_save = tracker.disagg_spec is None and (
            tracker.skip_save
            or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
            or (tracker.is_decode_phase and not save_decode_cache)
            or request_skip
        )

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        num_tokens_to_save = (
            (input_token_len // lmcache_chunk_size * lmcache_chunk_size)
            if not is_last_prefill or discard_partial_chunks
            else input_token_len
        )

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        token_ids = input_token_ids[:num_tokens_to_save]

        # If the request has multimodal hashes, apply them to the token ids
        if tracker.mm_hashes:
            # TODO: Optimize this
            token_ids = torch.tensor(token_ids)
            assert tracker.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_ids, tracker.mm_hashes, tracker.mm_positions
            )
            token_ids = token_ids.tolist()

        num_blocks = len(tracker.allocated_block_ids)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks."
                "Something might be wrong in scheduling logic!"
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids.reshape((num_blocks, 1)) * block_size
        )

        slot_mapping = slot_mapping.flatten()[: len(token_ids)]
        assert slot_mapping.dtype == torch.long  # TODO: this could be removed

        # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.lmcache_cached_tokens,
                tracker.req_id,
            )
        else:
            # Do not load if not in `can_load` state
            load_spec = None

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
            block_hashes=tracker.block_hashes,
        )


def need_gpu_interm_buffer(lmcache_config: LMCacheEngineConfig):
    if lmcache_config.enable_pd:
        return False
    else:
        return True


def _calculate_draft_layers(vllm_config, model_config):
    num_draft_layers = 0
    if vllm_config is not None and vllm_config.speculative_config is not None:
        logger.info(f"vllm_config.speculative_config: {vllm_config.speculative_config}")
        # TODO(baoloongmao): Support other MTP/draft methods
        if vllm_config.speculative_config.method == "deepseek_mtp":
            num_draft_layers = getattr(
                model_config.hf_config, "num_nextn_predict_layers", 0
            )
        elif vllm_config.speculative_config.use_eagle():
            try:
                draft_model_config = vllm_config.speculative_config.draft_model_config
                num_draft_layers = draft_model_config.get_num_layers(
                    vllm_config.parallel_config
                )
                logger.info(f"EAGLE detected {num_draft_layers} extra layer(s)")
            except Exception:
                logger.info(
                    "EAGLE detected, but failed to get the number of extra layers"
                    "falling back to 1"
                )
                num_draft_layers = 1
    return num_draft_layers


def _init_lmcache_engine(
    lmcache_config: LMCacheEngineConfig,
    vllm_config: "VllmConfig",
) -> LMCacheEngine:
    """Initialize the LMCache engine by the given model config and parallel
    config. This function will check the environment variable
    `LMCACHE_CONFIG_FILE` to load the configuration file. If that environment
    variable is not set, this function will return None.

    :param lmcache_config: The LMCache configuration.
    :type lmcache_config: LMCacheEngineConfig
    :param vllm_config: The vLLM configuration.
    :type vllm_config: VllmConfig

    :return: The initialized LMCache engine
    :rtype: LMCacheEngine
    """
    if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
        return curr_engine

    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config

    assert isinstance(lmcache_config, LMCacheEngineConfig), (
        "LMCache v1 configuration is should be passed."
    )

    kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype, model_config.dtype)

    use_mla = mla_enabled(model_config)
    if use_mla and (
        lmcache_config.remote_serde != "naive"
        and lmcache_config.remote_serde is not None
    ):
        raise ValueError("MLA only works with naive serde mode..")

    # construct kv shape (for mem pool)
    num_layer = model_config.get_num_layers(parallel_config)
    num_draft_layers = _calculate_draft_layers(vllm_config, model_config)
    num_layer += num_draft_layers
    chunk_size = lmcache_config.chunk_size
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)
    logger.info(
        f"use mla: {use_mla}, kv shape: {kv_shape}, num_draft_layers:{num_draft_layers}"
    )

    # Change current device.
    num_gpus = torch.cuda.device_count()
    local_rank = parallel_config.rank % num_gpus
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    metadata = LMCacheEngineMetadata(
        model_config.model,
        parallel_config.world_size,
        parallel_config.rank,
        "vllm",
        kv_dtype,
        kv_shape,
        use_mla,
    )

    use_gpu = need_gpu_interm_buffer(lmcache_config)
    vllm_gpu_connector: Union[
        VLLMBufferLayerwiseGPUConnector,
        VLLMPagedMemGPUConnectorV2,
        VLLMPagedMemLayerwiseGPUConnector,
    ]

    if use_mla and lmcache_config.use_layerwise:
        raise ValueError("layerwise MLA connector is not supported yet")

    # When use_mla is True, num_kv_head is 1
    hidden_dim_size = num_kv_head * head_size
    if lmcache_config.use_layerwise:
        if lmcache_config.enable_blending:
            # Use layerwise connector for blending
            vllm_gpu_connector = VLLMBufferLayerwiseGPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
        else:
            vllm_gpu_connector = VLLMPagedMemLayerwiseGPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
    else:
        vllm_gpu_connector = VLLMPagedMemGPUConnectorV2(
            hidden_dim_size,
            num_layer,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=kv_dtype,
            device=device,
            use_mla=use_mla,
        )
    tpg = get_tp_group()
    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME,
        lmcache_config,
        metadata,
        vllm_gpu_connector,
        tpg.broadcast,
        tpg.broadcast_object,
    )

    return engine


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)
    lookup_requests_in_step: list[str] = field(default_factory=list)

    @_lmcache_nvtx_annotate
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class LMCacheConnectorV1Impl:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        self._parent = parent
        self._vllm_config = vllm_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = vllm_config.parallel_config.tensor_parallel_size
        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        # Put the leading with "lmcache." and matched configs from
        # vllm extra_config to the config
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        if kv_connector_extra_config:
            for key, value in kv_connector_extra_config.items():
                if key.startswith("lmcache."):
                    config_key = key[8:]  # Remove "lmcache." prefix
                    if _validate_and_set_config_value(config, config_key, value):
                        logger.info(
                            f"Updated config {config_key} from vLLM "
                            f"extra config: {value}"
                        )

        self.config = config

        self.async_loading = config.enable_async_loading
        self.layerwise_retrievers: list[
            Generator[Optional[torch.Tensor], None, None]
        ] = []
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        if role == KVConnectorRole.SCHEDULER:
            # Create lookup client using factory
            self.lookup_client = LookupClientFactory.create_lookup_client(
                vllm_config, config
            )
            self._unfinished_requests: dict[str, Request] = {}
            self._lookup_requests_in_step: list[str] = []
            self.lmcache_engine = None
        else:
            self.lmcache_engine = _init_lmcache_engine(
                config,
                vllm_config,
            )

            self.use_layerwise = config.use_layerwise
            self.enable_blending = config.enable_blending

            if self.enable_blending:
                self.blender = LMCBlenderBuilder.get_or_create(
                    ENGINE_NAME,
                    self.lmcache_engine,
                    self.lmcache_engine.gpu_connector,
                    config,
                )

            # Create lookup server using factory
            assert self.lmcache_engine is not None
            self.lookup_server = LookupClientFactory.create_lookup_server(
                self.lmcache_engine, vllm_config
            )

            self.offload_server = ZMQOffloadServer(
                self.lmcache_engine,
                vllm_config,
                get_tensor_model_parallel_rank(),
            )

            # In case of MLA, the lookup server is only created on worker 0
            if self.async_loading and self.lookup_server is not None:
                assert isinstance(self.lookup_server, LMCacheAsyncLookupServer)
                self.lmcache_engine.post_init(async_lookup_server=self.lookup_server)

        self.kv_caches: dict[str, torch.Tensor] = {}

        self._block_size = vllm_config.cache_config.block_size

        # request_id -> (vllm cached tokens, lmcache cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}

        self.kv_cache_manager: Optional[KVCacheManager] = None

        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}

        # Whether to discard partial chunks
        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
            or not config.save_unfull_chunk
        )

        self._lmcache_chunk_size = config.chunk_size
        self._save_decode_cache = config.save_decode_cache

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.current_layer = 0
        self._timing_sink: Optional[GPUConnectorTimingSink] = None

        self.force_skip_save = bool(os.environ.get("LMCACHE_FORCE_SKIP_SAVE", False))

        self._requests_priority: dict[str, int] = {}
        # Map LMCache chunk hash to the covered vLLM block hash(es) and token size.
        self._hash_translation: dict[Any, list[Any]] = {}
        self._hash_translation_sizes: dict[Any, int] = {}
        self._hash_translation_lock = threading.Lock()
        self._debug_hash_translation = _env_flag(
            "LMCACHE_DEBUG_HASH_TRANSLATION", False
        )

        # TODO(baoloongmao): Internal api server & plugin framework support dp > 1
        if vllm_config.parallel_config.data_parallel_rank_local == 0:
            # Start internal API server if enabled
            # The enabled check is in the InternalAPIServer constructor
            self.api_server = InternalAPIServer(self)
            self.api_server.start()
            # Launch plugins
            self.plugin_launcher = PluginLauncher(
                self.config,
                role,
                self.worker_count,
                -1
                if self.lmcache_engine is None  # scheduler side
                else self.lmcache_engine.metadata.worker_id,
            )
            self.plugin_launcher.launch_plugins()
        else:
            self.api_server = None  # type: ignore[assignment]
            self.plugin_launcher = None  # type: ignore[assignment]
        logger.info(
            f"LMCache initialized for role {role} with version {utils.get_version()}, "
            f"vllm version {VLLM_VERSION}, "
            "lmcache cache_engine metadata: "
            f"{getattr(self.lmcache_engine, 'metadata', None)}"
        )

    def get_inference_info(self) -> dict:
        """Get inference information including vLLM config and related details.

        Returns:
            dict: Dictionary containing inference information
        """
        # Get vLLM config information
        vllm_config = self._vllm_config

        # Use vLLM config's string representation and add specific configs
        inference_info = {
            "vllm_version": VLLM_VERSION,
            "lmcache_version": utils.get_version(),
            "vllm_config": str(vllm_config),
            "model_config": {
                "model": getattr(vllm_config.model_config, "model", None),
                "dtype": str(getattr(vllm_config.model_config, "dtype", None)),
                "max_model_len": getattr(
                    vllm_config.model_config, "max_model_len", None
                ),
                "vocab_size": getattr(vllm_config.model_config, "vocab_size", None),
                "num_layers": getattr(
                    vllm_config.model_config, "get_num_layers", lambda _: None
                )(vllm_config.parallel_config),
                "num_attention_heads": getattr(
                    vllm_config.model_config, "get_num_attention_heads", lambda _: None
                )(vllm_config.parallel_config),
                "num_kv_heads": getattr(
                    vllm_config.model_config, "get_num_kv_heads", lambda _: None
                )(vllm_config.parallel_config),
                "head_size": getattr(
                    vllm_config.model_config, "get_head_size", lambda: None
                )(),
            },
            "cache_config": {
                "block_size": getattr(vllm_config.cache_config, "block_size", None),
                "cache_dtype": str(
                    getattr(vllm_config.cache_config, "cache_dtype", None)
                ),
                "gpu_memory_utilization": getattr(
                    vllm_config.cache_config, "gpu_memory_utilization", None
                ),
                "swap_space": getattr(vllm_config.cache_config, "swap_space", None),
                "enable_prefix_caching": getattr(
                    vllm_config.cache_config, "enable_prefix_caching", None
                ),
            },
        }

        return inference_info

    def get_inference_version(self) -> str:
        """Get vLLM version information.

        Returns:
            str: vLLM version string
        """
        return VLLM_VERSION

    @_lmcache_nvtx_annotate
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]

    ####################
    # Worker side APIs
    ####################

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        self.current_layer = 0

        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        assert self.lmcache_engine is not None

        self.lmcache_engine.post_init(kvcaches=kvcaches)

        self.layerwise_retrievers = []

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None:
                continue
            last_idx = idx

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None:
                continue

            tokens = request.token_ids
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = request.slot_mapping.cuda()
            assert len(tokens) == len(slot_mapping)

            self._stats_monitor.update_interval_vllm_hit_tokens(
                request.load_spec.vllm_cached_tokens
            )
            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            if self.use_layerwise:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    )
                else:
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        sync=sync,
                    )
                    # NOTE: retrieve for two layers at the first layer
                    next(layerwise_retriever)
                    next(layerwise_retriever)
                    self.layerwise_retrievers.append(layerwise_retriever)
            else:
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                )

                # Check the result
                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!"
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )

    @_lmcache_nvtx_annotate
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if self.layerwise_retrievers:
            logger.debug(f"Waiting for layer {self.current_layer} to be loaded")

        # Wait for the layer to be loaded
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)

            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info(f"Retrieved {num_retrieved_tokens} tokens")

        return

    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        assert self.lmcache_engine is not None

        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self._parent._connector_metadata is None:
            logger.warning(
                "In connector.save_kv_layer, but the connector metadata is None"
            )
            return
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0

        kvcaches = list(self.kv_caches.values())
        if self.current_layer == 0:
            self.layerwise_storers = []

            is_first = True

            for idx, request in enumerate(connector_metadata.requests):
                save_spec = request.save_spec
                if save_spec is None or not save_spec.can_save:
                    continue

                token_ids = request.token_ids
                assert isinstance(token_ids, list)

                slot_mapping = request.slot_mapping
                assert isinstance(slot_mapping, torch.Tensor)
                assert len(slot_mapping) == len(token_ids)

                # TODO: have a pre-allocated buffer to hold the slot_mappings
                slot_mapping = slot_mapping.cuda()

                if self.kv_role == "kv_producer":
                    skip_leading_tokens = 0
                else:
                    skip_leading_tokens = save_spec.skip_leading_tokens

                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    skip_leading_tokens = (
                        skip_leading_tokens
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )

                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                logger.info(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                self._record_hash_translation(
                    token_ids=token_ids,
                    store_mask=store_mask,
                    block_hashes=request.block_hashes,
                    request_configs=request.request_configs,
                )

                # TODO (Jiayi): need to make layerwise storing
                # compatible with disagg spec
                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    sync=is_first,
                )
                self.layerwise_storers.append(layerwise_storer)
                if is_first:
                    is_first = False

        for layerwise_storer in self.layerwise_storers:
            next(layerwise_storer)

        self.current_layer += 1

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        self.lmcache_engine.lookup_unpin(connector_metadata.lookup_requests_in_step)

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return

        if self.use_layerwise:
            for layerwise_storer in self.layerwise_storers:
                next(layerwise_storer)
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            token_ids = request.token_ids

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.cuda()

            skip_leading_tokens = save_spec.skip_leading_tokens
            if self.kv_role == "kv_producer":
                skip_leading_tokens = min(
                    skip_leading_tokens, request.disagg_spec.num_transferred_tokens
                )

            if skip_leading_tokens == len(token_ids):
                continue  # skip this request
            # Align to lmcache chunk size
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.info(
                "Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            is_last_prefill = request.is_last_prefill
            if is_last_prefill:
                if request.disagg_spec:
                    request.disagg_spec.is_last_prefill = True
            else:
                token_len = len(token_ids)
                aligned_token_len = (
                    token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                )
                token_ids = token_ids[:aligned_token_len]
                store_mask = store_mask[:aligned_token_len]
                slot_mapping = slot_mapping[:aligned_token_len]

            self._record_hash_translation(
                token_ids=token_ids,
                store_mask=store_mask,
                block_hashes=request.block_hashes,
                request_configs=request.request_configs,
            )

            self.lmcache_engine.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
                transfer_spec=request.disagg_spec,
                request_configs=request.request_configs,
            )

            # NOTE(Jiayi): We assume all tokens are saved
            save_spec.skip_leading_tokens = len(token_ids)
            if request.disagg_spec:
                request.disagg_spec.num_transferred_tokens = len(token_ids)

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return None, None

    def _record_hash_translation(
        self,
        token_ids: list[int],
        store_mask: torch.Tensor,
        block_hashes: list[Any] | None,
        request_configs: dict | None,
    ) -> None:
        if self.lmcache_engine is None or not block_hashes:
            return
        token_database = getattr(self.lmcache_engine, "token_database", None)
        if token_database is None:
            return
        try:
            entries = token_database.process_tokens(
                tokens=token_ids,
                mask=store_mask,
                make_key=False,
                request_configs=request_configs,
            )
        except Exception as exc:
            logger.debug("Failed to build hash translation table: %s", exc)
            return

        entries_total = 0
        mapped = 0
        skipped = 0
        with self._hash_translation_lock:
            for start, end, lm_hash in entries:
                entries_total += 1
                if end <= start:
                    skipped += 1
                    continue
                block_start_idx = start // self._block_size
                block_end_idx = cdiv(end, self._block_size)
                if block_end_idx > len(block_hashes):
                    skipped += 1
                    continue
                translated_hashes = [
                    _normalize_block_hash_for_kv_events(bh)
                    for bh in block_hashes[block_start_idx:block_end_idx]
                ]
                if not translated_hashes:
                    skipped += 1
                    continue
                self._hash_translation[lm_hash] = translated_hashes
                self._hash_translation_sizes[lm_hash] = end - start
                mapped += 1

            if self._debug_hash_translation:
                map_size = len(self._hash_translation)
                masked_tokens = int(store_mask.sum().item())
                logger.info(
                    "[dup-debug] hash map update: entries=%d mapped=%d skipped=%d "
                    "map_size=%d token_ids=%d masked_tokens=%d block_hashes=%d",
                    entries_total,
                    mapped,
                    skipped,
                    map_size,
                    len(token_ids),
                    masked_tokens,
                    len(block_hashes),
                )

    def _translate_kv_hashes(self, hashes: list[Any]) -> list[Any]:
        if not hashes:
            return []
        with self._hash_translation_lock:
            translated: list[Any] = []
            hits = 0
            misses = 0
            sample_misses: list[Any] = []
            for h in hashes:
                mapped_hashes = self._hash_translation.get(h)
                if mapped_hashes:
                    hits += 1
                    translated.extend(mapped_hashes)
                else:
                    misses += 1
                    translated.append(h)
                    if len(sample_misses) < 4:
                        sample_misses.append(h)

            if self._debug_hash_translation:
                logger.info(
                    "[dup-debug] hash translate: total=%d hits=%d misses=%d "
                    "map_size=%d sample_misses=%s",
                    len(hashes),
                    hits,
                    misses,
                    len(self._hash_translation),
                    sample_misses,
                )
            return translated

    def _evict_hash_translation(self, hashes: list[Any]) -> None:
        if not hashes:
            return
        removed = 0
        with self._hash_translation_lock:
            for h in hashes:
                removed += int(self._hash_translation.pop(h, None) is not None)
                self._hash_translation_sizes.pop(h, None)
            if self._debug_hash_translation:
                logger.info(
                    "[dup-debug] hash evict: requested=%d removed=%d map_size=%d",
                    len(hashes),
                    removed,
                    len(self._hash_translation),
                )

    def _translate_kv_hash_groups(
        self,
        hashes: list[Any],
        fallback_block_size: int,
    ) -> list[tuple[list[Any], int]]:
        """
        Translate LMCache hashes into vLLM hashes and normalize token units.

        The returned groups preserve order and merge adjacent ranges that share
        the same per-hash block_size.
        """
        if not hashes:
            return []

        groups: list[tuple[list[Any], int]] = []
        with self._hash_translation_lock:
            hits = 0
            misses = 0
            sample_misses: list[Any] = []
            for h in hashes:
                mapped_hashes = self._hash_translation.get(h)
                token_count = self._hash_translation_sizes.get(h)

                if mapped_hashes:
                    hits += 1
                    per_hash_block_size = int(fallback_block_size)
                    if (
                        isinstance(token_count, int)
                        and token_count > 0
                        and token_count % len(mapped_hashes) == 0
                    ):
                        per_hash_block_size = token_count // len(mapped_hashes)
                    else:
                        # Keep telemetry self-consistent: when we cannot infer
                        # a per-block size, fall back to one representative hash.
                        mapped_hashes = [mapped_hashes[0]]
                    translated_hashes = list(mapped_hashes)
                else:
                    misses += 1
                    per_hash_block_size = int(fallback_block_size)
                    translated_hashes = [h]
                    if len(sample_misses) < 4:
                        sample_misses.append(h)

                if groups and groups[-1][1] == per_hash_block_size:
                    groups[-1][0].extend(translated_hashes)
                else:
                    groups.append((translated_hashes, per_hash_block_size))

            if self._debug_hash_translation:
                logger.info(
                    "[dup-debug] hash translate: total=%d hits=%d misses=%d "
                    "groups=%d map_size=%d sample_misses=%s",
                    len(hashes),
                    hits,
                    misses,
                    len(groups),
                    len(self._hash_translation),
                    sample_misses,
                )
        return groups

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> list[CacheStoreEvent | CacheEvictEvent]:
        if self.lmcache_engine is None:
            return []
        events = self.lmcache_engine.get_kv_events()
        if not events:
            return []
        if self._debug_hash_translation:
            store_events = sum(1 for e in events if hasattr(e, "parent_block_hash"))
            evict_events = len(events) - store_events
            logger.info(
                "[dup-debug] raw kv events: total=%d stores=%d evicts=%d",
                len(events),
                store_events,
                evict_events,
            )
        translated: list[CacheStoreEvent | CacheEvictEvent] = []
        for event in events:
            translated_groups = self._translate_kv_hash_groups(
                event.block_hashes,
                int(event.block_size),
            )
            if hasattr(event, "parent_block_hash"):
                parent_hash = event.parent_block_hash
                if parent_hash is not None:
                    translated_parent_hashes = self._translate_kv_hashes([parent_hash])
                    if translated_parent_hashes:
                        parent_hash = translated_parent_hashes[-1]

                current_parent_hash = parent_hash
                for idx, (translated_hashes, translated_block_size) in enumerate(
                    translated_groups
                ):
                    if not translated_hashes:
                        continue
                    translated.append(
                        CacheStoreEvent(
                            block_hashes=translated_hashes,
                            parent_block_hash=current_parent_hash,
                            token_ids=event.token_ids if idx == 0 else [],
                            block_size=translated_block_size,
                            lora_id=getattr(event, "lora_id", None),
                            medium=event.medium,
                            lora_name=getattr(event, "lora_name", None),
                        )
                    )
                    current_parent_hash = translated_hashes[-1]
            else:
                for translated_hashes, translated_block_size in translated_groups:
                    if not translated_hashes:
                        continue
                    translated.append(
                        CacheEvictEvent(
                            block_hashes=translated_hashes,
                            block_size=translated_block_size,
                            medium=event.medium,
                        )
                    )
                self._evict_hash_translation(event.block_hashes)
        return translated

    ###################
    # Scheduler side APIs
    ####################

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.kv_role == "kv_producer" and not hasattr(
            self.lookup_client, "supports_producer_reuse"
        ):
            return 0

        self._requests_priority[request.request_id] = getattr(request, "priority", 0)

        token_ids = request.prompt_token_ids

        # If the request has multimodal hashes, apply them to the token ids
        mm_hashes, mm_positions = extract_mm_features(request)
        if mm_hashes and mm_positions:
            # TODO(Jiayi): Optimize this
            token_ids = torch.tensor(request.prompt_token_ids)
            apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)
            token_ids = token_ids.tolist()

        request_configs = extract_request_configs(request.sampling_params)
        if self.skip_last_n_tokens > 0:
            token_ids = token_ids[: -self.skip_last_n_tokens]
        lookup_prompt_len = len(token_ids)
        if self.async_loading:
            lookup_id = request.request_id
        else:
            lookup_id = str(uuid.uuid4())

        self._lookup_requests_in_step.append(lookup_id)

        num_external_hit_tokens = self.lookup_client.lookup(
            token_ids,
            lookup_id=lookup_id,
            request_configs=request_configs,
        )
        tier_min: dict[str, int] = {}
        tier_segments: list[tuple[str, int]] = []
        tier_stats = self.lookup_client.get_tier_stats(lookup_id)
        if tier_stats is not None:
            tier_min = _split_tier_stats(tier_stats)
        tier_segments_raw = self.lookup_client.get_tier_segments(lookup_id)
        if tier_segments_raw is not None:
            tier_segments = _normalize_tier_segments(tier_segments_raw)

        if num_external_hit_tokens is None:
            logger.info(
                "Reqid: %s, Lookup prompt tokens %d, LMCache hit tokens: None.",
                request.request_id,
                lookup_prompt_len,
            )
            return None

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # In, full-prompt-hit case, we need to recompute the last token
        recalc_last_token = (num_external_hit_tokens == lookup_prompt_len)
        if recalc_last_token:
            need_to_allocate -= 1

        logger.info(
            "Reqid: %s, Lookup prompt tokens %d, LMCache hit tokens: %d, need to load: %d",
            request.request_id,
            lookup_prompt_len,
            num_external_hit_tokens,
            need_to_allocate,
        )

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            can_load=False,
            lookup_prompt_len=lookup_prompt_len,
            recalc_last_token=recalc_last_token,
            lmcache_tier_hit_tokens=tier_min or None,
            lmcache_tier_hit_segments=tier_segments or None,
        )
        try:
            kv_params = getattr(request, "kv_transfer_params", None)
            if kv_params is None:
                kv_params = {}
                setattr(request, "kv_transfer_params", kv_params)
            kv_params["_lmcache_telemetry"] = {
                "lookup": {
                    "vllm_cached_tokens": int(num_computed_tokens),
                    "lmcache_cached_tokens": int(num_external_hit_tokens),
                    "lookup_prompt_len": int(lookup_prompt_len),
                    "recalc_last_token": bool(recalc_last_token),
                    "lmcache_tier_hit_tokens": dict(tier_min) if tier_min else None,
                    "lmcache_tier_hit_segments": (
                        list(tier_segments) if tier_segments else None
                    ),
                },
                "cache_accounting": None,
            }
        except Exception:
            logger.exception("Failed to attach _lmcache_telemetry to request.")

        if not self.async_loading:
            self.lookup_client.clear_lookup_status(lookup_id)

        if need_to_allocate <= 0:
            return 0

        # TODO: Align to vLLM block size. Should test whether it can be removed
        # need_to_allocate = need_to_allocate // self._block_size * \
        #        self._block_size

        return need_to_allocate

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(self, request: "Request", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """
        self.lookup_client.clear_lookup_status(request.request_id)

        kv_transfer_params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )

        if kv_transfer_params is not None and "disagg_spec" in kv_transfer_params:
            req_disagg_spec = kv_transfer_params["disagg_spec"]

            receiver_id = req_disagg_spec["receiver_host"] + str(
                req_disagg_spec["receiver_init_port"]
            )

            disagg_spec = DisaggSpec(
                req_id=req_disagg_spec["req_id"],
                receiver_id=receiver_id,
                receiver_host=req_disagg_spec["receiver_host"],
                receiver_init_port=req_disagg_spec["receiver_init_port"],
                receiver_alloc_port=req_disagg_spec["receiver_alloc_port"],
            )

            tmp_disagg_tracker[request.request_id] = disagg_spec
        self._unfinished_requests[request.request_id] = request

        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            if isinstance(kv_transfer_params, dict):
                telemetry = kv_transfer_params.get("_lmcache_telemetry")
                if isinstance(telemetry, dict):
                    telemetry["cache_accounting"] = _build_cache_accounting(
                        load_spec=self.load_specs[request.request_id],
                        host_fetched_tokens=0,
                    )
            return

        recalc_last = 1 if self.load_specs[request.request_id].recalc_last_token else 0
        assert (
            num_external_tokens > 0
            and num_external_tokens
            == self.load_specs[request.request_id].lmcache_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
            - recalc_last
        ), (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].lmcache_cached_tokens} - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens}"
            f" - {recalc_last} for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True
        if isinstance(kv_transfer_params, dict):
            telemetry = kv_transfer_params.get("_lmcache_telemetry")
            if isinstance(telemetry, dict):
                telemetry["cache_accounting"] = _build_cache_accounting(
                    load_spec=self.load_specs[request.request_id],
                    host_fetched_tokens=int(num_external_tokens),
                )

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        meta = LMCacheConnectorMetadata()

        # set and update lookup requests for unpin
        meta.lookup_requests_in_step = self._lookup_requests_in_step
        self._lookup_requests_in_step = []

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)

        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            request_priority = self._requests_priority.pop(request.req_id, 0)

            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            request_obj = self._unfinished_requests.get(request.req_id)
            block_hashes = None
            if request_obj is not None:
                block_hashes = list(getattr(request_obj, "block_hashes", []))

            request_tracker = RequestTracker.from_new_request(
                self.config,
                request,
                num_tokens_to_compute,
                lmcache_cached_tokens,
                skip_save,
                block_hashes=block_hashes,
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self._save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs

        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                request_tracker = self._request_trackers[req.req_id]
                request_tracker.update(req.new_token_ids, req.new_block_ids)
                if request_obj := self._unfinished_requests.get(req.req_id):
                    request_tracker.block_hashes = list(
                        getattr(request_obj, "block_hashes", [])
                    )

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=None,
                    discard_partial_chunks=self._discard_partial_chunks,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if request := self._unfinished_requests.get(req_id):
                num_current_tokens = len(request_tracker.token_ids)
                new_token_ids = request.all_token_ids[
                    num_current_tokens : num_current_tokens + num_new_tokens
                ]
            else:
                raise ValueError(
                    f"Request {req_id} is not in _unfinished_requests, "
                    f"but it is scheduled to be cached"
                )
            new_block_ids = cached_reqs.new_block_ids[i]

            request_tracker.update(new_token_ids, new_block_ids)
            request_tracker.block_hashes = list(getattr(request, "block_hashes", []))

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=None,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self._save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        return meta

    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        return_params = None

        # NOTE: Used to stream back the first token
        # for disagg prefill
        if params is not None and "ret_first_tok" in params:
            return_params = {
                "first_tok": request._output_token_ids[0],
            }

        finished_reason = None
        if hasattr(request, "get_finished_reason"):
            try:
                finished_reason = request.get_finished_reason()
            except Exception:
                finished_reason = None
        if finished_reason is not None and str(finished_reason).lower() == "abort":
            if isinstance(getattr(request, "kv_transfer_params", None), dict):
                request.kv_transfer_params.pop("_lmcache_telemetry", None)
            return False, return_params

        cache_accounting: dict[str, Any] = {
            "gpu_resident_tokens": 0,
            "host_fetched_tokens": 0,
            "host_fetched_tokens_by_tier": {},
            "total_cached_tokens": 0,
        }

        req_id = request.request_id
        load_spec = self.load_specs.pop(req_id, None)
        telemetry = None
        kv_params = getattr(request, "kv_transfer_params", None)
        if isinstance(kv_params, dict):
            telemetry = kv_params.get("_lmcache_telemetry")

        if isinstance(telemetry, dict):
            public_cache_accounting = telemetry.get("cache_accounting")
            if isinstance(public_cache_accounting, dict):
                cache_accounting = {
                    "gpu_resident_tokens": int(
                        public_cache_accounting.get("gpu_resident_tokens", 0) or 0
                    ),
                    "host_fetched_tokens": int(
                        public_cache_accounting.get("host_fetched_tokens", 0) or 0
                    ),
                    "host_fetched_tokens_by_tier": {
                        str(k): int(v)
                        for k, v in (
                            public_cache_accounting.get(
                                "host_fetched_tokens_by_tier"
                            )
                            or {}
                        ).items()
                        if int(v) > 0
                    },
                    "total_cached_tokens": int(
                        public_cache_accounting.get("total_cached_tokens", 0) or 0
                    ),
                }
        if load_spec is not None and not cache_accounting["total_cached_tokens"]:
            recalc_last = 1 if load_spec.recalc_last_token else 0
            host_fetched_tokens = max(
                0,
                int(load_spec.lmcache_cached_tokens)
                - int(load_spec.vllm_cached_tokens)
                - recalc_last,
            )
            cache_accounting = _build_cache_accounting(
                load_spec=load_spec,
                host_fetched_tokens=host_fetched_tokens,
            )

        if return_params is None:
            return_params = {}
        return_params.update(
            {
                "cache_hit": {
                    "gpu_resident_tokens": int(
                        cache_accounting["gpu_resident_tokens"]
                    ),
                    "host_fetched_tokens": int(
                        cache_accounting["host_fetched_tokens"]
                    ),
                    "host_fetched_tokens_by_tier": dict(
                        cache_accounting["host_fetched_tokens_by_tier"]
                    ),
                    "total_cached_tokens": int(
                        cache_accounting["total_cached_tokens"]
                    ),
                }
            }
        )
        if isinstance(getattr(request, "kv_transfer_params", None), dict):
            request.kv_transfer_params.pop("_lmcache_telemetry", None)

        return False, return_params

    def get_timing_sink(self) -> Optional[GPUConnectorTimingSink]:
        return self._timing_sink

    def set_timing_sink(
        self, sink: Optional[GPUConnectorTimingSink]
    ) -> None:
        self._timing_sink = sink
        if getattr(self, "lmcache_engine", None) is None:
            return
        gpu_connector = getattr(self.lmcache_engine, "gpu_connector", None)
        if gpu_connector is None:
            return
        if hasattr(gpu_connector, "set_timing_sink"):
            gpu_connector.set_timing_sink(sink)
        else:
            setattr(gpu_connector, "_timing_sink", sink)
