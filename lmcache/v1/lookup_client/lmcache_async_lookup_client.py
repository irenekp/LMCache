# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union
import threading
import time

# Third Party
from vllm.utils import make_zmq_socket
import msgspec
import torch
import zmq

# First Party
from lmcache.integration.vllm.utils import create_lmcache_metadata, mla_enabled
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _normalize_tier_segments(raw: object) -> list[tuple[str, int]]:
    if not isinstance(raw, list):
        return []
    out: list[tuple[str, int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        backend_name, token_count = item
        if not isinstance(backend_name, str):
            continue
        if not isinstance(token_count, int) or token_count <= 0:
            continue
        out.append((backend_name, int(token_count)))
    return out


def _truncate_tier_segments(
    segments: list[tuple[str, int]],
    limit_tokens: int,
) -> list[tuple[str, int]]:
    if limit_tokens <= 0:
        return []
    out: list[tuple[str, int]] = []
    remaining = int(limit_tokens)
    for backend_name, token_count in segments:
        if remaining <= 0:
            break
        take = min(int(token_count), remaining)
        if take <= 0:
            continue
        if out and out[-1][0] == backend_name:
            last_backend, last_tokens = out[-1]
            out[-1] = (last_backend, last_tokens + take)
        else:
            out.append((backend_name, take))
        remaining -= take
    return out


# NOTE(Jiayi): Prefetch could load extra redundant cache if multiple
# workers has different hit tokens.
class LMCacheAsyncLookupClient(LookupClientInterface):
    """
    ZMQ-based lookup client that communicates with a lookup server.

    Related extra_config:
    - create_lookup_server_only_on_worker_0_for_mla:
        is a flag to control whether to create lookup server only on worker 0.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
    ):
        metadata, config = create_lmcache_metadata(vllm_config)

        self.encoder = msgspec.msgpack.Encoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0
        )
        self.tensor_parallel_size = vllm_config.parallel_config.tensor_parallel_size
        use_mla = mla_enabled(vllm_config.model_config)
        self.create_lookup_server_only_on_worker_0_for_mla = (
            config.get_extra_config_value(
                "create_lookup_server_only_on_worker_0_for_mla", use_mla
            )
        )
        ranks = self.tensor_parallel_size
        self.push_sockets = []
        if self.create_lookup_server_only_on_worker_0_for_mla:
            ranks = 1
        for tp_rank in range(ranks):
            worker_socket_path = get_zmq_rpc_path_lmcache(
                vllm_config, "lookup_worker", rpc_port, tp_rank
            )
            logger.info(
                f"lmcache lookup client connect to tp_rank {tp_rank} "
                f"with worker socket path {worker_socket_path}"
            )

            push_socket = make_zmq_socket(
                self.ctx,
                worker_socket_path,
                zmq.PUSH,  # type: ignore[attr-defined]
                bind=False,
            )

            self.push_sockets.append(push_socket)

        scheduler_socket_path = get_zmq_rpc_path_lmcache(
            vllm_config, "lookup_scheduler", rpc_port, 0
        )
        self.pull_socket = make_zmq_socket(
            self.ctx,
            scheduler_socket_path,
            zmq.PULL,  # type: ignore[attr-defined]
            bind=True,
        )
        logger.info(
            f"lmcache lookup client connect to scheduler "
            f"with socket path {scheduler_socket_path}"
        )

        # First Party
        from lmcache.v1.token_database import (
            ChunkedTokenDatabase,
            SegmentTokenDatabase,
            TokenDatabase,
        )

        self.token_database: TokenDatabase
        if config.enable_blending:
            self.token_database = SegmentTokenDatabase(config, metadata)
        else:
            self.token_database = ChunkedTokenDatabase(config, metadata)

        # A lock is needed since we need another thread to pull
        # responses from the lookup_and_prefetch server
        # (e.g., worker process).
        self.lock = threading.Lock()

        # map from lookup_id to req's status.
        # None indicates ongoing.
        # int indicates number of hit tokens.
        self.reqs_status: dict[str, Optional[int]] = {}
        self.reqs_tier_stats: dict[str, dict[str, int]] = {}
        self.reqs_tier_segments: dict[str, list[tuple[str, int]]] = {}

        # map from lookup_id to number of hit tokens for each worker
        self.res_for_each_worker: dict[str, list[int]] = {}
        self.tier_for_each_worker: dict[str, list[dict[str, int]]] = {}
        self.segments_for_each_worker: dict[str, list[list[tuple[str, int]]]] = {}

        # The required parts are [lookup_id, num_hit_tokens], with optional
        # per-tier aggregates and ordered tier segments appended after that.
        self.num_parts = 2

        self.running = True

        self.thread = threading.Thread(
            target=self.process_responses_from_workers, daemon=True
        )
        self.thread.start()

        # default backoff time
        self.lookup_backoff_time = 0.01
        if config.extra_config is not None:
            self.lookup_backoff_time = float(
                config.extra_config.get("lookup_backoff_time", self.lookup_backoff_time)
            )

    # TODO(Jiayi): Consider batching here
    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        with self.lock:
            # -1 indicates not found; None indicates ongoing.
            req_status = self.reqs_status.get(lookup_id, -1)
            if req_status is None:
                time.sleep(self.lookup_backoff_time)
                return None
            elif req_status != -1:
                self.reqs_status.pop(lookup_id)
                return req_status
            self.reqs_status[lookup_id] = None
        hashes = []
        offsets = []
        for start, end, hash_val in self.token_database.process_tokens(
            token_ids, make_key=False
        ):
            hashes.append(hash_val)
            offsets.append(end - start)
        hash_buf = self.encoder.encode(hashes)
        offset_buf = self.encoder.encode(offsets)

        lookup_id_buf = lookup_id.encode("utf-8")
        request_configs_str = ""
        if request_configs is not None and len(request_configs) != 0:
            request_configs_str = "@".join(
                [f"{k}%{v}" for k, v in request_configs.items()]
            )
        request_configs_buf = request_configs_str.encode("utf-8")

        msg_buf = [
            lookup_id_buf,
            hash_buf,
            offset_buf,
            request_configs_buf,
        ]

        ranks = self.tensor_parallel_size
        if self.create_lookup_server_only_on_worker_0_for_mla:
            ranks = 1
        for i in range(ranks):
            self.push_sockets[i].send_multipart(msg_buf, copy=False)
        time.sleep(self.lookup_backoff_time)
        return None

    def process_responses_from_workers(self):
        while self.running:
            frames = self.pull_socket.recv_multipart(copy=False)
            if len(frames) < self.num_parts:
                logger.warning("Malformed response received: %s frames", len(frames))
                continue
            lookup_id = frames[0].bytes.decode("utf-8")
            res = int.from_bytes(frames[1], "big")
            tier_stats: dict[str, int] = {}
            if len(frames) >= 3:
                try:
                    tier_stats = msgspec.msgpack.decode(
                        frames[2].bytes, type=dict[str, int]
                    )
                except Exception:
                    tier_stats = {}
            if len(frames) >= 4:
                try:
                    tier_segments = _normalize_tier_segments(
                        msgspec.msgpack.decode(frames[3].bytes)
                    )
                except Exception:
                    tier_segments = []
            else:
                tier_segments = []

            with self.lock:
                if lookup_id not in self.res_for_each_worker:
                    self.res_for_each_worker[lookup_id] = [res]
                else:
                    self.res_for_each_worker[lookup_id].append(res)
                all_res = self.res_for_each_worker[lookup_id]

                if lookup_id not in self.tier_for_each_worker:
                    self.tier_for_each_worker[lookup_id] = [tier_stats]
                else:
                    self.tier_for_each_worker[lookup_id].append(tier_stats)
                all_tiers = self.tier_for_each_worker[lookup_id]

                if lookup_id not in self.segments_for_each_worker:
                    self.segments_for_each_worker[lookup_id] = [tier_segments]
                else:
                    self.segments_for_each_worker[lookup_id].append(tier_segments)
                all_segments = self.segments_for_each_worker[lookup_id]

                expected = (
                    1
                    if self.create_lookup_server_only_on_worker_0_for_mla
                    else self.tensor_parallel_size
                )
                if len(all_res) == expected:
                    self.res_for_each_worker.pop(lookup_id, None)
                    self.tier_for_each_worker.pop(lookup_id, None)
                    self.segments_for_each_worker.pop(lookup_id, None)

                    # NOTE: it is possible that the number of hit
                    # tokens is different across TP ranks, so we
                    # can use the minimum value as the number of
                    # hit tokens.
                    min_hit_tokens = min(all_res)
                    self.reqs_status[lookup_id] = min_hit_tokens
                    min_by: dict[str, int] = {}
                    all_keys = set()
                    for d in all_tiers:
                        all_keys.update(d.keys())
                    for k in all_keys:
                        vals = [d.get(k, 0) for d in all_tiers]
                        min_by[k] = min(vals) if vals else 0
                    self.reqs_tier_stats[lookup_id] = min_by
                    min_idx = all_res.index(min_hit_tokens)
                    chosen_segments = (
                        all_segments[min_idx] if min_idx < len(all_segments) else []
                    )
                    self.reqs_tier_segments[lookup_id] = _truncate_tier_segments(
                        chosen_segments,
                        min_hit_tokens,
                    )

    def get_tier_stats(self, lookup_id: str) -> Optional[dict[str, int]]:
        with self.lock:
            return self.reqs_tier_stats.get(lookup_id)

    def get_tier_segments(
        self, lookup_id: str
    ) -> Optional[list[tuple[str, int]]]:
        with self.lock:
            return self.reqs_tier_segments.get(lookup_id)

    def clear_lookup_status(self, lookup_id: str) -> None:
        with self.lock:
            self.reqs_status.pop(lookup_id, None)
            self.reqs_tier_stats.pop(lookup_id, None)
            self.reqs_tier_segments.pop(lookup_id, None)
            self.res_for_each_worker.pop(lookup_id, None)
            self.tier_for_each_worker.pop(lookup_id, None)
            self.segments_for_each_worker.pop(lookup_id, None)

    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
            for s in self.push_sockets:
                s.close(linger=0)  # type: ignore[arg-type]
            self.pull_socket.close(linger=0)  # type: ignore[arg-type]
            self.ctx.term()
        except Exception as e:
            logger.warning(f"Failed to join thread during close: {e}")


class LMCacheAsyncLookupServer:
    """ZMQ-based async lookup server that handles lookup and prefetch
    requests using LMCacheEngine."""

    def __init__(self, lmcache_engine: LMCacheEngine, vllm_config: "VllmConfig"):
        self.decoder = msgspec.msgpack.Decoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0
        )
        worker_socket_path = get_zmq_rpc_path_lmcache(
            vllm_config, "lookup_worker", rpc_port, vllm_config.parallel_config.rank
        )
        scheduler_socket_path = get_zmq_rpc_path_lmcache(
            vllm_config, "lookup_scheduler", rpc_port, 0
        )
        self.push_socket = make_zmq_socket(
            self.ctx,
            scheduler_socket_path,
            zmq.PUSH,  # type: ignore[attr-defined]
            bind=False,
        )
        self.pull_socket = make_zmq_socket(
            self.ctx,
            worker_socket_path,
            zmq.PULL,  # type: ignore[attr-defined]
            bind=True,
        )

        self.lmcache_engine = lmcache_engine
        self.running = True

        logger.info(
            "lmcache lookup server start with"
            f" scheduler socket path {scheduler_socket_path}, "
            f"worker socket path {worker_socket_path}"
        )
        self.thread = threading.Thread(
            target=self.process_requests_from_scheduler, daemon=True
        )
        self.thread.start()

        # The four parts are [hash, offset, lookup_id, request_configs]
        self.num_parts = 4

    def process_requests_from_scheduler(self):
        while self.running:
            frames = self.pull_socket.recv_multipart(copy=False)
            num_frames = len(frames)
            assert num_frames % self.num_parts == 0
            for i in range(0, num_frames, self.num_parts):
                lookup_id = frames[i].bytes.decode("utf-8")

                hash_frame = frames[i + 1]
                hashes = self.decoder.decode(hash_frame)

                offset_frame = frames[i + 2]
                offsets = self.decoder.decode(offset_frame)

                request_configs_str = frames[i + 3].bytes.decode("utf-8")
                request_configs = None
                if request_configs_str != "":
                    request_configs = {}
                    request_configs_list = request_configs_str.split("@")
                    for kv in request_configs_list:
                        kvs = kv.split("%", 1)
                        if len(kvs) != 2:
                            raise ValueError(f"Unexpected tags_str: {kvs}")
                        request_configs[kvs[0]] = kvs[1]

                self.lmcache_engine.async_lookup_and_prefetch(
                    lookup_id=lookup_id,
                    hashes=hashes,
                    offsets=offsets,
                    pin=True,
                    request_configs=request_configs,
                )

    def send_response_to_scheduler(self, lookup_id: str, num_hit_tokens: int):
        lookup_id_buf = lookup_id.encode("utf-8")
        num_hit_tokens_buf = num_hit_tokens.to_bytes(4, "big")
        tier_stats = self.lmcache_engine.lookup_tier_hit_tokens.get(lookup_id, {})
        tier_stats_buf = msgspec.msgpack.encode(tier_stats)
        tier_segments = self.lmcache_engine.lookup_tier_hit_segments.get(lookup_id, [])
        tier_segments_buf = msgspec.msgpack.encode(tier_segments)
        self.push_socket.send_multipart(
            [lookup_id_buf, num_hit_tokens_buf, tier_stats_buf, tier_segments_buf],
            copy=False,
        )

    def close(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
            for s in self.push_sockets:
                s.close(linger=0)  # type: ignore[arg-type]
            self.pull_socket.close(linger=0)  # type: ignore[arg-type]
            self.ctx.term()
        except Exception as e:
            logger.warning(f"Failed to join thread during close: {e}")
