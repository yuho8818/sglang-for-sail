"""Attention-TP row sharding for the DSV4.1 low-ratio indexer decode/verify
path (DSV41-S5.9 port).

The V4.1 low-ratio indexer's projections are replicated across the
attention-TP group (``DeepseekV41Indexer.wq_b`` / ``weights_proj`` are
``ReplicatedLinear``), so under TP8 every rank scores every verify row
identically and every rank selects the same top-k. This module splits that
redundant work: each rank scores only its contiguous share of the query rows
and the selections are all-gathered back to full width.

Contract kept with the replicated path:

* **The index-K side stays replicated.** The compressor (index-K projection
  and cache writes) runs on the full rows on every rank before the indexer
  (see ``forward_low_ratio_sources``); this module never touches it, and
  every rank keeps reading the full index-K table it already holds.
* **A dense score depends only on its own query row.** The torch scoring
  (``decode_scores``: queries / head_weights / slot gather / matmul) and the
  top-k / sort / write steps are all row-independent, so a gathered
  selection is bitwise the replicated one.
* **Either every rank reaches the all-gather or none does.** The shard
  decision depends only on the env switch, the attention-TP size and the
  row count, all identical across the group; the early returns taken before
  any collective (empty batch, ``lmax == 0``) depend on shared metadata only.

The split path is CUDA-graph compatible: every shape is static at capture
time, and the all-gather rides the same collective mechanism the decode
graph already captures for its all-reduces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_ppu

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


@dataclass(frozen=True)
class IndexerRowShard:
    """This rank's contiguous share of the decode/verify query rows across the
    attention-TP group, whose ranks all hold the same rows."""

    group: "GroupCoordinator"
    rank: int
    world: int

    def rows_per_rank(self, num_rows: int) -> int:
        return -(-num_rows // self.world)

    def local_rows(self, num_rows: int) -> slice:
        per = self.rows_per_rank(num_rows)
        start = min(self.rank * per, num_rows)
        return slice(start, min(start + per, num_rows))

    def gather(self, local: torch.Tensor, num_rows: int) -> torch.Tensor:
        """``[num_rows, ...]`` from every rank's ``[rows_per_rank, ...]`` share;
        the last share's padding rows fall past ``num_rows`` and are dropped."""
        gathered = local.new_empty((self.world * local.shape[0], *local.shape[1:]))
        self.group.all_gather_into_tensor(
            gathered.view(-1), local.contiguous().view(-1)
        )
        return gathered[:num_rows]


def maybe_indexer_row_shard(num_rows: int) -> Optional[IndexerRowShard]:
    """The shard to use for this decode/verify indexer call, if any.

    Every input of the decision is identical on all attention-TP ranks, so
    the group either all shards or all replicates."""
    if not envs.SGLANG_OPT_DSV4_INDEXER_TP_SHARD.get():
        return None
    if not is_ppu():
        return None
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if parallel.attn_tp_size <= 1:
        return None
    if num_rows < envs.SGLANG_OPT_DSV4_INDEXER_TP_SHARD_MIN_ROWS.get():
        return None
    return IndexerRowShard(
        group=parallel.attn_tp_group,
        rank=parallel.attn_tp_rank,
        world=parallel.attn_tp_size,
    )
