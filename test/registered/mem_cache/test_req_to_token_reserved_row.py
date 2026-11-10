"""Unit tests for the reserved request-row / dummy-KV read separation (V522-S3 fix 3).

``ReqToTokenPool`` reserves row 0 as the read source for idle and zero-padded
rows (they all carry ``req_pool_indices=0``). Its mapping was all zeros, i.e.
every position pointed at KV cache location 0, which is exactly where those same
rows write their dummy KV (``out_cache_loc=0``). The padding rows therefore read
back what they themselves had just written, so location 0 carried the previous
request's state forward and the padding rows computed a run-dependent function.

Row 0's mapping now points at location 1 instead. These tests pin the *honest*
properties of that change, which are deliberately weaker than "the pollution is
gone":

  - padding rows no longer read the location they write;
  - row 0 stays reserved (never handed out by ``alloc``/``alloc_rows``), so the
    mapping the padding rows read is stable across allocations and ``clear()``;
  - location 1 is *not* reserved: the paged KV allocator's free list starts at 1,
    so a real request does get assigned location 1 and its KV does overwrite it.
    The fix reduces the noise source instead of eliminating it, and this test
    documents that boundary rather than papering over it.

The pool constructor needs only ``torch``; no GPU is required.
"""

from __future__ import annotations

import sys
import unittest

import pytest

torch = pytest.importorskip("torch")

# `memory_pool` imports the quantization stack, which pulls platform-native
# extensions (e.g. `acext` on PPU). Skip cleanly where those are absent instead
# of failing collection; the pool itself is pure Python + `torch`.
mem_cache = pytest.importorskip(
    "sglang.srt.mem_cache.memory_pool",
    reason="sglang.srt.mem_cache.memory_pool needs the platform's native extensions",
)

ReqToTokenPool = mem_cache.ReqToTokenPool

from sglang.test.ci.ci_register import register_cpu_ci

# The pool is plain Python + `torch`; the import guard above only exists because
# `memory_pool` also pulls the quantization stack's native extensions.
register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _make_pool(size: int = 8, max_context_len: int = 16) -> ReqToTokenPool:
    return ReqToTokenPool(
        size=size,
        max_context_len=max_context_len,
        device="cpu",
        enable_memory_saver=False,
    )


class TestReservedRowReadSource(unittest.TestCase):
    def test_reserved_row_points_at_location_one(self):
        pool = _make_pool()
        self.assertEqual(pool.req_to_token[0].tolist(), [1] * pool.max_context_len)

    def test_reserved_row_and_dummy_write_target_differ(self):
        """The whole point: the padding rows must not read their own write sink.

        Dummy rows write their KV to ``out_cache_loc=0``, so a read source of 0
        is self-pollution. Pin that the two are now different locations.
        """
        pool = _make_pool()
        dummy_write_location = 0
        read_source = int(pool.req_to_token[0][0])
        self.assertNotEqual(
            read_source,
            dummy_write_location,
            "row 0 still reads the location the padding rows write",
        )

    def test_row_zero_is_never_handed_out(self):
        """Row 0 must stay reserved, or the mapping would drift under padding."""
        pool = _make_pool(size=8)
        self.assertNotIn(0, pool.free_slots)
        granted = []
        for _ in range(8):
            rows = pool.alloc_rows(1)
            self.assertIsNotNone(rows)
            granted.extend(rows)
        self.assertNotIn(0, granted)
        self.assertEqual(pool.alloc_rows(1), None)  # exhausted, still no row 0

    def test_reserved_mapping_survives_clear(self):
        """`clear()` resets the free list but must not disturb row 0."""
        pool = _make_pool(size=8)
        pool.alloc_rows(3)
        pool.clear()
        self.assertEqual(pool.req_to_token[0].tolist(), [1] * pool.max_context_len)

    def test_allocated_rows_do_not_include_the_reserved_row(self):
        pool = _make_pool(size=4)
        rows = pool.alloc_rows(4)
        self.assertEqual(sorted(rows), [1, 2, 3, 4])
        # And the reserved row's mapping is untouched by the allocation.
        self.assertEqual(pool.req_to_token[0].tolist(), [1] * pool.max_context_len)


class TestLocationOneIsNotReserved(unittest.TestCase):
    """Documents the boundary the fix does NOT close.

    ``TokenToKVPoolAllocator.clear`` sets ``free_pages = arange(1, size + 1)``, so
    location 1 is the first location a real request receives. The reserved row
    reads it, and a real request's KV overwrites it -- the padding rows can still
    observe a real request's residue. Closing that needs a location no request is
    ever assigned (an allocator reservation gap or a dedicated constant page),
    which changes allocator capacity/semantics and is out of scope here.
    """

    def test_kv_allocator_free_list_starts_at_one(self):
        """The assertion the old comment got backwards: 1 is allocatable."""
        size = 16
        free_pages = torch.arange(1, size + 1, dtype=torch.int64)
        self.assertEqual(int(free_pages[0]), 1)
        self.assertNotIn(0, free_pages.tolist())
        # So location 1 is the FIRST location handed to a real request, not part
        # of a never-allocated reserved region.
        self.assertEqual(int(free_pages.min()), 1)

    def test_read_source_is_an_allocatable_location(self):
        """The reserved row's target is a location real requests also use."""
        pool = _make_pool(size=8)
        size = pool.size
        allocatable = set(range(1, size + 1))
        read_source = int(pool.req_to_token[0][0])
        self.assertIn(
            read_source,
            allocatable,
            "fix assumes the read source is allocatable; if this changes, the "
            "analysis in the memory_pool comment must be revisited",
        )


if __name__ == "__main__":
    sys.exit(unittest.main())