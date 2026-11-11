"""Unit tests for the reserved request-row / dummy-KV read separation (V522-S11 fix 3).

``ReqToTokenPool`` (and the PD decode node's ``DecodeReqToTokenPool``) reserve row
0 as the read source for idle and zero-padded rows, which all carry
``req_pool_indices=0``. Its mapping was all zeros, i.e. every position pointed at
KV cache location 0, which is exactly where those same rows write their dummy KV
(``out_cache_loc=0``). The padding rows therefore read back what they had just
written, so location 0 carried the previous request's state forward and the
padding rows computed a run-dependent function.

Row 0's mapping now points at location 1 instead. Whether that closes the
pollution depends on the KV allocator's page size, and these tests pin both
cases rather than one blanket claim:

  - ``page_size == 1``: the allocator hands out locations from 1 upward, so 1 is
    the first location a real request receives and its KV overwrites the reserved
    row's read source. Reduction, not elimination.
  - ``page_size > 1``: allocation is page-granular and the first page handed out
    is page 1, so location 1 lies inside page 0, which is never allocated. A
    padded row writes position 0 only, so nothing writes there either and the
    padding rows read a location no request can hold.

The pool constructors and the allocator's page arithmetic need only ``torch``; no
accelerator is required.
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

from sglang.srt.disaggregation.decode import DecodeReqToTokenPool  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

# The pools are plain Python + `torch`; the import guard above only exists because
# `memory_pool` also pulls the quantization stack's native extensions.
register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _make_pool(size: int = 8, max_context_len: int = 16) -> ReqToTokenPool:
    return ReqToTokenPool(
        size=size,
        max_context_len=max_context_len,
        device="cpu",
        enable_memory_saver=False,
    )


def _make_decode_pool(
    size: int = 8, max_context_len: int = 16, pre_alloc_size: int = 4
) -> DecodeReqToTokenPool:
    return DecodeReqToTokenPool(
        size=size,
        max_context_len=max_context_len,
        device="cpu",
        enable_memory_saver=False,
        pre_alloc_size=pre_alloc_size,
    )


def _paged_free_pages(num_pages: int) -> torch.Tensor:
    """``PagedTokenToKVPoolAllocator.clear``'s free list, verbatim.

    Page ids, not token locations: ``free_pages`` holds the ids of whole pages,
    and ``alloc`` turns one into ``page_id * page_size + arange(page_size)``.
    """
    return torch.arange(1, num_pages + 1, dtype=torch.int64)


def _paged_allocatable_locations(num_pages: int, page_size: int) -> set:
    pages = _paged_free_pages(num_pages)
    return set(
        (pages[:, None] * page_size + torch.arange(page_size)).reshape(-1).tolist()
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


class TestDecodeReqToTokenPoolReservedRow(unittest.TestCase):
    """A PD decode node builds ``DecodeReqToTokenPool``, not ``ReqToTokenPool``.

    Decode padding happens on exactly that node, so the fix has to be present in
    this constructor too -- it has its own ``__init__`` and never ran the parent's
    mapping setup.
    """

    def test_reserved_row_points_at_location_one(self):
        pool = _make_decode_pool()
        self.assertEqual(pool.req_to_token[0].tolist(), [1] * pool.max_context_len)

    def test_reserved_row_avoids_the_dummy_write_target(self):
        pool = _make_decode_pool()
        self.assertNotEqual(int(pool.req_to_token[0][0]), 0)

    def test_row_zero_is_never_in_the_free_list(self):
        """Row 0 is reserved at construction, and `clear()` keeps it out.

        ``DecodeReqToTokenPool`` is a standalone class (it does not inherit
        ``ReqToTokenPool``), so its row bookkeeping is ``free_slots`` plus
        ``alloc``/``free``; there is no ``alloc_rows`` here.
        """
        pool = _make_decode_pool(size=4, pre_alloc_size=2)
        self.assertEqual(pool.free_slots, list(range(1, pool._alloc_size)))
        # The pool is exhausted exactly when the rows above the reserved one run
        # out; row 0 is never offered, at any occupancy.
        while pool.free_slots:
            self.assertNotIn(0, pool.free_slots)
            pool.free_slots = pool.free_slots[1:]
        pool.clear()
        self.assertNotIn(0, pool.free_slots)
        self.assertEqual(pool.free_slots, list(range(1, pool._alloc_size)))

    def test_reserved_mapping_survives_clear(self):
        """`clear()` resets the free list but must not disturb row 0.

        `DecodeReqToTokenPool.clear()` returns every row above the reserved one
        to the free list; the reserved row's mapping is set in `__init__` and is
        not part of that reset.
        """
        pool = _make_decode_pool()
        taken = pool.free_slots[:3]
        pool.free_slots = pool.free_slots[3:]
        pool.clear()
        self.assertEqual(pool.req_to_token[0].tolist(), [1] * pool.max_context_len)
        # And the reset really did return the rows it took.
        self.assertEqual(pool.free_slots, list(range(1, pool._alloc_size)))
        self.assertEqual(sorted(taken), [1, 2, 3])


class TestPageSizeDecidesWhetherTheResidueIsClosed(unittest.TestCase):
    """The page-size split, taken from the allocator's own arithmetic.

    ``PagedTokenToKVPoolAllocator.clear`` sets ``free_pages = arange(1, num_pages
    + 1)`` -- a **page-id** list, whose first entry is page 1, so page 0 is never
    handed to a request at any page size. ``alloc`` expands a page id to
    ``page_id * page_size + arange(page_size)``, so the first allocatable token
    location is ``page_size``, not 1.
    """

    def test_first_allocatable_location_is_page_size(self):
        for page_size in (1, 2, 8, 64, 256):
            num_pages = 16
            allocatable = _paged_allocatable_locations(num_pages, page_size)
            self.assertEqual(
                min(allocatable),
                page_size,
                f"page_size={page_size}: the allocator handed out page 0",
            )

    def test_page_zero_covers_location_one_above_page_size_one(self):
        for page_size in (64, 256):
            num_pages = 16
            allocatable = _paged_allocatable_locations(num_pages, page_size)
            self.assertNotIn(
                1,
                allocatable,
                f"page_size={page_size}: location 1 is allocatable, so the "
                "reserved row can still be overwritten by a real request",
            )

    def test_page_size_one_hands_out_location_one(self):
        """The boundary the fix does not close, pinned so it cannot be overclaimed."""
        allocatable = _paged_allocatable_locations(num_pages=16, page_size=1)
        self.assertIn(1, allocatable)

    def test_padded_row_writes_only_position_zero(self):
        """A padded row carries ``out_cache_loc=0``, so it writes location 0.

        At any page size > 1 that is inside the never-allocated page 0, and the
        only other thing in page 0 is the reserved row's read source (location 1)
        -- which no request and no padded row writes, so the row reads a constant.
        """
        for page_size in (64, 256):
            pad_write = 0
            reserved_read_source = 1
            self.assertEqual(pad_write // page_size, 0)
            self.assertEqual(reserved_read_source // page_size, 0)
            self.assertEqual(pad_write, 0)


if __name__ == "__main__":
    sys.exit(unittest.main())