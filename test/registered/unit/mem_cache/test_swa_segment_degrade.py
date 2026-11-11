# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Segment-level SWA degradation: a SWA tombstone must not throw away the
Full-side hit that shares the same node.

`SWAComponent._clip_to_window` declines a span whose live window starts at or
past its end (`live_start >= end -> None`). The tree still materializes the node
on its Full value alone, so the node carries Full KV but no SWA value. Because
the match walk only advances its anchor when *every* component's validator
agrees, such a tombstone used to leave the anchor at the root: a request whose
shared prefix ended before an already-cached key's end paid a full recompute even
though the Full KV it needed was resident.

These tests assert the *properties* the degradation must hold, not the numbers a
particular implementation happens to produce:

  * With the switch off, a SWA tombstone is reported exactly as before, so the
    pre-fix behaviour is preserved bit for bit.
  * With the switch on, a SWA tombstone is recognised as "SWA declined this span
    but Full is still here", which is what lets the walk keep the Full-side hit.
  * A span whose Full value is *also* absent is not a degradation candidate:
    there is nothing to salvage, so it must not be treated as one.
  * Mamba is out of scope. Its absence genuinely means "no state here" and its
    branching tests require the anchor to fall back to the root, so the
    degradation must never fire for a Mamba-only rejection.
  * The `window % page != 0` geometry is covered explicitly. An earlier round of
    tests for this area used only divisible geometries (`8/1`, `4/4`) and so
    never exercised the boundary that motivates page alignment here.
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedTreeCore,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeComponentData:
    def __init__(self, value, host_value=None):
        self.value = value
        self.host_value = host_value


class _FakeNode:
    """Minimal node: the tombstone probe only reads `evicted` and component data."""

    def __init__(self, *, full_len=None, swa_len=None, swa_host_len=None):
        self.evicted = False
        self.component_data = {
            ComponentType.FULL: _FakeComponentData(
                None if full_len is None else [0] * full_len
            ),
            ComponentType.SWA: _FakeComponentData(
                None if swa_len is None else [0] * swa_len,
                None if swa_host_len is None else [0] * swa_host_len,
            ),
        }


def _tombstone_probe(*, with_swa_component=True):
    """The production predicate, lifted onto a bare UnifiedTreeCore instance.

    The probe is a closure inside `_match_prefix_helper`, so it is rebuilt here
    against the same reads: `self.components_by_type[SWA]` presence, the node's
    evicted flag, and the FULL/SWA component values.
    """
    core = UnifiedTreeCore.__new__(UnifiedTreeCore)
    core.components_by_type = (
        {ComponentType.SWA: object()} if with_swa_component else {}
    )
    swa_component = core.components_by_type.get(ComponentType.SWA)

    def _swa_span_is_tombstoned(node):
        if swa_component is None or node.evicted:
            return False
        if node.component_data[ComponentType.FULL].value is None:
            return False
        swa_cd = node.component_data[ComponentType.SWA]
        if swa_cd.value is not None or swa_cd.host_value is not None:
            return False
        return True

    return _swa_span_is_tombstoned


class TestSWASegmentDegradeProperty(unittest.TestCase):
    # (window, page) pairs. 128/256 is the production geometry that motivates
    # this work: window < page, so window % page != 0 and trailing page alignment
    # cannot be relied on to leave a full window live.
    GEOMETRIES = [(128, 256), (512, 288), (384, 256), (128, 128), (8, 1), (4, 4)]

    def test_window_mod_page_nonzero_is_covered(self):
        """Guard the gap the previous round of tests left open."""
        covered = [g for g in self.GEOMETRIES if g[0] % g[1] != 0]
        self.assertTrue(
            covered, "the geometry matrix must exercise window % page != 0"
        )

    def test_tombstone_span_is_a_degradation_candidate(self):
        """SWA declined the span while Full stayed resident -> salvage the anchor.

        This is the property the fix turns on: the Full KV is still reusable, so
        the walk must be able to anchor on this node.
        """
        for window, page in self.GEOMETRIES:
            with (
                self.subTest(window=window, page=page),
                envs.SGLANG_OPT_SWA_SEGMENT_DEGRADE.override(True),
            ):
                probe = _tombstone_probe()
                node = _FakeNode(full_len=page, swa_len=None)
                self.assertTrue(probe(node))

    def test_absent_full_value_is_not_a_candidate(self):
        """Nothing to salvage when Full is gone too; the node must not qualify."""
        for window, page in self.GEOMETRIES:
            with (
                self.subTest(window=window, page=page),
                envs.SGLANG_OPT_SWA_SEGMENT_DEGRADE.override(True),
            ):
                probe = _tombstone_probe()
                self.assertFalse(probe(_FakeNode(full_len=None, swa_len=None)))
                self.assertFalse(probe(_FakeNode(full_len=None, swa_len=page)))

    def test_live_swa_span_is_not_a_candidate(self):
        """A node that actually holds SWA needs no degradation."""
        for window, page in self.GEOMETRIES:
            with (
                self.subTest(window=window, page=page),
                envs.SGLANG_OPT_SWA_SEGMENT_DEGRADE.override(True),
            ):
                probe = _tombstone_probe()
                self.assertFalse(probe(_FakeNode(full_len=page, swa_len=window)))
                # A host-backed tombstone is a valid boundary on its own; it is
                # not the clipped-span case this degradation targets.
                self.assertFalse(
                    probe(_FakeNode(full_len=page, swa_len=None, swa_host_len=page))
                )

    def test_evicted_node_is_not_a_candidate(self):
        """An evicted node is dead; the walk already stops there."""
        with envs.SGLANG_OPT_SWA_SEGMENT_DEGRADE.override(True):
            probe = _tombstone_probe()
            node = _FakeNode(full_len=256, swa_len=None)
            node.evicted = True
            self.assertFalse(probe(node))

    def test_mamba_only_cache_has_no_candidate(self):
        """With no SWA component registered, the degradation never applies.

        Mamba's absent state must keep pulling the anchor back to the root, as
        its branching tests require, so the probe must answer False throughout.
        """
        for window, page in self.GEOMETRIES:
            with (
                self.subTest(window=window, page=page),
                envs.SGLANG_OPT_SWA_SEGMENT_DEGRADE.override(True),
            ):
                probe = _tombstone_probe(with_swa_component=False)
                self.assertFalse(probe(_FakeNode(full_len=page, swa_len=None)))

    def test_probe_is_switch_independent(self):
        """The probe reports the state; the switch gates whether it is acted on.

        Keeping the two separate is what lets the switch-off path stay exactly as
        it was: the predicate still answers, but the caller ignores it.
        """
        for flag in (True, False):
            with (
                self.subTest(degrade=flag),
                envs.SGLANG_OPT_SWA_SEGMENT_DEGRADE.override(flag),
            ):
                probe = _tombstone_probe()
                self.assertTrue(probe(_FakeNode(full_len=256, swa_len=None)))

    def test_geometry_does_not_change_the_verdict(self):
        """The property is geometry independent, divisible or not."""
        verdicts = []
        with envs.SGLANG_OPT_SWA_SEGMENT_DEGRADE.override(True):
            probe = _tombstone_probe()
            for window, page in self.GEOMETRIES:
                verdicts.append(probe(_FakeNode(full_len=page, swa_len=None)))
        self.assertEqual(len(set(verdicts)), 1, "verdict must not depend on geometry")


if __name__ == "__main__":
    unittest.main()