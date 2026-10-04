"""AUDIT (needs GPU): the per-slot lone-stream graph sets (multi_solo._graphs_to) stay bounded by the slot count, a
resized slot's set is gone (never replayed with stale cache pointers), replies keep their serial bits when the graphs
come back to a slot, and each kept set's private-pool memory is printed (the memory gate does not count it)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402
from test_flashnext_L7 import _ref, _run  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402


def _sets(dec):
    return {id(g): (st, g) for st, g in getattr(dec, "_slot_graphs", {}).values()}


@pytest.mark.parametrize("slots", [3, 4])
def test_slot_graph_sets_stay_bounded_fresh_and_exact(slots):
    w = _model()
    dec = MultiDecoder(w, slots=slots, capacity=1024, depth=3, confidence=0.3, graphs=True)
    smp = Sampling(seed=5, top_k=20, top_p=0.95)
    convs = [[(11 * i + 5 * j + 3) % (V - 1) + 1 for i in range(6 + j)] for j in range(2 * slots)]
    for p in convs:                                   # fill every slot with a kept end
        assert _run(dec, p, 8, smp).out == _ref(w, p, 8, smp)
    torch.cuda.synchronize()
    before = torch.cuda.memory_reserved()
    for turn in range(3):                             # conversations take turns: the graphs visit their slots
        for j in range(slots):
            p = convs[j]
            s = _run(dec, p, 8, smp)
            assert s.out == _ref(w, p, 8, smp), (turn, j)
            convs[j] = p + s.out[:-1] + [50 + turn]
            sets = _sets(dec)
            assert len(sets) <= slots                 # at most one set a slot
            for st, g in sets.values():
                assert g.e is dec.solo                # every set replays through the one solo engine's buffers
    torch.cuda.synchronize()
    sets = _sets(dec)
    print(f"[audit] {len(sets)} slot graph sets, reserved +{(torch.cuda.memory_reserved() - before) / 2**20:.1f} MiB")
    # a resize must drop that slot's set: grow an idle non-graph slot that holds one
    idle = next((st for st, g in sets.values() if st is not dec.solo.st and id(st) not in dec._busy()), None)
    if idle is not None:
        assert dec._grow(idle, idle.capacity + 1)
        assert id(idle) not in getattr(dec, "_slot_graphs", {})
    for j in range(slots):                            # and the replies still match after the resize
        p = convs[j]
        assert _run(dec, p, 8, smp).out == _ref(w, p, 8, smp)
