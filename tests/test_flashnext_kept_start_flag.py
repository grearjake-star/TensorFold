"""Only the end of a shared system block is kept as a message-start state.

A drafting prompt keeps states at its message starts (the second message's start, the last assistant start), at
the system-block checkpoint and one token before its end. Eviction keeps message-start states longest, resumes move
them to the newest and a fresh request takes their slot last: that protection is for the system block that many
conversations share. The last assistant start is one conversation's own state, like its end; marked as a start it
made every conversation's slot look like the system block's, so a fresh request could take the system block's slot.
Slot and eviction choices only: no kernel or kept state changes.
"""

from types import SimpleNamespace

from tensorfold.cuda.markers import MIN_GAP, snapshot_points
from tensorfold.families.qwen4_exp.cuda import prefixes
from tensorfold.families.qwen4_exp.cuda.multi_fill import PromptPasses
from tests.test_flashnext_prefix_victim import Owner, Slot, entry

OPEN, ASSISTANT, SYSTEM_ROLE, USER_ROLE = 1, 2, 3, 4


def message(role, n, seed):
    return [OPEN, role] + [100 + (seed * 7 + i) % 900 for i in range(n)]


def second_turn():
    """System block, user, assistant reply, user, and the generation prompt: four kept points."""

    return (message(SYSTEM_ROLE, 600, 1) + message(USER_ROLE, 300, 2) + message(ASSISTANT, 300, 3)
            + message(USER_ROLE, 300, 4) + [OPEN, ASSISTANT, 5, 6, 7])


class Fill(PromptPasses):
    """The fill step's bookkeeping around one prompt piece that ends at a kept point (no forward, no sampling)."""

    def __init__(self, points):
        self.points, self.flags = points, []
        self.w = SimpleNamespace(comm=None)

    def _remember(self, ids, st, snap, tail, start=False, checkpoint=False):
        self.flags.append((len(ids), start, checkpoint))

    def _joined_ranks(self):
        pass

    def keep(self, s, pos):
        self.fills = {s.sid: [None, True, 0, ({"pos": pos}, None)]}
        self._joined([(s, 0, pos)], None, [None], 0.0)
        return self.flags[-1]


def stream(prompt):
    return SimpleNamespace(prompt=prompt, sid=1, prefill_s=0.0, draft=True, st=SimpleNamespace(image_positions=None))


def test_the_kept_points_of_a_second_turn():
    prompt = second_turn()
    pts = snapshot_points([OPEN], [OPEN, ASSISTANT], checkpoint=0)
    found = pts(prompt)
    block_end, last_assistant = 602, len(prompt) - 5
    assert found == [block_end, last_assistant] and last_assistant - block_end >= MIN_GAP
    assert pts.block_end(prompt) == block_end


def test_only_the_system_blocks_end_is_a_start():
    prompt = second_turn()
    fill = Fill(snapshot_points([OPEN], [OPEN, ASSISTANT], checkpoint=0))
    s = stream(prompt)
    assert fill.keep(s, 602) == (602, True, False)                       # the shared system block
    assert fill.keep(s, len(prompt) - 5) == (len(prompt) - 5, False, False)  # this conversation's last assistant start


def test_the_checkpoint_is_still_flagged():
    prompt = message(SYSTEM_ROLE, 2400, 1) + message(USER_ROLE, 300, 2) + [OPEN, ASSISTANT, 5]
    pts = snapshot_points([OPEN], [OPEN, ASSISTANT], checkpoint=1024)
    cp = pts.checkpoint(prompt)
    assert cp == 2048
    fill = Fill(pts)
    n, _, checkpoint = fill.keep(stream(prompt), cp)
    assert n == cp and checkpoint                    # remember(checkpoint=True) also makes it a start


def test_a_fresh_request_takes_a_conversations_slot_not_the_system_blocks():
    """Why the flag matters: slot a keeps the system block (and its first conversation), slot b a second conversation
    with its last assistant start. Marked as a start, b's entry held b, every slot looked like the system block's
    and the oldest (a) went."""

    a, b = Slot("a"), Slot("b")
    system = list(range(300))
    one, two = system + list(range(1000, 1400)), system + list(range(2000, 2600))
    asst_two = two[:500]

    def owner(marked):
        o = Owner([entry(system, a, "system"), entry(one, a, "one"), entry(asst_two, b, "asst-two"),
                   entry(two, b, "two")])
        o.starts = {tuple(system)} | ({tuple(asst_two)} if marked else set())
        return o

    st, _, _ = prefixes.slot_for(owner(marked=True), list(range(9000, 9100)), True)
    assert st is a                                         # the old flags: the system block's slot went
    o = owner(marked=False)
    st, _, _ = prefixes.slot_for(o, list(range(9000, 9100)), True)
    assert st is b and {k[3] for k in o.kept} == {"system", "one"}
