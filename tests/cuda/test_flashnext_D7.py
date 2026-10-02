"""W5-8 guard: TF_DRAFT_VOCAB picks the MTP drafts' scored ids; unset, the shipped list is unchanged (CPU only)."""
import hashlib
from pathlib import Path

import numpy as np

from tensorfold.families.qwen4_exp.cuda import weight_types as WT

HERE = Path(WT.__file__).parent
FREQ = HERE / "draft_vocab_freq40k.txt"


def test_unset_keeps_the_shipped_list(monkeypatch):
    monkeypatch.delenv("TF_DRAFT_VOCAB", raising=False)
    ids = WT.draft_token_ids("default")
    assert len(ids) == 79591 and ids[0] == 0
    assert WT.draft_token_ids(None) is None


def test_env_file_by_name_and_path(monkeypatch):
    for value in ("draft_vocab_freq40k.txt", str(FREQ)):
        monkeypatch.setenv("TF_DRAFT_VOCAB", value)
        ids = WT.draft_token_ids("default")
        assert len(ids) == 40960
        assert np.all(np.diff(ids) > 0) and ids[-1] < 248320
        assert np.isin(np.arange(1024), ids).all()                 # bytes and low ids always draftable
        assert np.isin(np.arange(248044, 248077), ids).all()       # chat/think/tool special tokens
    assert WT.draft_token_ids(None) is None                        # no drafts stays no drafts
    assert WT.draft_token_ids(1000).tolist() == list(range(1000))  # an explicit argument wins over the env


def test_env_integer(monkeypatch):
    monkeypatch.setenv("TF_DRAFT_VOCAB", "32768")
    assert WT.draft_token_ids("default").tolist() == list(range(32768))


def test_list_is_pinned():
    digest = hashlib.sha256(FREQ.read_bytes()).hexdigest()
    assert digest == "873e59010820175afe37ca15678dacd45fd05a1691d3e2ac6abd05c35d68eb19"
