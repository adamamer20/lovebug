import numpy as np
import polars as pl

from lovebug import model as love_model
from lovebug.model import LoveModel, get_effective_pref, hamming_similarity


def test_hamming_similarity_all_bits_different():
    a = np.array([0], dtype=np.uint32)
    b = np.array([0xFFFF], dtype=np.uint32)
    sim = hamming_similarity(a, b)
    assert sim[0] == 0


def test_model_run_increases_population():
    model = LoveModel(50)
    initial = len(model.agents)
    model.run(5)
    assert len(model.agents) >= initial


def test_get_effective_pref_respects_weight(monkeypatch):
    df = pl.DataFrame(
        {
            "genome": np.array([0x12340000], dtype=np.uint32),
            "pref_culture": np.array([0xAB], dtype=np.uint8),
        }
    )

    monkeypatch.setattr(love_model, "CULTURE_WEIGHT", 1.0)
    eff = get_effective_pref(df)
    assert eff[0] == 0xAB

    monkeypatch.setattr(love_model, "CULTURE_WEIGHT", 0.0)
    eff = get_effective_pref(df)
    gene = (df["genome"].to_numpy() & love_model.PREF_MASK) >> love_model.PREF_SHIFT
    assert eff[0] == gene[0]
