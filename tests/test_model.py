import numpy as np

from lovebug.model import LoveModel, hamming_similarity


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
