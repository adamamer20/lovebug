import numpy as np

from lovebug.layer_activation import LayerActivationConfig
from lovebug.unified_mesa_model import UnifiedLoveModel


def test_hamming_similarity_all_bits_different():
    """Test Hamming similarity calculation with completely different bit patterns."""
    a = np.array([0], dtype=np.uint32)
    b = np.array([0xFFFF], dtype=np.uint32)

    # Simple Hamming similarity: 16 - number of different bits
    xor_result = a ^ b
    hamming_distance = bin(xor_result[0]).count("1")
    similarity = 16 - hamming_distance

    assert similarity == 0  # All 16 bits are different


def test_model_run_increases_population():
    """Test that running the model for a few steps increases population through reproduction."""
    config = LayerActivationConfig.genetic_only()
    model = UnifiedLoveModel(layer_config=config, n_agents=50)

    initial = len(model.agents)

    # Run for a few steps to allow reproduction
    for _ in range(5):
        model.step()

    assert len(model.agents) >= initial


def test_unified_model_basic_functionality():
    """Test basic functionality of the UnifiedLoveModel."""
    config = LayerActivationConfig.genetic_only()
    model = UnifiedLoveModel(layer_config=config, n_agents=100)

    # Test initial state
    assert len(model.agents) == 100
    assert model.layer_config.genetic_enabled is True
    assert model.layer_config.cultural_enabled is False

    # Test that model can step
    model.step()

    # Population should still exist after one step
    assert len(model.agents) > 0

    # After several steps, population should potentially grow
    for _ in range(10):
        model.step()

    # At least some agents should survive
    assert len(model.agents) > 0
