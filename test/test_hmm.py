import pytest
from hmm import HiddenMarkovModel
import numpy as np


def _load_hmm(path):
    """Helper to load an HMM from an .npz file and return a HiddenMarkovModel instance."""
    data = np.load(path)
    return HiddenMarkovModel(
        observation_states=data['observation_states'],
        hidden_states=data['hidden_states'],
        prior_p=data['prior_p'],
        transition_p=data['transition_p'],
        emission_p=data['emission_p'],
    )


def test_mini_weather():
    """
    Create an instance of HMM using the mini_weather_hmm.npz file.
    Run Forward and Viterbi on the observation sequence from mini_weather_sequences.npz.
    Validate both outputs.
    """
    model = _load_hmm('./data/mini_weather_hmm.npz')
    sequences = np.load('./data/mini_weather_sequences.npz')

    obs_seq = sequences['observation_state_sequence']
    expected_hidden = list(sequences['best_hidden_state_sequence'])

    # Test Forward algorithm: probability should be > 0 and <= 1
    forward_prob = model.forward(obs_seq)
    assert forward_prob > 0, "Forward probability should be positive"
    assert forward_prob <= 1, "Forward probability should be at most 1"

    # Test Viterbi algorithm: decoded sequence should match expected
    decoded = model.viterbi(obs_seq)
    assert len(decoded) == len(expected_hidden), \
        f"Viterbi output length {len(decoded)} != expected {len(expected_hidden)}"
    assert decoded == expected_hidden, \
        f"Viterbi decoded {decoded} != expected {expected_hidden}"


def test_full_weather():
    """
    Create an instance of HMM using the full_weather_hmm.npz file.
    Run Forward and Viterbi on the observation sequence from full_weather_sequences.npz.
    Validate both outputs.
    """
    model = _load_hmm('./data/full_weather_hmm.npz')
    sequences = np.load('./data/full_weather_sequences.npz')

    obs_seq = sequences['observation_state_sequence']
    expected_hidden = list(sequences['best_hidden_state_sequence'])

    # Test Forward algorithm
    forward_prob = model.forward(obs_seq)
    assert forward_prob > 0, "Forward probability should be positive"
    assert forward_prob <= 1, "Forward probability should be at most 1"

    # Test Viterbi algorithm
    decoded = model.viterbi(obs_seq)
    assert len(decoded) == len(expected_hidden), \
        f"Viterbi output length {len(decoded)} != expected {len(expected_hidden)}"
    assert decoded == expected_hidden, \
        f"Viterbi decoded {decoded} != expected {expected_hidden}"


def test_edge_case_empty_sequence():
    """
    Edge case: empty observation sequence should return 0.0 for forward
    and an empty list for viterbi.
    """
    model = _load_hmm('./data/mini_weather_hmm.npz')
    empty_seq = np.array([])

    assert model.forward(empty_seq) == 0.0
    assert model.viterbi(empty_seq) == []


def test_edge_case_invalid_observation():
    """
    Edge case: observation not in the model's observation states should raise ValueError.
    """
    model = _load_hmm('./data/mini_weather_hmm.npz')
    bad_seq = np.array(['snowy'])  # 'snowy' not in mini model

    with pytest.raises(ValueError):
        model.forward(bad_seq)

    with pytest.raises(ValueError):
        model.viterbi(bad_seq)


def test_edge_case_single_observation():
    """
    Edge case: single observation should still produce a valid result
    driven only by prior and emission probabilities.
    """
    model = _load_hmm('./data/mini_weather_hmm.npz')
    single_seq = np.array(['sunny'])

    # Forward probability = sum_j prior(j) * emission(j, sunny)
    # = 0.6*0.65 + 0.4*0.15 = 0.39 + 0.06 = 0.45
    forward_prob = model.forward(single_seq)
    assert np.isclose(forward_prob, 0.45), f"Expected ~0.45, got {forward_prob}"

    # Viterbi: hot gives 0.6*0.65=0.39, cold gives 0.4*0.15=0.06 → hot wins
    decoded = model.viterbi(single_seq)
    assert len(decoded) == 1
    assert decoded == ['hot']













