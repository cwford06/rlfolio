import numpy as np
import pytest

from rlfolio.agents import (
    AllocationMDP,
    DoubleQLearningAgent,
    DQNConfig,
    DuelingDQNAgent,
    DuelingNetwork,
    MDPConfig,
    PrioritizedReplay,
    QLearningConfig,
)
from rlfolio.sleeves import SleeveBook


def regime_book(n=400, seed=0) -> SleeveBook:
    """Two assets and a known answer: hold asset 0 when feature > 0, else asset 1."""
    rng = np.random.default_rng(seed)
    regime = rng.integers(2, size=n)
    signal = np.where(regime == 1, 1.0, -1.0)
    features = np.column_stack([signal * 2, rng.normal(size=n), signal * 2, rng.normal(size=n)])
    # Tabular agents read feature columns 1, 3 and 7, so pad the matrix to nine columns.
    features = np.column_stack([features[:, :1], features[:, :1], features[:, 1:2], features[:, 1:2],
                                rng.normal(size=(n, 3)), features[:, :1], rng.normal(size=n)])
    R = np.column_stack([np.where(regime == 1, 0.02, -0.02), np.where(regime == 1, -0.02, 0.02)])
    w0, w1, ew = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])
    W = np.stack([np.tile(x, (n, 1)) for x in (np.zeros(2), ew, w0, w1)], axis=1)
    sleeves = {"ew": np.tile(ew, (n, 1))}
    actions = ("cash", "ew@100", "a0@100", "a1@100")
    return SleeveBook(np.arange(n) * 5 + 300, features, sleeves, actions, W, R, ("A", "B"))


def correct_share(agent, book, n=300):
    good = {1.0: 2, -1.0: 3}
    hits = [agent.act(book.features[k], 1) == good[np.sign(book.features[k, 0])] for k in range(n)]
    return float(np.mean(hits))


def test_mdp_rewards_charge_turnover_and_use_baseline():
    book = regime_book()
    free = AllocationMDP(book, 100, MDPConfig(cost_bps=0))
    paid = AllocationMDP(book, 100, MDPConfig(cost_bps=50))
    assert np.allclose(free.rewards[5, :, 1], 0.0)  # baseline action earns zero advantage
    assert (paid.rewards[5, 0, 2] < free.rewards[5, 0, 2])  # trading from cash costs money
    assert paid.rewards.shape == (100, 4, 4)


def test_mdp_refuses_unrealized_returns():
    book = regime_book(50)
    book.R[40] = np.nan
    with pytest.raises(ValueError):
        AllocationMDP(book, 45)


def test_q_learning_recovers_the_known_policy():
    book = regime_book()
    agent = DoubleQLearningAgent(QLearningConfig(episodes=30)).fit(AllocationMDP(book, 350), seed=0)
    assert correct_share(agent, book) > 0.95


def test_classic_epsilon_greedy_q_learning_also_learns():
    book = regime_book()
    config = QLearningConfig(episodes=150, full_sweep=False)
    agent = DoubleQLearningAgent(config).fit(AllocationMDP(book, 350), seed=0)
    assert correct_share(agent, book) > 0.8


def test_dqn_recovers_the_known_policy():
    book = regime_book()
    config = DQNConfig(total_steps=3000, ensemble=1)
    agent = DuelingDQNAgent(config).fit(AllocationMDP(book, 350), seed=0)
    assert correct_share(agent, book) > 0.9


def test_network_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    net = DuelingNetwork(5, 3, 8, rng)
    x = rng.normal(size=(6, 5))
    actions, targets, weights = rng.integers(3, size=6), rng.normal(size=6), rng.uniform(0.5, 1, 6)
    _, _, grads = net.loss_and_grads(x, actions, targets, weights)
    eps = 1e-6
    for name in DuelingNetwork.NAMES:
        param = net.params[name]
        for index in list(np.ndindex(param.shape))[:6]:
            original = param[index]
            param[index] = original + eps
            up = net.loss_and_grads(x, actions, targets, weights)[0]
            param[index] = original - eps
            down = net.loss_and_grads(x, actions, targets, weights)[0]
            param[index] = original
            assert grads[name][index] == pytest.approx((up - down) / (2 * eps), abs=1e-6)


def test_prioritized_replay_favours_high_error_samples():
    rng = np.random.default_rng(0)
    replay = PrioritizedReplay(100)
    for i in range(100):
        replay.add(i, 0, 0)
    replay.update(np.arange(100), np.r_[np.full(99, 0.001), 50.0])
    index, _, weights = replay.sample(1000, 0.4, rng)
    assert (index == 99).mean() > 0.5
    assert weights.max() == pytest.approx(1.0)
