"""The allocation MDP and two value-based agents: Double Q-learning and Dueling DQN.

Both agents exploit one structural fact: a small trader's actions do not move the
market. Returns are exogenous, so at any historical step the reward of *every*
(previous action, action) pair can be computed exactly. That counterfactual data
multiplies the effective sample size without touching the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .sleeves import TABULAR_FEATURES, SleeveBook


@dataclass(frozen=True)
class MDPConfig:
    cost_bps: float = 8.0  # one-way cost charged during training
    risk_aversion: float = 5.0
    gamma: float = 0.9
    reward_scale: float = 100.0


class AllocationMDP:
    """Decision steps ``[0, k_end)`` of a book, with rewards tabulated in advance.

    State is (regime features at k, previous action). ``rewards[k, prev, a]`` is the
    risk-adjusted growth of action ``a`` net of the cost of trading out of the drifted
    ``prev`` portfolio, minus the utility of the equal-weight baseline. The baseline
    does not depend on the action, so subtracting it leaves the optimal policy
    unchanged while removing most of the market noise from the learning target.
    """

    def __init__(self, book: SleeveBook, k_end: int, config: MDPConfig = MDPConfig()):
        if k_end < 2:
            raise ValueError("training window contains fewer than two decisions")
        self.config = config
        self.n_steps = k_end
        self.n_actions = book.n_actions
        self.baseline_action = book.baseline_action
        self.features = book.features[: k_end + 1]

        W, R = book.W[:k_end], book.R[:k_end]
        if not np.isfinite(R).all():
            raise ValueError("training window includes unrealized returns")
        gross = np.einsum("kan,kn->ka", W, R)
        drifted = W * (1 + R[:, None, :]) / (1 + gross)[:, :, None]
        held = np.concatenate([np.zeros_like(drifted[:1]), drifted[:-1]])
        turnover = np.abs(W[:, None, :, :] - held[:, :, None, :]).sum(axis=-1)
        net = gross[:, None, :] - config.cost_bps / 1e4 * turnover

        def utility(p: np.ndarray) -> np.ndarray:
            return np.log(np.maximum(1 + p, 1e-4)) - 0.5 * config.risk_aversion * p**2

        baseline = utility(gross[:, self.baseline_action])[:, None, None]
        self.rewards = config.reward_scale * (utility(net) - baseline)  # (k, prev, a)


class Agent(Protocol):
    name: str

    def fit(self, mdp: AllocationMDP, seed: int) -> "Agent": ...

    def act(self, features: np.ndarray, prev_action: int) -> int: ...


# --------------------------------------------------------------------------- tabular


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 60
    alpha: float = 0.5
    alpha_decay: float = 0.6
    alpha_min: float = 0.02
    full_sweep: bool = True  # update every (prev, action) pair at each step
    epsilon_start: float = 1.0  # exploration schedule, used when full_sweep is False
    epsilon_end: float = 0.05


class DoubleQLearningAgent:
    """Tabular Double Q-learning over discretized market regimes."""

    name = "q_learning"

    def __init__(self, config: QLearningConfig = QLearningConfig()):
        self.config = config

    def _regime(self, features: np.ndarray) -> np.ndarray:
        digits = [
            np.digitize(features[..., col], self.edges[i])
            for i, col in enumerate(TABULAR_FEATURES)
        ]
        return digits[0] * 9 + digits[1] * 3 + digits[2]

    def fit(self, mdp: AllocationMDP, seed: int = 0) -> "DoubleQLearningAgent":
        c, rng = self.config, np.random.default_rng(seed)
        gamma = mdp.config.gamma
        n_a, n_k = mdp.n_actions, mdp.n_steps
        self.baseline = mdp.baseline_action
        train = mdp.features[:n_k]
        self.edges = [np.quantile(train[:, col], [1 / 3, 2 / 3]) for col in TABULAR_FEATURES]
        regimes = self._regime(mdp.features)

        q = np.zeros((2, 27, n_a, n_a))
        visits = np.zeros(27)
        everyone = np.arange(n_a)
        for episode in range(c.episodes):
            prev = int(rng.integers(n_a))
            epsilon = c.epsilon_start + (c.epsilon_end - c.epsilon_start) * episode / max(
                c.episodes - 1, 1
            )
            for k in range(n_k):
                s, s_next = regimes[k], regimes[k + 1]
                visits[s] += 1
                alpha = max(c.alpha / visits[s] ** c.alpha_decay, c.alpha_min)
                i = int(rng.integers(2))  # table updated this step; the other evaluates
                best = q[i, s_next].argmax(axis=1)  # greedy next action for each new prev
                bootstrap = np.zeros(n_a) if k == n_k - 1 else q[1 - i, s_next, everyone, best]
                target = mdp.rewards[k] + gamma * bootstrap[None, :]
                if c.full_sweep:
                    q[i, s] += alpha * (target - q[i, s])
                else:
                    explore = rng.random() < epsilon
                    action = int(rng.integers(n_a)) if explore else int(q[:, s, prev].sum(0).argmax())
                    q[i, s, prev, action] += alpha * (target[prev, action] - q[i, s, prev, action])
                    prev = action
        self.q, self.visits = q.sum(axis=0) / 2, visits
        return self

    def act(self, features: np.ndarray, prev_action: int) -> int:
        regime = int(self._regime(features))
        if self.visits[regime] == 0:  # a regime never seen in training: stay at baseline
            return self.baseline
        return int(self.q[regime, prev_action].argmax())


# ------------------------------------------------------------------------------- DQN


class DuelingNetwork:
    """Two-hidden-layer dueling Q-network with hand-written backpropagation."""

    NAMES = ("w1", "b1", "w2", "b2", "wv", "bv", "wa", "ba")

    def __init__(self, n_inputs: int, n_actions: int, hidden: int, rng: np.random.Generator):
        def he(a: int, b: int, gain: float = 2.0) -> np.ndarray:
            return rng.normal(0.0, np.sqrt(gain / a), (a, b))

        self.params = {
            "w1": he(n_inputs, hidden), "b1": np.zeros(hidden),
            "w2": he(hidden, hidden), "b2": np.zeros(hidden),
            "wv": he(hidden, 1, 0.01), "bv": np.zeros(1),
            "wa": he(hidden, n_actions, 0.01), "ba": np.zeros(n_actions),
        }

    def copy_from(self, other: "DuelingNetwork", tau: float = 1.0) -> None:
        for name in self.NAMES:
            self.params[name] = (1 - tau) * self.params[name] + tau * other.params[name]

    def _forward(self, x: np.ndarray):
        p = self.params
        h1 = np.maximum(x @ p["w1"] + p["b1"], 0.0)
        h2 = np.maximum(h1 @ p["w2"] + p["b2"], 0.0)
        value = h2 @ p["wv"] + p["bv"]
        advantage = h2 @ p["wa"] + p["ba"]
        q = value + advantage - advantage.mean(axis=1, keepdims=True)
        return q, h1, h2

    def q_values(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[0]

    def loss_and_grads(
        self,
        x: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        delta: float = 1.0,
    ) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
        """Importance-weighted Huber loss on the chosen actions."""
        p = self.params
        q, h1, h2 = self._forward(x)
        n, n_actions = len(x), q.shape[1]
        rows = np.arange(n)
        td = q[rows, actions] - targets
        small = np.abs(td) <= delta
        huber = np.where(small, 0.5 * td**2, delta * (np.abs(td) - 0.5 * delta))
        loss = float((weights * huber).mean())

        dq = weights * np.clip(td, -delta, delta) / n  # dL / dQ(s, a)
        d_adv = np.tile((-dq / n_actions)[:, None], (1, n_actions))
        d_adv[rows, actions] += dq
        d_value = dq[:, None]

        grads = {
            "wv": h2.T @ d_value, "bv": d_value.sum(axis=0),
            "wa": h2.T @ d_adv, "ba": d_adv.sum(axis=0),
        }
        d_h2 = (d_value @ p["wv"].T + d_adv @ p["wa"].T) * (h2 > 0)
        grads["w2"], grads["b2"] = h1.T @ d_h2, d_h2.sum(axis=0)
        d_h1 = (d_h2 @ p["w2"].T) * (h1 > 0)
        grads["w1"], grads["b1"] = x.T @ d_h1, d_h1.sum(axis=0)
        return loss, td, grads


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float, max_norm: float = 10.0):
        self.lr, self.max_norm, self.t = lr, max_norm, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        norm = np.sqrt(sum(float((g**2).sum()) for g in grads.values()))
        scale = min(1.0, self.max_norm / (norm + 1e-12))
        for key, grad in grads.items():
            grad = grad * scale
            self.m[key] = 0.9 * self.m[key] + 0.1 * grad
            self.v[key] = 0.999 * self.v[key] + 0.001 * grad**2
            m_hat = self.m[key] / (1 - 0.9**self.t)
            v_hat = self.v[key] / (1 - 0.999**self.t)
            params[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


class PrioritizedReplay:
    """Proportional prioritized replay storing (k, prev, action) index triples."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity, self.alpha = capacity, alpha
        self.data = np.zeros((capacity, 3), dtype=np.int32)
        self.priorities = np.zeros(capacity)
        self.size = self.cursor = 0

    def add(self, k: int, prev: int, action: int) -> None:
        self.data[self.cursor] = (k, prev, action)
        self.priorities[self.cursor] = self.priorities[: self.size].max() if self.size else 1.0
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, beta: float, rng: np.random.Generator):
        scaled = self.priorities[: self.size] ** self.alpha
        probs = scaled / scaled.sum()
        index = np.minimum(
            np.searchsorted(np.cumsum(probs), rng.random(batch)), self.size - 1
        )
        weights = (self.size * probs[index]) ** -beta
        return index, self.data[index], weights / weights.max()

    def update(self, index: np.ndarray, td: np.ndarray) -> None:
        self.priorities[index] = np.abs(td) + 1e-3


@dataclass(frozen=True)
class DQNConfig:
    total_steps: int = 5000
    episode_length: int = 104
    warmup: int = 300
    train_every: int = 2
    batch_size: int = 64
    hidden: int = 64
    learning_rate: float = 1e-3
    tau: float = 0.01
    counterfactuals: int = 3  # extra (prev, action) pairs stored per environment step
    replay_capacity: int = 20_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    per_beta_start: float = 0.4
    ensemble: int = 3


class DuelingDQNAgent:
    """Ensemble of Double Dueling DQNs with prioritized, counterfactual replay."""

    name = "dqn"

    def __init__(self, config: DQNConfig = DQNConfig()):
        self.config = config
        self.networks: list[DuelingNetwork] = []

    def _encode(self, features: np.ndarray, prev: np.ndarray) -> np.ndarray:
        scaled = np.clip((features - self.mean) / self.std, -5.0, 5.0)
        return np.concatenate([scaled, self.eye[prev]], axis=-1)

    def _fit_member(self, mdp: AllocationMDP, seed: int) -> DuelingNetwork:
        c, rng = self.config, np.random.default_rng(seed)
        n_a, n_k = mdp.n_actions, mdp.n_steps
        gamma = mdp.config.gamma
        n_inputs = mdp.features.shape[1] + n_a
        online = DuelingNetwork(n_inputs, n_a, c.hidden, rng)
        target = DuelingNetwork(n_inputs, n_a, c.hidden, rng)
        target.copy_from(online)
        optimizer = Adam(online.params, c.learning_rate)
        replay = PrioritizedReplay(c.replay_capacity)

        step = 0
        while step < c.total_steps:
            k = int(rng.integers(max(n_k - c.episode_length, 1)))
            prev = int(rng.integers(n_a))
            for k in range(k, min(k + c.episode_length, n_k)):
                progress = min(step / max(c.total_steps * 0.7, 1), 1.0)
                epsilon = c.epsilon_start + (c.epsilon_end - c.epsilon_start) * progress
                if rng.random() < epsilon:
                    action = int(rng.integers(n_a))
                else:
                    state = self._encode(mdp.features[k], np.array(prev))[None, :]
                    action = int(online.q_values(state)[0].argmax())
                replay.add(k, prev, action)
                for _ in range(c.counterfactuals):
                    replay.add(k, int(rng.integers(n_a)), int(rng.integers(n_a)))

                if step >= c.warmup and step % c.train_every == 0:
                    beta = c.per_beta_start + (1 - c.per_beta_start) * step / c.total_steps
                    index, batch, weights = replay.sample(c.batch_size, min(beta, 1.0), rng)
                    ks, prevs, actions = batch.T
                    next_states = self._encode(mdp.features[ks + 1], actions)
                    best = online.q_values(next_states).argmax(axis=1)
                    bootstrap = target.q_values(next_states)[np.arange(len(ks)), best]
                    not_done = (ks < n_k - 1).astype(float)
                    y = mdp.rewards[ks, prevs, actions] + gamma * not_done * bootstrap
                    states = self._encode(mdp.features[ks], prevs)
                    _, td, grads = online.loss_and_grads(states, actions, y, weights)
                    optimizer.step(online.params, grads)
                    target.copy_from(online, c.tau)
                    replay.update(index, td)
                prev = action
                step += 1
                if step >= c.total_steps:
                    break
        return online

    def fit(self, mdp: AllocationMDP, seed: int = 0) -> "DuelingDQNAgent":
        train = mdp.features[: mdp.n_steps]
        self.mean, self.std = train.mean(axis=0), train.std(axis=0) + 1e-8
        self.eye = np.eye(mdp.n_actions)
        self.networks = [
            self._fit_member(mdp, seed + 1000 * i) for i in range(self.config.ensemble)
        ]
        return self

    def q_values(self, features: np.ndarray, prev_action: int) -> np.ndarray:
        state = self._encode(features, np.array(prev_action))[None, :]
        return np.mean([net.q_values(state)[0] for net in self.networks], axis=0)

    def act(self, features: np.ndarray, prev_action: int) -> int:
        return int(self.q_values(features, prev_action).argmax())


AGENTS = {"qlearning": DoubleQLearningAgent, "dqn": DuelingDQNAgent}
