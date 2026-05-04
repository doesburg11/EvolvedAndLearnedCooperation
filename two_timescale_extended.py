"""
Two-timescale cooperation model — extended version.

Builds on the Q-learning model with three additional mechanisms:

1. Reputation / indirect reciprocity
   Agents broadcast observations of partner behaviour. When meeting a
   stranger, an agent augments its Q-value prior with the partner's
   public reputation score.

2. Partner choice / social exclusion
   Before each interaction an agent can reject a partner whose
   reputation falls below an evolved rejection threshold. Rejected
   partners receive no benefit and gain no reputation update.
   This makes reputation actionable.

3. Forgiveness
   After being defected against, an agent's Q-value for that partner
   is penalised. An evolved forgiveness_rate gene determines how
   quickly that penalty decays back toward the prior on subsequent
   cooperative rounds. Fast forgiveness supports relationship repair;
   slow forgiveness makes punishment credible.

Evolved parameters per agent
-----------------------------
exploration_rate    epsilon for epsilon-greedy action selection
learning_rate       alpha, Q-learning step size
discount_factor     gamma, future reward weight
initial_q_bias      starting optimism/pessimism for unknown partners
rejection_threshold reputation score below which partner is refused
forgiveness_rate    per-round decay of post-betrayal Q penalty
reputation_weight   how much public reputation shifts the Q-prior
"""

import argparse
import dataclasses
import numpy as np
import matplotlib.pyplot as plt

from live_viewer import SimulationViewer


COOPERATE = 0
DEFECT = 1

# Public reputation scores are shared across all agents each round.
# Shape: (num_agents,). Updated after every interaction.
_reputation: np.ndarray | None = None


def _init_reputation(num_agents: int) -> None:
    global _reputation
    _reputation = np.zeros(num_agents)


def _update_reputation(agent_id: int, cooperated: bool) -> None:
    """Increment or decrement agent's public reputation."""
    assert _reputation is not None
    delta = 0.1 if cooperated else -0.1
    _reputation[agent_id] = np.clip(_reputation[agent_id] + delta, -1.0, 1.0)


def _get_reputation(agent_id: int) -> float:
    assert _reputation is not None
    return float(_reputation[agent_id])


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Agent:
    """Agent with Q-learning + reputation + partner choice + forgiveness."""

    # Evolved traits
    exploration_rate: float
    learning_rate: float
    discount_factor: float
    initial_q_bias: float
    rejection_threshold: float   # refuse partners below this reputation
    forgiveness_rate: float      # per-round decay of betrayal penalty [0,1]
    reputation_weight: float     # how much public rep shifts Q prior

    # Lifetime state
    Q: dict = dataclasses.field(default_factory=dict)
    betrayal_penalty: dict = dataclasses.field(default_factory=dict)
    payoff: float = 0.0

    def get_q(self, partner_id: int) -> tuple[float, float]:
        """
        Return (Q_cooperate, Q_defect) for a partner.

        For unknown partners, seed from initial_q_bias + public reputation.
        """
        rep = _get_reputation(partner_id)
        prior = self.initial_q_bias + self.reputation_weight * rep
        penalty = self.betrayal_penalty.get(partner_id, 0.0)
        q_c = self.Q.get((partner_id, COOPERATE), prior) - penalty
        q_d = self.Q.get((partner_id, DEFECT), prior)
        return q_c, q_d

    def set_q(self, partner_id: int, action: int, value: float) -> None:
        self.Q[(partner_id, action)] = value

    def select_action(
        self, partner_id: int, rng: np.random.Generator
    ) -> int:
        """Epsilon-greedy, informed by reputation-adjusted Q-values."""
        if rng.random() < self.exploration_rate:
            return int(rng.integers(0, 2))
        q_c, q_d = self.get_q(partner_id)
        return COOPERATE if q_c >= q_d else DEFECT

    def will_interact(self, partner_id: int) -> bool:
        """Return False if partner reputation is below rejection threshold."""
        return _get_reputation(partner_id) >= self.rejection_threshold

    def learn(
        self,
        partner_id: int,
        action: int,
        reward: float,
        next_max_q: float,
    ) -> None:
        """Standard Q-learning update."""
        q_c, q_d = self.get_q(partner_id)
        current_q = q_c if action == COOPERATE else q_d
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.set_q(partner_id, action, new_q)

    def apply_betrayal(self, partner_id: int, penalty: float) -> None:
        """Record a betrayal penalty for this partner."""
        self.betrayal_penalty[partner_id] = (
            self.betrayal_penalty.get(partner_id, 0.0) + penalty
        )

    def decay_forgiveness(self, partner_id: int) -> None:
        """
        Decay betrayal penalty toward zero at forgiveness_rate.
        Called when partner cooperated this round.
        """
        if partner_id in self.betrayal_penalty:
            self.betrayal_penalty[partner_id] *= (1.0 - self.forgiveness_rate)
            if abs(self.betrayal_penalty[partner_id]) < 1e-4:
                del self.betrayal_penalty[partner_id]

    def reset_for_generation(self) -> None:
        self.Q.clear()
        self.betrayal_penalty.clear()
        self.payoff = 0.0


# ---------------------------------------------------------------------------
# Network topology
# ---------------------------------------------------------------------------

def make_ring_neighbors(num_agents: int) -> list[list[int]]:
    neighbors = []
    for i in range(num_agents):
        left = (i - 1) % num_agents
        right = (i + 1) % num_agents
        neighbors.append([left, right])
    return neighbors


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------

BETRAYAL_PENALTY_MAGNITUDE = 0.5


def _interact_pair(
    agents: list[Agent],
    i: int,
    j: int,
    benefit: float,
    cost: float,
    rng: np.random.Generator,
) -> tuple[int | None, int | None]:
    """
    Simultaneous interaction with partner choice, reputation update,
    and forgiveness.
    """
    agent_i = agents[i]
    agent_j = agents[j]

    # Partner choice: either agent can refuse
    i_accepts = agent_i.will_interact(j)
    j_accepts = agent_j.will_interact(i)

    if not (i_accepts and j_accepts):
        # Rejected partner loses a little reputation
        if not i_accepts:
            _update_reputation(j, cooperated=False)
        if not j_accepts:
            _update_reputation(i, cooperated=False)
        return None, None

    # Simultaneous action selection
    action_i = agent_i.select_action(j, rng)
    action_j = agent_j.select_action(i, rng)

    # Rewards
    reward_i = 0.0
    reward_j = 0.0

    if action_i == COOPERATE:
        reward_i -= cost
        reward_j += benefit
    if action_j == COOPERATE:
        reward_j -= cost
        reward_i += benefit

    agent_i.payoff += reward_i
    agent_j.payoff += reward_j

    # Reputation updates (based on what was observed by neighbours)
    _update_reputation(i, cooperated=(action_i == COOPERATE))
    _update_reputation(j, cooperated=(action_j == COOPERATE))

    # Forgiveness / betrayal tracking
    if action_j == DEFECT:
        agent_i.apply_betrayal(j, BETRAYAL_PENALTY_MAGNITUDE)
    else:
        agent_i.decay_forgiveness(j)

    if action_i == DEFECT:
        agent_j.apply_betrayal(i, BETRAYAL_PENALTY_MAGNITUDE)
    else:
        agent_j.decay_forgiveness(i)

    # Q-learning updates
    q_c_i, q_d_i = agent_i.get_q(j)
    next_max_q_i = max(q_c_i, q_d_i)
    agent_i.learn(j, action_i, reward_i, next_max_q_i)

    q_c_j, q_d_j = agent_j.get_q(i)
    next_max_q_j = max(q_c_j, q_d_j)
    agent_j.learn(i, action_j, reward_j, next_max_q_j)
    return action_i, action_j


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    num_agents: int = 100,
    num_generations: int = 120,
    lifetime_rounds: int = 80,
    benefit: float = 3.0,
    cost: float = 1.0,
    mutation_std: float = 0.1,
    generation_callback=None,
    round_callback=None,
) -> dict[str, list[float]]:
    """Run the extended simulation."""
    rng = np.random.default_rng(seed=42)
    _init_reputation(num_agents)

    agents = [
        Agent(
            exploration_rate=rng.uniform(0.1, 0.5),
            learning_rate=rng.uniform(0.1, 0.5),
            discount_factor=rng.uniform(0.5, 0.95),
            initial_q_bias=rng.normal(0, 1),
            rejection_threshold=rng.uniform(-0.5, 0.5),
            forgiveness_rate=rng.uniform(0.0, 0.5),
            reputation_weight=rng.uniform(0.0, 1.0),
        )
        for _ in range(num_agents)
    ]

    neighbors = make_ring_neighbors(num_agents)

    history: dict[str, list[float]] = {
        "mean_cooperation": [],
        "mean_payoff": [],
        "mean_exploration_rate": [],
        "mean_learning_rate": [],
        "mean_discount_factor": [],
        "mean_initial_q_bias": [],
        "mean_rejection_threshold": [],
        "mean_forgiveness_rate": [],
        "mean_reputation_weight": [],
        "mean_reputation": [],
    }

    for _generation in range(num_generations):
        _init_reputation(num_agents)

        for round_idx in range(lifetime_rounds):
            round_events: list[tuple[int, int, int, int]] = []
            for i in range(num_agents):
                for j in neighbors[i]:
                    act_i, act_j = _interact_pair(
                        agents, i, j, benefit, cost, rng
                    )
                    if act_i is not None and act_j is not None:
                        round_events.append((i, j, act_i, act_j))
            if round_callback is not None:
                round_callback(_generation, round_idx, round_events)

        # Cooperation rate: sample across all neighbor pairs
        coop_actions = [
            agents[i].select_action(j, rng)
            for i in range(num_agents)
            for j in neighbors[i]
        ]
        history["mean_cooperation"].append(
            float(np.mean([a == COOPERATE for a in coop_actions]))
        )
        if generation_callback is not None:
            generation_callback(_generation, history["mean_cooperation"][-1])
        history["mean_payoff"].append(
            float(np.mean([a.payoff for a in agents]))
        )
        history["mean_exploration_rate"].append(
            float(np.mean([a.exploration_rate for a in agents]))
        )
        history["mean_learning_rate"].append(
            float(np.mean([a.learning_rate for a in agents]))
        )
        history["mean_discount_factor"].append(
            float(np.mean([a.discount_factor for a in agents]))
        )
        history["mean_initial_q_bias"].append(
            float(np.mean([a.initial_q_bias for a in agents]))
        )
        history["mean_rejection_threshold"].append(
            float(np.mean([a.rejection_threshold for a in agents]))
        )
        history["mean_forgiveness_rate"].append(
            float(np.mean([a.forgiveness_rate for a in agents]))
        )
        history["mean_reputation_weight"].append(
            float(np.mean([a.reputation_weight for a in agents]))
        )
        assert _reputation is not None
        history["mean_reputation"].append(float(np.mean(_reputation)))

        # Reproduction
        payoffs = np.array([a.payoff for a in agents])
        shifted = payoffs - payoffs.min() + 0.1
        weights = shifted / shifted.sum()
        parent_indices = rng.choice(num_agents, size=num_agents, p=weights)

        def _mutate(val: float, lo: float, hi: float) -> float:
            return float(np.clip(val + rng.normal(0, mutation_std), lo, hi))

        new_agents = []
        for idx in parent_indices:
            p = agents[idx]
            new_agents.append(
                Agent(
                    exploration_rate=_mutate(p.exploration_rate, 0.0, 1.0),
                    learning_rate=_mutate(p.learning_rate, 0.01, 1.0),
                    discount_factor=_mutate(p.discount_factor, 0.0, 1.0),
                    initial_q_bias=p.initial_q_bias
                    + rng.normal(0, mutation_std),
                    rejection_threshold=_mutate(
                        p.rejection_threshold, -1.0, 1.0
                    ),
                    forgiveness_rate=_mutate(p.forgiveness_rate, 0.0, 1.0),
                    reputation_weight=_mutate(p.reputation_weight, 0.0, 2.0),
                )
            )

        agents = new_agents
        for a in agents:
            a.reset_for_generation()

    return history


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def summarize(history: dict[str, list[float]], label: str) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print(
        f"Final cooperation:          {history['mean_cooperation'][-1]:.3f}"
    )
    print(f"Final payoff:               {history['mean_payoff'][-1]:.3f}")
    print(
        "Final exploration rate:     "
        f"{history['mean_exploration_rate'][-1]:.3f}"
    )
    print(
        "Final learning rate:        "
        f"{history['mean_learning_rate'][-1]:.3f}"
    )
    print(
        "Final discount factor:      "
        f"{history['mean_discount_factor'][-1]:.3f}"
    )
    print(
        "Final initial Q-bias:       "
        f"{history['mean_initial_q_bias'][-1]:.3f}"
    )
    print(
        "Final rejection threshold:  "
        f"{history['mean_rejection_threshold'][-1]:.3f}"
    )
    print(
        "Final forgiveness rate:     "
        f"{history['mean_forgiveness_rate'][-1]:.3f}"
    )
    print(
        "Final reputation weight:    "
        f"{history['mean_reputation_weight'][-1]:.3f}"
    )
    print(
        "Final mean reputation:      "
        f"{history['mean_reputation'][-1]:.3f}"
    )


def plot_history(
    history: dict[str, list[float]],
    title: str,
    save_prefix: str | None = None,
) -> None:
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(history["mean_cooperation"], linewidth=2, label="Cooperation")
    plt.plot(
        history["mean_reputation"],
        linewidth=2,
        linestyle="--",
        label="Mean reputation",
    )
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_prefix:
        fig1.savefig(f"output/{save_prefix}_cooperation.png", dpi=150)

    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(
        history["mean_exploration_rate"],
        label="Exploration rate (ε)",
        linewidth=2,
    )
    plt.plot(
        history["mean_learning_rate"],
        label="Learning rate (α)",
        linewidth=2,
    )
    plt.plot(
        history["mean_discount_factor"],
        label="Discount factor (γ)",
        linewidth=2,
    )
    plt.plot(
        history["mean_initial_q_bias"],
        label="Initial Q-bias",
        linewidth=2,
    )
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Mean parameter value", fontsize=12)
    plt.title(title + " — Q-learning parameters", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_prefix:
        fig2.savefig(f"output/{save_prefix}_ql_params.png", dpi=150)

    fig3 = plt.figure(figsize=(10, 6))
    plt.plot(
        history["mean_rejection_threshold"],
        label="Rejection threshold",
        linewidth=2,
    )
    plt.plot(
        history["mean_forgiveness_rate"],
        label="Forgiveness rate",
        linewidth=2,
    )
    plt.plot(
        history["mean_reputation_weight"],
        label="Reputation weight",
        linewidth=2,
    )
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Mean parameter value", fontsize=12)
    plt.title(title + " — social parameters", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_prefix:
        fig3.savefig(f"output/{save_prefix}_social_params.png", dpi=150)


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(
        description="Run extended two-timescale simulation"
    )
    parser.add_argument(
        "--no-live-grid",
        action="store_true",
        help="Disable live cooperation and encounter-matrix windows",
    )
    args = parser.parse_args()
    live = not args.no_live_grid

    os.makedirs("output", exist_ok=True)

    print("Running extended model (reputation + partner choice + forgiveness)")
    print("This may take a moment...\n")

    # One-shot
    os_viewer = (
        SimulationViewer(
            100, 120, 1,
            title="One-shot interaction (Extended model)",
        )
        if live else None
    )
    one_shot = run_simulation(
        lifetime_rounds=1,
        generation_callback=os_viewer.update_generation if os_viewer else None,
        round_callback=os_viewer.update_encounter_round if os_viewer else None,
    )
    summarize(one_shot, "One-shot interaction (extended model)")

    # Repeated
    rep_viewer = (
        SimulationViewer(
            100, 120, 80,
            title="Repeated interaction (Extended model)",
        )
        if live else None
    )
    repeated = run_simulation(
        lifetime_rounds=80,
        generation_callback=rep_viewer.update_generation if rep_viewer else None,
        round_callback=rep_viewer.update_encounter_round if rep_viewer else None,
    )
    summarize(repeated, "Repeated interaction (extended model)")

    plot_history(
        one_shot,
        "One-shot interaction (extended model)",
        save_prefix="ext_one_shot",
    )
    plot_history(
        repeated,
        "Repeated interaction (extended model)",
        save_prefix="ext_repeated",
    )

    if live:
        print("\nLive viewer: close plot windows to finish.")
        plt.ioff()
    plt.show()
