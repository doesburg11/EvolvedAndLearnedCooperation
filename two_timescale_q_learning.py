"""
Two-timescale reciprocity simulation with true Q-learning.

Agents learn partner-specific Q-values for cooperation vs defection.
Evolution acts on Q-learning parameters: exploration rate, learning rate,
discount factor, and initial Q-bias.

This is a more sophisticated model than the simple trust-learning version.
Agents explicitly learn the expected value of each action with each partner.
"""

import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple


COOPERATE = 0
DEFECT = 1


class Params(NamedTuple):
    """Simulation parameters."""

    num_agents: int = 100
    num_generations: int = 120
    lifetime_rounds: int = 80
    benefit: float = 3.0
    cost: float = 1.0
    mutation_std: float = 0.1


@dataclasses.dataclass
class Agent:
    """An agent with Q-learning parameters and Q-table."""

    # Evolved traits (inherited, mutated each generation)
    exploration_rate: float  # epsilon for epsilon-greedy
    learning_rate: float  # alpha, Q-learning step size
    discount_factor: float  # gamma, future reward weight
    initial_q_bias: float  # initial optimism/pessimism for Q-values

    # Learned during lifetime
    # Q[partner_id, action] = expected value of action with partner
    Q: dict = dataclasses.field(default_factory=dict)

    # Lifetime stats
    payoff: float = 0.0

    def get_q(self, partner_id: int, action: int) -> float:
        """Get Q-value, initializing if needed."""
        if (partner_id, action) not in self.Q:
            self.Q[(partner_id, action)] = self.initial_q_bias
        return self.Q[(partner_id, action)]

    def set_q(self, partner_id: int, action: int, value: float) -> None:
        """Set Q-value."""
        self.Q[(partner_id, action)] = value

    def select_action(
        self, partner_id: int, rng: np.random.Generator
    ) -> int:
        """
        Epsilon-greedy action selection.
        With prob epsilon: random action.
        With prob 1-epsilon: greedy (best Q-value).
        """
        if rng.random() < self.exploration_rate:
            return int(rng.integers(0, 2))  # random action

        q_coop = self.get_q(partner_id, COOPERATE)
        q_defect = self.get_q(partner_id, DEFECT)
        return COOPERATE if q_coop >= q_defect else DEFECT

    def learn(
        self, partner_id: int, action: int, reward: float, next_max_q: float
    ) -> None:
        """
        Q-learning update.
        Q[s,a] += alpha * (r + gamma * max_Q[s'] - Q[s,a])
        """
        current_q = self.get_q(partner_id, action)
        new_q = (
            current_q
            + self.learning_rate
            * (reward + self.discount_factor * next_max_q - current_q)
        )
        self.set_q(partner_id, action, new_q)

    def reset_for_generation(self) -> None:
        """Clear lifetime state for next generation."""
        self.Q.clear()
        self.payoff = 0.0


def make_ring_neighbors(num_agents: int) -> list[list[int]]:
    """Create a ring topology: each agent has left and right neighbors."""
    neighbors = []
    for i in range(num_agents):
        left = (i - 1) % num_agents
        right = (i + 1) % num_agents
        neighbors.append([left, right])
    return neighbors


def run_simulation(params: Params) -> dict[str, list[float]]:
    """Run a two-timescale Q-learning simulation."""
    rng = np.random.default_rng(seed=42)

    # Initialize population
    agents = [
        Agent(
            exploration_rate=rng.uniform(0.1, 0.5),
            learning_rate=rng.uniform(0.1, 0.5),
            discount_factor=rng.uniform(0.5, 0.95),
            initial_q_bias=rng.normal(0, 1),
        )
        for _ in range(params.num_agents)
    ]

    neighbors = make_ring_neighbors(params.num_agents)

    history = {
        "mean_cooperation": [],
        "mean_payoff": [],
        "mean_exploration_rate": [],
        "mean_learning_rate": [],
        "mean_discount_factor": [],
        "mean_initial_q_bias": [],
    }

    # Run generations
    for generation in range(params.num_generations):
        # Lifetime interactions
        for _ in range(params.lifetime_rounds):
            for i in range(params.num_agents):
                for j in neighbors[i]:
                    # Both directions of interaction
                    _interact_pair(agents, i, j, params, rng)

        # Record statistics
        cooperation = np.mean(
            [
                agents[i].select_action(j, rng)
                for i in range(params.num_agents)
                for j in neighbors[i]
            ]
        )
        history["mean_cooperation"].append(cooperation)
        history["mean_payoff"].append(
            np.mean([a.payoff for a in agents])
        )
        history["mean_exploration_rate"].append(
            np.mean([a.exploration_rate for a in agents])
        )
        history["mean_learning_rate"].append(
            np.mean([a.learning_rate for a in agents])
        )
        history["mean_discount_factor"].append(
            np.mean([a.discount_factor for a in agents])
        )
        history["mean_initial_q_bias"].append(
            np.mean([a.initial_q_bias for a in agents])
        )

        # Reproduction based on payoff (shift to ensure non-negative)
        payoffs = np.array([a.payoff for a in agents])
        # Shift payoffs so minimum is 0.1 (ensure all positive for weighting)
        shifted_payoffs = payoffs - payoffs.min() + 0.1
        weights = shifted_payoffs / shifted_payoffs.sum()

        parent_indices = rng.choice(
            params.num_agents, size=params.num_agents, p=weights
        )
        new_agents = []
        for parent_idx in parent_indices:
            parent = agents[parent_idx]
            child = Agent(
                exploration_rate=parent.exploration_rate
                + rng.normal(0, params.mutation_std),
                learning_rate=parent.learning_rate
                + rng.normal(0, params.mutation_std),
                discount_factor=parent.discount_factor
                + rng.normal(0, params.mutation_std),
                initial_q_bias=parent.initial_q_bias
                + rng.normal(0, params.mutation_std),
            )
            # Clamp parameters to valid ranges
            child.exploration_rate = np.clip(child.exploration_rate, 0.0, 1.0)
            child.learning_rate = np.clip(child.learning_rate, 0.01, 1.0)
            child.discount_factor = np.clip(child.discount_factor, 0.0, 1.0)
            new_agents.append(child)

        agents = new_agents

        # Reset for next generation
        for a in agents:
            a.reset_for_generation()

    return history


def _interact_pair(
    agents: list[Agent],
    i: int,
    j: int,
    params: Params,
    rng: np.random.Generator,
) -> None:
    """
    Single simultaneous interaction between agents i and j.

    Both agents choose their actions before observing the other's choice,
    so rewards include what each received from the other. next_max_q is
    the agent's current best Q-value for this same partner, giving the
    discount factor a real role in bootstrapping future relationship value.
    """
    agent_i = agents[i]
    agent_j = agents[j]

    # Simultaneous action selection
    action_i = agent_i.select_action(j, rng)
    action_j = agent_j.select_action(i, rng)

    # Full reward: what i paid + what i received from j
    reward_i = 0.0
    if action_i == COOPERATE:
        reward_i -= params.cost
        agent_j.payoff += params.benefit
    if action_j == COOPERATE:
        reward_i += params.benefit
        agent_j.payoff -= params.cost

    reward_j = 0.0
    if action_j == COOPERATE:
        reward_j -= params.cost
    if action_i == COOPERATE:
        reward_j += params.benefit

    agent_i.payoff += reward_i
    agent_j.payoff += reward_j

    # next_max_q: best Q-value agent currently holds for this same partner.
    # This bootstraps the long-term value of the relationship, making the
    # discount factor (gamma) genuinely functional.
    next_max_q_i = max(
        agent_i.get_q(j, COOPERATE), agent_i.get_q(j, DEFECT)
    )
    next_max_q_j = max(
        agent_j.get_q(i, COOPERATE), agent_j.get_q(i, DEFECT)
    )

    agent_i.learn(j, action_i, reward_i, next_max_q_i)
    agent_j.learn(i, action_j, reward_j, next_max_q_j)


def plot_history(
    history: dict[str, list[float]],
    title: str,
    save_prefix: str | None = None,
) -> None:
    """Plot cooperation and traits over time."""
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(history["mean_cooperation"], linewidth=2)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Mean cooperation", fontsize=12)
    plt.title(title, fontsize=14)
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
    plt.title(title + " — evolved Q-learning parameters", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_prefix:
        fig2.savefig(f"output/{save_prefix}_parameters.png", dpi=150)


def summarize(history: dict[str, list[float]], label: str) -> None:
    """Print final statistics."""
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Final cooperation:       {history['mean_cooperation'][-1]:.3f}")
    print(f"Final payoff:            {history['mean_payoff'][-1]:.3f}")
    print(
        f"Final exploration rate:  {history['mean_exploration_rate'][-1]:.3f}"
    )
    print(
        f"Final learning rate:     {history['mean_learning_rate'][-1]:.3f}"
    )
    print(
        f"Final discount factor:   {history['mean_discount_factor'][-1]:.3f}"
    )
    print(
        f"Final initial Q-bias:    {history['mean_initial_q_bias'][-1]:.3f}"
    )


if __name__ == "__main__":
    import os

    os.makedirs("output", exist_ok=True)

    # Case 1: one-shot interaction
    one_shot_params = Params(lifetime_rounds=1)
    one_shot = run_simulation(one_shot_params)
    summarize(one_shot, "One-shot interaction (Q-learning)")

    # Case 2: repeated interaction
    repeated_params = Params(lifetime_rounds=80)
    repeated = run_simulation(repeated_params)
    summarize(repeated, "Repeated interaction (Q-learning)")

    plot_history(
        one_shot, "One-shot interaction (Q-learning)", save_prefix="q_one_shot"
    )
    plot_history(
        repeated,
        "Repeated interaction (Q-learning)",
        save_prefix="q_repeated",
    )

    plt.show()
