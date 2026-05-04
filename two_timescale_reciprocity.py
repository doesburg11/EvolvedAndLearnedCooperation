import argparse

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from live_viewer import SimulationViewer


@dataclass
class Params:
    seed: int = 4
    population_size: int = 120
    generations: int = 120

    # Fast timescale: number of interaction rounds inside one
    # lifetime/generation. Try 1 versus 80. One-shot interaction usually
    # destroys reciprocity.
    lifetime_rounds: int = 80

    # Network reciprocity: agents repeatedly meet local neighbors instead of
    # the whole population.
    neighbors_per_agent: int = 8

    # Donation-game / Prisoner's-Dilemma-like payoffs.
    benefit: float = 3.0
    cost: float = 1.0

    # Evolutionary selection strength between generations.
    selection_strength: float = 0.7

    # Mistakes: even a cooperative agent sometimes defects, and vice versa.
    tremble_probability: float = 0.01

    # Initial inherited traits.
    initial_trust_mean: float = 0.1
    initial_trust_sd: float = 0.2
    initial_learning_rate_min: float = 0.05
    initial_learning_rate_max: float = 0.25
    initial_responsiveness_min: float = 0.5
    initial_responsiveness_max: float = 2.0

    # Mutation per generation.
    mutation_sd_trust: float = 0.10
    mutation_sd_learning_rate: float = 0.03
    mutation_sd_responsiveness: float = 0.08


def make_ring_neighbors(n: int, k: int) -> list[np.ndarray]:
    """
    Ring network: each agent interacts mostly with k local neighbors.
    This creates repeated encounters, which is important for direct
    reciprocity.
    """
    if k % 2 != 0:
        raise ValueError("neighbors_per_agent must be even.")

    neighbors = []
    half = k // 2

    for i in range(n):
        local = []
        for distance in range(1, half + 1):
            local.append((i - distance) % n)
            local.append((i + distance) % n)
        neighbors.append(np.array(local, dtype=int))

    return neighbors


def initialize_genes(
    params: Params, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """
    These inherited traits are the 'nature' layer.

    trust_prior:
        Initial tendency to cooperate with an unknown partner.
        Higher means more willing to start cooperatively.

    learning_rate:
        How quickly the agent updates trust from experience.

    responsiveness:
        How strongly learned trust changes future behavior.
    """
    n = params.population_size

    return {
        "trust_prior": rng.normal(
            params.initial_trust_mean,
            params.initial_trust_sd,
            n,
        ),
        "learning_rate": rng.uniform(
            params.initial_learning_rate_min,
            params.initial_learning_rate_max,
            n,
        ),
        "responsiveness": rng.uniform(
            params.initial_responsiveness_min,
            params.initial_responsiveness_max,
            n,
        ),
    }


def interact(
    i: int,
    j: int,
    genes: dict[str, np.ndarray],
    learned_trust: np.ndarray,
    payoff: np.ndarray,
    params: Params,
    rng: np.random.Generator,
) -> int:
    """
    One social interaction.

    The learned_trust matrix is the 'nurture' layer:
    learned_trust[i, j] is what i has learned about j during this lifetime.

    Returns the number of cooperative actions in this interaction: 0, 1, or 2.
    """
    # Decision rule:
    # inherited starting bias + inherited responsiveness *
    # learned partner-specific trust
    score_i = (
        genes["trust_prior"][i]
        + genes["responsiveness"][i] * learned_trust[i, j]
    )
    score_j = (
        genes["trust_prior"][j]
        + genes["responsiveness"][j] * learned_trust[j, i]
    )

    cooperate_i = score_i > 0.0
    cooperate_j = score_j > 0.0

    # Occasional mistakes prevent the simulation from becoming too clean.
    if rng.random() < params.tremble_probability:
        cooperate_i = not cooperate_i
    if rng.random() < params.tremble_probability:
        cooperate_j = not cooperate_j

    # Donation game:
    # If I cooperate, I pay a cost and the other receives a benefit.
    if cooperate_i:
        payoff[i] -= params.cost
        payoff[j] += params.benefit

    if cooperate_j:
        payoff[j] -= params.cost
        payoff[i] += params.benefit

    # Learning:
    # I become more trusting of a partner who cooperated,
    # and less trusting of a partner who defected.
    target_for_i = 1.0 if cooperate_j else -1.0
    target_for_j = 1.0 if cooperate_i else -1.0

    alpha_i = genes["learning_rate"][i]
    alpha_j = genes["learning_rate"][j]

    learned_trust[i, j] += alpha_i * (target_for_i - learned_trust[i, j])
    learned_trust[j, i] += alpha_j * (target_for_j - learned_trust[j, i])

    return int(cooperate_i) + int(cooperate_j), cooperate_i, cooperate_j


def reproduce(
    genes: dict[str, np.ndarray],
    payoff: np.ndarray,
    params: Params,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Slow evolutionary step.

    Agents with higher lifetime payoff are more likely to become parents.
    Offspring inherit the parent's traits with mutation.
    """
    n = params.population_size

    # Softmax-style selection. Standardizing keeps it numerically stable.
    standardized_payoff = (payoff - payoff.mean()) / (payoff.std() + 1e-9)
    reproductive_weight = np.exp(
        params.selection_strength * standardized_payoff
    )
    reproductive_weight /= reproductive_weight.sum()

    parents = rng.choice(n, size=n, replace=True, p=reproductive_weight)

    child_genes = {
        name: values[parents].copy()
        for name, values in genes.items()
    }

    # Mutation.
    child_genes["trust_prior"] += rng.normal(0.0, params.mutation_sd_trust, n)
    child_genes["learning_rate"] += rng.normal(
        0.0, params.mutation_sd_learning_rate, n
    )
    child_genes["responsiveness"] += rng.normal(
        0.0, params.mutation_sd_responsiveness, n
    )

    # Keep values in reasonable ranges.
    child_genes["trust_prior"] = np.clip(child_genes["trust_prior"], -3.0, 3.0)
    child_genes["learning_rate"] = np.clip(
        child_genes["learning_rate"], 0.01, 0.80
    )
    child_genes["responsiveness"] = np.clip(
        child_genes["responsiveness"], 0.00, 5.00
    )

    return child_genes


def run_simulation(
    params: Params,
    generation_callback=None,
    round_callback=None,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(params.seed)
    n = params.population_size

    genes = initialize_genes(params, rng)
    neighbors = make_ring_neighbors(n, params.neighbors_per_agent)

    history = {
        "mean_cooperation": [],
        "mean_payoff": [],
        "mean_trust_prior": [],
        "mean_learning_rate": [],
        "mean_responsiveness": [],
    }

    for generation in range(params.generations):
        # Fast timescale starts fresh each generation:
        # offspring inherit genes, not the parent's lifetime memories.
        learned_trust = np.zeros((n, n), dtype=float)
        payoff = np.zeros(n, dtype=float)

        cooperative_actions = 0
        total_actions = 0

        # Fast developmental / learning process.
        for round_idx in range(params.lifetime_rounds):
            round_events: list[tuple[int, int, int, int]] = []
            for i in range(n):
                j = int(rng.choice(neighbors[i]))

                coop_count, coop_i, coop_j = interact(
                    i=i,
                    j=j,
                    genes=genes,
                    learned_trust=learned_trust,
                    payoff=payoff,
                    params=params,
                    rng=rng,
                )
                cooperative_actions += coop_count
                total_actions += 2
                round_events.append((i, j, int(coop_i), int(coop_j)))

            if round_callback is not None:
                round_callback(generation, round_idx, round_events)

        history["mean_cooperation"].append(cooperative_actions / total_actions)
        if generation_callback is not None:
            generation_callback(generation, history["mean_cooperation"][-1])
        history["mean_payoff"].append(float(payoff.mean()))
        history["mean_trust_prior"].append(float(genes["trust_prior"].mean()))
        history["mean_learning_rate"].append(
            float(genes["learning_rate"].mean())
        )
        history["mean_responsiveness"].append(
            float(genes["responsiveness"].mean())
        )

        # Slow evolutionary process.
        genes = reproduce(genes, payoff, params, rng)

    return history


def plot_history(
        history: dict[str, list[float]],
        title: str, save_prefix: str | None = None
        ) -> None:
    fig1 = plt.figure()
    plt.plot(history["mean_cooperation"])
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Generation")
    plt.ylabel("Mean cooperation")
    plt.title(title)
    plt.tight_layout()
    if save_prefix:
        fig1.savefig(f"output/{save_prefix}_cooperation.png", dpi=150)

    fig2 = plt.figure()
    plt.plot(history["mean_trust_prior"], label="Inherited trust prior")
    plt.plot(history["mean_learning_rate"], label="Inherited learning rate")
    plt.plot(history["mean_responsiveness"], label="Inherited responsiveness")
    plt.xlabel("Generation")
    plt.ylabel("Mean inherited trait value")
    plt.title(title + " — evolved traits")
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        fig2.savefig(f"output/{save_prefix}_traits.png", dpi=150)


def summarize(history: dict[str, list[float]], label: str) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Final cooperation:       {history['mean_cooperation'][-1]:.3f}")
    print(f"Final payoff:            {history['mean_payoff'][-1]:.3f}")
    print(f"Final trust prior:       {history['mean_trust_prior'][-1]:.3f}")
    print(f"Final learning rate:     {history['mean_learning_rate'][-1]:.3f}")
    print(f"Final responsiveness:    {history['mean_responsiveness'][-1]:.3f}")


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(
        description="Run trust-learning two-timescale simulation"
    )
    parser.add_argument(
        "--no-live-grid",
        action="store_true",
        help="Disable live cooperation and encounter-matrix windows",
    )
    args = parser.parse_args()
    live = not args.no_live_grid

    # Case 1: one-shot social life.
    one_shot_params = Params(lifetime_rounds=1)
    os_viewer = (
        SimulationViewer(
            one_shot_params.population_size,
            one_shot_params.generations,
            one_shot_params.lifetime_rounds,
            title="One-shot interaction (Trust learning)",
        )
        if live else None
    )
    one_shot = run_simulation(
        one_shot_params,
        generation_callback=os_viewer.update_generation if os_viewer else None,
        round_callback=os_viewer.update_encounter_round if os_viewer else None,
    )
    summarize(one_shot, "Mostly one-shot interaction")

    # Case 2: repeated social life.
    repeated_params = Params(lifetime_rounds=80)
    rep_viewer = (
        SimulationViewer(
            repeated_params.population_size,
            repeated_params.generations,
            repeated_params.lifetime_rounds,
            title="Repeated interaction (Trust learning)",
        )
        if live else None
    )
    repeated = run_simulation(
        repeated_params,
        generation_callback=rep_viewer.update_generation if rep_viewer else None,
        round_callback=rep_viewer.update_encounter_round if rep_viewer else None,
    )
    summarize(repeated, "Repeated interaction")

    os.makedirs("output", exist_ok=True)

    plot_history(
        one_shot,
        "Mostly one-shot interaction",
        save_prefix="one_shot",
    )
    plot_history(
        repeated,
        "Repeated interaction",
        save_prefix="repeated",
    )

    if live:
        print("\nLive viewer: close plot windows to finish.")
        plt.ioff()
    plt.show()
