"""
Experiment: does reputation + partner choice dominate in larger,
more diverse networks?

The hypothesis: in a small ring (k=2 fixed neighbours) agents always
interact with the same two people. Personal Q-history is sufficient and
reputation / partner choice add little. As stranger exposure increases
(larger neighbourhood k or more random encounters), reputation and
partner choice become the primary cooperation-enabling mechanism and
payoffs in the extended model should rise relative to the simpler models.

Experimental design
-------------------
We vary `stranger_fraction` — the probability that each interaction is
with a randomly chosen agent rather than a fixed ring neighbour.

  stranger_fraction = 0.0  -> pure ring (always same 2 neighbours)
  stranger_fraction = 0.5  -> half random encounters
  stranger_fraction = 1.0  -> fully random encounters (anonymous market)

All three models are run at each stranger_fraction level:
  1. Trust learning        (two_timescale_reciprocity)
  2. Basic Q-learning      (two_timescale_q_learning)
  3. Extended model        (reputation + partner choice + forgiveness)

The key output is final-generation mean payoff per model per condition.
"""

import numpy as np
import matplotlib.pyplot as plt
import dataclasses
import argparse
from collections.abc import Callable
from live_viewer import ExperimentViewer

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────
COOPERATE = 0
DEFECT = 1
BENEFIT = 3.0
COST = 1.0
NUM_AGENTS = 100
NUM_GENERATIONS = 80        # reduced for speed across multiple conditions
LIFETIME_ROUNDS = 40
MUTATION_STD = 0.1
SEED = 42

STRANGER_FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

MODEL_ORDER = ["trust", "q_learning", "extended"]
MODEL_LABELS = {
    "trust": "Trust",
    "q_learning": "Q-learning",
    "extended": "Extended",
}


# ─────────────────────────────────────────────────────────────────────────────
# Network helpers
# ─────────────────────────────────────────────────────────────────────────────

def ring_neighbors(num_agents: int) -> list[list[int]]:
    return [
        [(i - 1) % num_agents, (i + 1) % num_agents]
        for i in range(num_agents)
    ]


def sample_partners(
    i: int,
    fixed_neighbors: list[list[int]],
    stranger_fraction: float,
    num_agents: int,
    rng: np.random.Generator,
) -> list[int]:
    """
    Return interaction partners for agent i this round.
    With probability `stranger_fraction` each slot is a random stranger,
    otherwise it is the fixed ring neighbour.
    """
    partners = []
    for nb in fixed_neighbors[i]:
        if rng.random() < stranger_fraction:
            # Random stranger (excluding self)
            stranger = int(rng.integers(0, num_agents - 1))
            if stranger >= i:
                stranger += 1
            partners.append(stranger)
        else:
            partners.append(nb)
    return partners


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — Trust learning
# ─────────────────────────────────────────────────────────────────────────────

def run_trust(
    stranger_fraction: float,
    rng: np.random.Generator,
    generation_callback: Callable[[int, float], None] | None = None,
    round_callback: Callable[[int, int, list[tuple[int, int, int, int]]], None]
    | None = None,
) -> float:
    """
    Return final-generation mean payoff for the trust-learning model
    at a given stranger_fraction.
    """
    genes = {
        "trust_prior": rng.normal(0, 1, NUM_AGENTS),
        "learning_rate": np.abs(rng.normal(0.2, 0.1, NUM_AGENTS)),
        "responsiveness": rng.normal(1.0, 0.5, NUM_AGENTS),
    }
    neighbors = ring_neighbors(NUM_AGENTS)

    for generation in range(NUM_GENERATIONS):
        learned_trust = np.zeros((NUM_AGENTS, NUM_AGENTS))
        payoffs = np.zeros(NUM_AGENTS)
        coop_actions = 0
        total_actions = 0

        for _ in range(LIFETIME_ROUNDS):
            round_idx = _
            round_events: list[tuple[int, int, int, int]] = []
            for i in range(NUM_AGENTS):
                for j in sample_partners(
                    i, neighbors, stranger_fraction, NUM_AGENTS, rng
                ):
                    score_i = (
                        genes["trust_prior"][i]
                        + genes["responsiveness"][i] * learned_trust[i, j]
                    )
                    score_j = (
                        genes["trust_prior"][j]
                        + genes["responsiveness"][j] * learned_trust[j, i]
                    )
                    coop_i = score_i > 0.0
                    coop_j = score_j > 0.0
                    coop_actions += int(coop_i) + int(coop_j)
                    total_actions += 2
                    round_events.append(
                        (i, j, int(coop_i), int(coop_j))
                    )

                    if coop_i:
                        payoffs[i] -= COST
                        payoffs[j] += BENEFIT
                    if coop_j:
                        payoffs[j] -= COST
                        payoffs[i] += BENEFIT

                    target_i = 1.0 if coop_j else -1.0
                    target_j = 1.0 if coop_i else -1.0
                    alpha_i = genes["learning_rate"][i]
                    alpha_j = genes["learning_rate"][j]
                    learned_trust[i, j] += alpha_i * (
                        target_i - learned_trust[i, j]
                    )
                    learned_trust[j, i] += alpha_j * (
                        target_j - learned_trust[j, i]
                    )

            if round_callback is not None:
                round_callback(generation, round_idx, round_events)

        if generation_callback is not None and total_actions > 0:
            generation_callback(generation, coop_actions / total_actions)

        # Reproduction
        shifted = payoffs - payoffs.min() + 0.1
        weights = shifted / shifted.sum()
        parents = rng.choice(NUM_AGENTS, size=NUM_AGENTS, p=weights)

        for key in genes:
            genes[key] = (
                genes[key][parents]
                + rng.normal(0, MUTATION_STD, NUM_AGENTS)
            )
        genes["learning_rate"] = np.clip(genes["learning_rate"], 0.01, 1.0)

    return float(np.mean(payoffs))


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — Basic Q-learning
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class QAgent:
    """Minimal Q-learning agent."""
    exploration_rate: float
    learning_rate: float
    discount_factor: float
    initial_q_bias: float
    Q: dict = dataclasses.field(default_factory=dict)
    payoff: float = 0.0

    def get_q(self, pid: int) -> tuple[float, float]:
        q_c = self.Q.get((pid, COOPERATE), self.initial_q_bias)
        q_d = self.Q.get((pid, DEFECT), self.initial_q_bias)
        return q_c, q_d

    def act(self, pid: int, rng: np.random.Generator) -> int:
        if rng.random() < self.exploration_rate:
            return int(rng.integers(0, 2))
        q_c, q_d = self.get_q(pid)
        return COOPERATE if q_c >= q_d else DEFECT

    def learn(
        self, pid: int, action: int, reward: float, next_max_q: float
    ) -> None:
        q_c, q_d = self.get_q(pid)
        cur = q_c if action == COOPERATE else q_d
        new = cur + self.learning_rate * (
            reward + self.discount_factor * next_max_q - cur
        )
        self.Q[(pid, action)] = new

    def reset(self) -> None:
        self.Q.clear()
        self.payoff = 0.0


def _q_interact(
    a: QAgent, i: int, b: QAgent, j: int, rng: np.random.Generator
) -> tuple[int, int, int]:
    act_a = a.act(j, rng)
    act_b = b.act(i, rng)
    r_a = 0.0
    r_b = 0.0
    if act_a == COOPERATE:
        r_a -= COST
        r_b += BENEFIT
    if act_b == COOPERATE:
        r_b -= COST
        r_a += BENEFIT
    a.payoff += r_a
    b.payoff += r_b
    next_q_a = max(a.get_q(j))
    next_q_b = max(b.get_q(i))
    a.learn(j, act_a, r_a, next_q_a)
    b.learn(i, act_b, r_b, next_q_b)
    return int(act_a == COOPERATE) + int(act_b == COOPERATE), act_a, act_b


def run_q_learning(
    stranger_fraction: float,
    rng: np.random.Generator,
    generation_callback: Callable[[int, float], None] | None = None,
    round_callback: Callable[[int, int, list[tuple[int, int, int, int]]], None]
    | None = None,
) -> float:
    agents = [
        QAgent(
            exploration_rate=rng.uniform(0.1, 0.5),
            learning_rate=rng.uniform(0.1, 0.5),
            discount_factor=rng.uniform(0.5, 0.95),
            initial_q_bias=rng.normal(0, 1),
        )
        for _ in range(NUM_AGENTS)
    ]
    neighbors = ring_neighbors(NUM_AGENTS)

    for generation in range(NUM_GENERATIONS):
        coop_actions = 0
        total_actions = 0
        for _ in range(LIFETIME_ROUNDS):
            round_idx = _
            round_events: list[tuple[int, int, int, int]] = []
            for i in range(NUM_AGENTS):
                for j in sample_partners(
                    i, neighbors, stranger_fraction, NUM_AGENTS, rng
                ):
                    coop_count, act_i, act_j = _q_interact(
                        agents[i], i, agents[j], j, rng
                    )
                    coop_actions += coop_count
                    total_actions += 2
                    round_events.append((i, j, act_i, act_j))

            if round_callback is not None:
                round_callback(generation, round_idx, round_events)

        if generation_callback is not None and total_actions > 0:
            generation_callback(generation, coop_actions / total_actions)

        payoffs = np.array([a.payoff for a in agents])
        shifted = payoffs - payoffs.min() + 0.1
        weights = shifted / shifted.sum()
        parents = rng.choice(NUM_AGENTS, size=NUM_AGENTS, p=weights)
        last_payoffs_q = payoffs

        def _clamp(v: float, lo: float, hi: float) -> float:
            return float(np.clip(v + rng.normal(0, MUTATION_STD), lo, hi))

        new_agents = []
        for idx in parents:
            p = agents[idx]
            new_agents.append(
                QAgent(
                    exploration_rate=_clamp(p.exploration_rate, 0.0, 1.0),
                    learning_rate=_clamp(p.learning_rate, 0.01, 1.0),
                    discount_factor=_clamp(p.discount_factor, 0.0, 1.0),
                    initial_q_bias=p.initial_q_bias
                    + rng.normal(0, MUTATION_STD),
                )
            )
        agents = new_agents
        for a in agents:
            a.reset()

    return float(np.mean(last_payoffs_q))


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — Extended (reputation + partner choice + forgiveness)
# ─────────────────────────────────────────────────────────────────────────────

_rep: np.ndarray | None = None


def _rep_init(n: int) -> None:
    global _rep
    _rep = np.zeros(n)


def _rep_get(pid: int) -> float:
    assert _rep is not None
    return float(_rep[pid])


def _rep_update(pid: int, cooperated: bool) -> None:
    assert _rep is not None
    delta = 0.1 if cooperated else -0.1
    _rep[pid] = float(np.clip(_rep[pid] + delta, -1.0, 1.0))


BETRAYAL_PENALTY = 0.5


@dataclasses.dataclass
class ExtAgent:
    exploration_rate: float
    learning_rate: float
    discount_factor: float
    initial_q_bias: float
    rejection_threshold: float
    forgiveness_rate: float
    reputation_weight: float
    Q: dict = dataclasses.field(default_factory=dict)
    betrayal: dict = dataclasses.field(default_factory=dict)
    payoff: float = 0.0

    def _prior(self, pid: int) -> float:
        return (
            self.initial_q_bias + self.reputation_weight * _rep_get(pid)
        )

    def get_q(self, pid: int) -> tuple[float, float]:
        prior = self._prior(pid)
        pen = self.betrayal.get(pid, 0.0)
        q_c = self.Q.get((pid, COOPERATE), prior) - pen
        q_d = self.Q.get((pid, DEFECT), prior)
        return q_c, q_d

    def act(self, pid: int, rng: np.random.Generator) -> int:
        if rng.random() < self.exploration_rate:
            return int(rng.integers(0, 2))
        q_c, q_d = self.get_q(pid)
        return COOPERATE if q_c >= q_d else DEFECT

    def accepts(self, pid: int) -> bool:
        return _rep_get(pid) >= self.rejection_threshold

    def learn(
        self, pid: int, action: int, reward: float, next_max_q: float
    ) -> None:
        q_c, q_d = self.get_q(pid)
        cur = q_c if action == COOPERATE else q_d
        new = cur + self.learning_rate * (
            reward + self.discount_factor * next_max_q - cur
        )
        self.Q[(pid, action)] = new

    def note_betrayal(self, pid: int) -> None:
        self.betrayal[pid] = self.betrayal.get(pid, 0.0) + BETRAYAL_PENALTY

    def forgive(self, pid: int) -> None:
        if pid in self.betrayal:
            self.betrayal[pid] *= (1.0 - self.forgiveness_rate)
            if self.betrayal[pid] < 1e-4:
                del self.betrayal[pid]

    def reset(self) -> None:
        self.Q.clear()
        self.betrayal.clear()
        self.payoff = 0.0


def _ext_interact(
    a: ExtAgent, i: int, b: ExtAgent, j: int, rng: np.random.Generator
) -> tuple[int, int | None, int | None]:
    if not (a.accepts(j) and b.accepts(i)):
        if not a.accepts(j):
            _rep_update(j, cooperated=False)
        if not b.accepts(i):
            _rep_update(i, cooperated=False)
        return 0, None, None

    act_a = a.act(j, rng)
    act_b = b.act(i, rng)

    r_a = 0.0
    r_b = 0.0
    if act_a == COOPERATE:
        r_a -= COST
        r_b += BENEFIT
    if act_b == COOPERATE:
        r_b -= COST
        r_a += BENEFIT

    a.payoff += r_a
    b.payoff += r_b

    _rep_update(i, cooperated=(act_a == COOPERATE))
    _rep_update(j, cooperated=(act_b == COOPERATE))

    if act_b == DEFECT:
        a.note_betrayal(j)
    else:
        a.forgive(j)

    if act_a == DEFECT:
        b.note_betrayal(i)
    else:
        b.forgive(i)

    a.learn(j, act_a, r_a, max(a.get_q(j)))
    b.learn(i, act_b, r_b, max(b.get_q(i)))
    return int(act_a == COOPERATE) + int(act_b == COOPERATE), act_a, act_b


def run_extended(
    stranger_fraction: float,
    rng: np.random.Generator,
    generation_callback: Callable[[int, float], None] | None = None,
    round_callback: Callable[[int, int, list[tuple[int, int, int, int]]], None]
    | None = None,
) -> float:
    _rep_init(NUM_AGENTS)
    agents = [
        ExtAgent(
            exploration_rate=rng.uniform(0.1, 0.5),
            learning_rate=rng.uniform(0.1, 0.5),
            discount_factor=rng.uniform(0.5, 0.95),
            initial_q_bias=rng.normal(0, 1),
            rejection_threshold=rng.uniform(-0.5, 0.5),
            forgiveness_rate=rng.uniform(0.0, 0.5),
            reputation_weight=rng.uniform(0.0, 1.0),
        )
        for _ in range(NUM_AGENTS)
    ]
    neighbors = ring_neighbors(NUM_AGENTS)

    for generation in range(NUM_GENERATIONS):
        _rep_init(NUM_AGENTS)
        coop_actions = 0
        total_actions = 0

        for _ in range(LIFETIME_ROUNDS):
            round_idx = _
            round_events: list[tuple[int, int, int, int]] = []
            for i in range(NUM_AGENTS):
                for j in sample_partners(
                    i, neighbors, stranger_fraction, NUM_AGENTS, rng
                ):
                    coop_count, act_i, act_j = _ext_interact(
                        agents[i], i, agents[j], j, rng
                    )
                    coop_actions += coop_count
                    total_actions += 2
                    if act_i is not None and act_j is not None:
                        round_events.append((i, j, act_i, act_j))

            if round_callback is not None:
                round_callback(generation, round_idx, round_events)

        if generation_callback is not None and total_actions > 0:
            generation_callback(generation, coop_actions / total_actions)

        payoffs = np.array([a.payoff for a in agents])
        shifted = payoffs - payoffs.min() + 0.1
        weights = shifted / shifted.sum()
        parents = rng.choice(NUM_AGENTS, size=NUM_AGENTS, p=weights)
        last_payoffs_ext = payoffs

        def _clamp(v: float, lo: float, hi: float) -> float:
            return float(np.clip(v + rng.normal(0, MUTATION_STD), lo, hi))

        new_agents = []
        for idx in parents:
            p = agents[idx]
            new_agents.append(
                ExtAgent(
                    exploration_rate=_clamp(p.exploration_rate, 0.0, 1.0),
                    learning_rate=_clamp(p.learning_rate, 0.01, 1.0),
                    discount_factor=_clamp(p.discount_factor, 0.0, 1.0),
                    initial_q_bias=p.initial_q_bias
                    + rng.normal(0, MUTATION_STD),
                    rejection_threshold=_clamp(
                        p.rejection_threshold, -1.0, 1.0
                    ),
                    forgiveness_rate=_clamp(p.forgiveness_rate, 0.0, 1.0),
                    reputation_weight=_clamp(p.reputation_weight, 0.0, 2.0),
                )
            )
        agents = new_agents
        for a in agents:
            a.reset()

    return float(np.mean(last_payoffs_ext))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(live_view: bool = False) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {
        "trust": [],
        "q_learning": [],
        "extended": [],
    }

    viewer = (
        ExperimentViewer(
            MODEL_ORDER, MODEL_LABELS, STRANGER_FRACTIONS,
            NUM_AGENTS, NUM_GENERATIONS, LIFETIME_ROUNDS,
        )
        if live_view
        else None
    )
    gen_cb = viewer.update_generation if viewer is not None else None
    round_cb = viewer.update_encounter_round if viewer is not None else None

    for idx, sf in enumerate(STRANGER_FRACTIONS):
        pct = int(sf * 100)
        print(f"  stranger_fraction={sf:.2f}  ({pct}% random encounters)")
        if viewer is not None:
            viewer.start_condition(idx)

        rng = np.random.default_rng(seed=SEED)
        if viewer is not None:
            viewer.start_model("trust")
        p_trust = run_trust(
            sf, rng, generation_callback=gen_cb, round_callback=round_cb
        )
        results["trust"].append(p_trust)
        print(f"    Trust learning:  payoff={p_trust:.1f}")
        if viewer is not None:
            viewer.update_payoff_cell(
                "trust", idx, p_trust,
                f"condition {idx + 1}/{len(STRANGER_FRACTIONS)}",
            )

        rng = np.random.default_rng(seed=SEED)
        if viewer is not None:
            viewer.start_model("q_learning")
        p_q = run_q_learning(
            sf, rng, generation_callback=gen_cb, round_callback=round_cb
        )
        results["q_learning"].append(p_q)
        print(f"    Q-learning:      payoff={p_q:.1f}")
        if viewer is not None:
            viewer.update_payoff_cell(
                "q_learning", idx, p_q,
                f"condition {idx + 1}/{len(STRANGER_FRACTIONS)}",
            )

        rng = np.random.default_rng(seed=SEED)
        if viewer is not None:
            viewer.start_model("extended")
        p_ext = run_extended(
            sf, rng, generation_callback=gen_cb, round_callback=round_cb
        )
        results["extended"].append(p_ext)
        print(f"    Extended:        payoff={p_ext:.1f}")
        if viewer is not None:
            viewer.update_payoff_cell(
                "extended", idx, p_ext,
                f"condition {idx + 1}/{len(STRANGER_FRACTIONS)}",
            )

    return results


def plot_results(results: dict[str, list[float]]) -> None:
    x = [int(sf * 100) for sf in STRANGER_FRACTIONS]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        x,
        results["trust"],
        "o-",
        linewidth=2,
        markersize=8,
        label="Trust learning",
    )
    ax.plot(
        x,
        results["q_learning"],
        "s-",
        linewidth=2,
        markersize=8,
        label="Basic Q-learning",
    )
    ax.plot(
        x,
        results["extended"],
        "^-",
        linewidth=2,
        markersize=8,
        label="Extended (rep + choice + forgiveness)",
    )

    ax.set_xlabel("Stranger encounters (%)", fontsize=13)
    ax.set_ylabel("Final mean payoff", fontsize=13)
    ax.set_title(
        "Payoff vs stranger exposure across three models",
        fontsize=14,
    )
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig("output/experiment_network_diversity.png", dpi=150)
    print(
        "\nChart saved: output/experiment_network_diversity.png"
    )


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(
        description="Run payoff-vs-stranger-exposure experiment"
    )
    parser.add_argument(
        "--no-live-grid",
        action="store_true",
        help="Disable live heatmap/cooperation windows",
    )
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    print("Experiment: payoff vs stranger exposure")
    print("=" * 45)
    live_view_enabled = not args.no_live_grid
    results = run_experiment(live_view=live_view_enabled)

    print("\nSummary table")
    print(
        f"{'Strangers':>10}  "
        f"{'Trust':>10}  "
        f"{'Q-learning':>12}  "
        f"{'Extended':>10}"
    )
    print("-" * 50)
    for i, sf in enumerate(STRANGER_FRACTIONS):
        print(
            f"{int(sf*100):>9}%  "
            f"{results['trust'][i]:>10.1f}  "
            f"{results['q_learning'][i]:>12.1f}  "
            f"{results['extended'][i]:>10.1f}"
        )

    plot_results(results)

    if live_view_enabled:
        print("\nLive grid enabled: close plot windows to finish.")
        plt.ioff()
        plt.show()
