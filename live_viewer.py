"""live_viewer.py — Live matplotlib viewers for two-timescale cooperation experiments.

Two classes:

  SimulationViewer
      Single-run viewer: cooperation-over-generations line + encounter matrix.
      For use with the standalone experiment scripts.

  ExperimentViewer
      Multi-model, multi-condition viewer: payoff heatmap + cooperation lines
      + encounter matrix.  For use with experiment_network_diversity.py.

Both share the same callback API so simulation functions need not know which
viewer is attached:

    viewer.update_generation(generation, mean_cooperation)
    viewer.update_encounter_round(generation, round_idx, events)

where ``events`` is a list of ``(i, j, act_i, act_j)`` tuples with
act = 0 (cooperate) or 1 (defect).

Call ``viewer.block_until_closed()`` after all runs to keep windows open.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

COOPERATE = 0
_DRAW_STRIDE_DEFAULT = 4   # redraw encounter matrix every N rounds


# ─────────────────────────────────────────────────────────────────────────────
# SimulationViewer — single model, single run
# ─────────────────────────────────────────────────────────────────────────────

class SimulationViewer:
    """
    Live viewer for a single simulation run.

    Two windows:
      1. Cooperation over generations — single line, updated each generation.
      2. Encounter matrix            — micro snapshot, updated every
                                       ``draw_stride`` rounds.

    Usage::

        viewer = SimulationViewer(num_agents, num_generations, lifetime_rounds,
                                  title="My Run")
        history = run_simulation(
            ...,
            generation_callback=viewer.update_generation,
            round_callback=viewer.update_encounter_round,
        )
        viewer.block_until_closed()
    """

    def __init__(
        self,
        num_agents: int,
        num_generations: int,
        lifetime_rounds: int,
        title: str = "Simulation",
        draw_stride: int = _DRAW_STRIDE_DEFAULT,
    ) -> None:
        self.enabled = True
        self.num_agents = num_agents
        self.num_generations = num_generations
        self.lifetime_rounds = lifetime_rounds
        self.draw_stride = max(1, draw_stride)
        self._title = title
        self._coop_data: list[float] = []

        try:
            plt.ion()

            # cooperation line
            self.fig_line, self.ax_line = plt.subplots(figsize=(9, 4.5))
            (self._coop_line,) = self.ax_line.plot(
                [], [], linewidth=2, color="#1f77b4"
            )
            self.ax_line.set_xlim(0, max(num_generations - 1, 1))
            self.ax_line.set_ylim(-0.02, 1.02)
            self.ax_line.set_xlabel("Generation", fontsize=11)
            self.ax_line.set_ylabel("Mean cooperation", fontsize=11)
            self.ax_line.set_title(title, fontsize=13)
            self.ax_line.grid(alpha=0.3)
            self.fig_line.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)

            # encounter matrix
            self._enc_mat = np.full((num_agents, num_agents), np.nan)
            cmap = plt.colormaps["RdYlGn"].copy()
            cmap.set_bad("#232323")
            self.fig_enc, self.ax_enc = plt.subplots(figsize=(7.5, 6.8))
            self._im_enc = self.ax_enc.imshow(
                self._enc_mat,
                interpolation="nearest",
                cmap=cmap,
                vmin=-1.0,
                vmax=1.0,
            )
            cbar = self.fig_enc.colorbar(self._im_enc, ax=self.ax_enc)
            cbar.set_label(
                "Action toward partner  (Defect = \u22121,  Cooperate = +1)",
                fontsize=10,
            )
            self.ax_enc.set_xlabel("Partner j", fontsize=11)
            self.ax_enc.set_ylabel("Agent i", fontsize=11)
            self.ax_enc.set_title(
                f"{title}\nLive encounter matrix", fontsize=13
            )
            self.fig_enc.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)

        except Exception as exc:
            print(f"Live viewer disabled: {exc}")
            self.enabled = False

    # ── callbacks ────────────────────────────────────────────────────────────

    def update_generation(
        self, generation: int, mean_cooperation: float
    ) -> None:
        if not self.enabled:
            return
        while len(self._coop_data) <= generation:
            self._coop_data.append(np.nan)
        self._coop_data[generation] = mean_cooperation
        self._coop_line.set_data(
            np.arange(len(self._coop_data)), self._coop_data
        )
        self.ax_line.set_title(
            f"{self._title}  —  gen {generation + 1}/{self.num_generations}",
            fontsize=13,
        )
        self.fig_line.canvas.draw_idle()
        plt.pause(0.001)

    def update_encounter_round(
        self,
        generation: int,
        round_idx: int,
        events: list[tuple[int, int, int, int]],
    ) -> None:
        if not self.enabled or round_idx % self.draw_stride != 0:
            return
        self._enc_mat[:, :] = np.nan
        for i, j, act_i, act_j in events:
            self._enc_mat[i, j] = 1.0 if act_i == COOPERATE else -1.0
            self._enc_mat[j, i] = 1.0 if act_j == COOPERATE else -1.0
        self._im_enc.set_data(self._enc_mat)
        self.ax_enc.set_title(
            f"{self._title}  —  "
            f"gen {generation + 1}/{self.num_generations},  "
            f"round {round_idx + 1}/{self.lifetime_rounds}",
            fontsize=12,
        )
        self.fig_enc.canvas.draw_idle()
        plt.pause(0.001)

    def block_until_closed(self) -> None:
        """Turn off interactive mode and block until all windows are closed."""
        if self.enabled:
            plt.ioff()
            plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentViewer — multi-model, multi-condition
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentViewer:
    """
    Live viewer for multi-model, multi-condition experiment runs.

    Three windows:
      1. Payoff heatmap   — model × condition grid, fills as runs complete.
      2. Cooperation plot — one line per model, reset at each condition.
      3. Encounter matrix — micro snapshot, same as SimulationViewer.

    Call ``start_condition(idx)`` at the start of each stranger-fraction
    condition and ``start_model(key)`` before each model run.
    ``update_generation`` and ``update_encounter_round`` use the current model
    context set by ``start_model``, so the same callbacks can be reused
    across all models.

    Usage::

        viewer = ExperimentViewer(model_order, model_labels,
                                  stranger_fractions,
                                  num_agents, num_generations, lifetime_rounds)
        gen_cb   = viewer.update_generation
        round_cb = viewer.update_encounter_round

        for idx, sf in enumerate(stranger_fractions):
            viewer.start_condition(idx)
            for key in model_order:
                viewer.start_model(key)
                payoff = run_model(sf,
                                   generation_callback=gen_cb,
                                   round_callback=round_cb)
                viewer.update_payoff_cell(key, idx, payoff)

        viewer.block_until_closed()
    """

    _COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    def __init__(
        self,
        model_order: list[str],
        model_labels: dict[str, str],
        stranger_fractions: list[float],
        num_agents: int,
        num_generations: int,
        lifetime_rounds: int,
        draw_stride: int = _DRAW_STRIDE_DEFAULT,
    ) -> None:
        self.enabled = True
        self.model_order = model_order
        self.model_labels = model_labels
        self.stranger_fractions = stranger_fractions
        self.num_agents = num_agents
        self.num_generations = num_generations
        self.lifetime_rounds = lifetime_rounds
        self.draw_stride = max(1, draw_stride)
        self._current_model_key: str = model_order[0]
        self._current_condition_idx: int = 0
        self._line_data: dict[str, list[float]] = {k: [] for k in model_order}

        try:
            plt.ion()

            # 1. Payoff heatmap
            self._payoff_mat = np.full(
                (len(model_order), len(stranger_fractions)), np.nan
            )
            self._cell_texts: dict[tuple[int, int], Any] = {}
            self.fig_grid, self.ax_grid = plt.subplots(figsize=(10, 4.6))
            self._im_grid = self.ax_grid.imshow(
                self._payoff_mat,
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=0.0,
                vmax=200.0,
            )
            cbar_grid = self.fig_grid.colorbar(
                self._im_grid, ax=self.ax_grid
            )
            cbar_grid.set_label("Mean payoff", fontsize=11)
            self.ax_grid.set_xticks(range(len(stranger_fractions)))
            self.ax_grid.set_xticklabels(
                [f"{int(sf * 100)}%" for sf in stranger_fractions]
            )
            self.ax_grid.set_yticks(range(len(model_order)))
            self.ax_grid.set_yticklabels(
                [model_labels[k] for k in model_order]
            )
            self.ax_grid.set_xlabel("Stranger encounters", fontsize=11)
            self.ax_grid.set_ylabel("Model", fontsize=11)
            self.ax_grid.set_title("Live payoff grid", fontsize=13)
            self.fig_grid.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)

            # 2. Cooperation over generations (one line per model)
            colors = (self._COLORS * 2)[: len(model_order)]
            self.fig_line, self.ax_line = plt.subplots(figsize=(10, 4.8))
            self._line_handles: dict[str, Any] = {}
            for key, color in zip(model_order, colors):
                (handle,) = self.ax_line.plot(
                    [], [], linewidth=2, label=model_labels[key], color=color
                )
                self._line_handles[key] = handle
            self.ax_line.set_xlim(0, max(num_generations - 1, 1))
            self.ax_line.set_ylim(-0.02, 1.02)
            self.ax_line.set_xlabel("Generation", fontsize=11)
            self.ax_line.set_ylabel("Mean cooperation", fontsize=11)
            self.ax_line.set_title("Live cooperation by generation", fontsize=13)
            self.ax_line.grid(alpha=0.3)
            self.ax_line.legend(loc="lower right", fontsize=10)
            self.fig_line.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)

            # 3. Encounter matrix
            self._enc_mat = np.full((num_agents, num_agents), np.nan)
            cmap = plt.colormaps["RdYlGn"].copy()
            cmap.set_bad("#232323")
            self.fig_enc, self.ax_enc = plt.subplots(figsize=(7.5, 6.8))
            self._im_enc = self.ax_enc.imshow(
                self._enc_mat,
                interpolation="nearest",
                cmap=cmap,
                vmin=-1.0,
                vmax=1.0,
            )
            cbar_enc = self.fig_enc.colorbar(self._im_enc, ax=self.ax_enc)
            cbar_enc.set_label(
                "Action toward partner  (Defect = \u22121,  Cooperate = +1)",
                fontsize=10,
            )
            self.ax_enc.set_xlabel("Partner j", fontsize=11)
            self.ax_enc.set_ylabel("Agent i", fontsize=11)
            self.ax_enc.set_title("Live encounter matrix", fontsize=13)
            self.fig_enc.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)

        except Exception as exc:
            print(f"Live viewer disabled: {exc}")
            self.enabled = False

    # ── context setters ──────────────────────────────────────────────────────

    def start_condition(self, condition_idx: int) -> None:
        if not self.enabled:
            return
        self._current_condition_idx = condition_idx
        sf = self.stranger_fractions[condition_idx]
        for k in self.model_order:
            self._line_data[k] = []
            self._line_handles[k].set_data([], [])
        self.ax_line.set_title(
            "Live cooperation by generation\n"
            f"stranger_fraction={sf:.2f}  ({int(sf * 100)}% random)",
            fontsize=13,
        )
        self.fig_line.canvas.draw_idle()
        plt.pause(0.001)

    def start_model(self, model_key: str) -> None:
        if not self.enabled:
            return
        self._current_model_key = model_key
        sf = self.stranger_fractions[self._current_condition_idx]
        self._enc_mat[:, :] = np.nan
        self._im_enc.set_data(self._enc_mat)
        self.ax_enc.set_title(
            "Live encounter matrix\n"
            f"{self.model_labels[model_key]},  stranger_fraction={sf:.2f}",
            fontsize=13,
        )
        self.fig_enc.canvas.draw_idle()
        plt.pause(0.001)

    # ── update callbacks (same signature as SimulationViewer) ────────────────

    def update_generation(
        self, generation: int, mean_cooperation: float
    ) -> None:
        if not self.enabled:
            return
        key = self._current_model_key
        data = self._line_data[key]
        while len(data) <= generation:
            data.append(np.nan)
        data[generation] = mean_cooperation
        self._line_handles[key].set_data(np.arange(len(data)), data)
        self.fig_line.canvas.draw_idle()
        plt.pause(0.001)

    def update_encounter_round(
        self,
        generation: int,
        round_idx: int,
        events: list[tuple[int, int, int, int]],
    ) -> None:
        if not self.enabled or round_idx % self.draw_stride != 0:
            return
        self._enc_mat[:, :] = np.nan
        for i, j, act_i, act_j in events:
            self._enc_mat[i, j] = 1.0 if act_i == COOPERATE else -1.0
            self._enc_mat[j, i] = 1.0 if act_j == COOPERATE else -1.0
        self._im_enc.set_data(self._enc_mat)
        sf = self.stranger_fractions[self._current_condition_idx]
        self.ax_enc.set_title(
            "Live encounter matrix\n"
            f"{self.model_labels[self._current_model_key]},  "
            f"stranger_fraction={sf:.2f},  "
            f"gen {generation + 1}/{self.num_generations},  "
            f"round {round_idx + 1}/{self.lifetime_rounds}",
            fontsize=12,
        )
        self.fig_enc.canvas.draw_idle()
        plt.pause(0.001)

    def update_payoff_cell(
        self,
        model_key: str,
        condition_idx: int,
        payoff: float,
        progress_label: str = "",
    ) -> None:
        if not self.enabled:
            return
        row = self.model_order.index(model_key)
        self._payoff_mat[row, condition_idx] = payoff
        self._im_grid.set_data(self._payoff_mat)
        finite = self._payoff_mat[np.isfinite(self._payoff_mat)]
        if finite.size:
            vmax = max(float(np.max(finite)) * 1.05, 1.0)
            self._im_grid.set_clim(vmin=0.0, vmax=vmax)
        cell_key = (row, condition_idx)
        if cell_key in self._cell_texts:
            self._cell_texts[cell_key].set_text(f"{payoff:.1f}")
        else:
            self._cell_texts[cell_key] = self.ax_grid.text(
                condition_idx,
                row,
                f"{payoff:.1f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )
        title = "Live payoff grid"
        if progress_label:
            title += f"\n{progress_label}"
        self.ax_grid.set_title(title, fontsize=13)
        self.fig_grid.canvas.draw_idle()
        plt.pause(0.001)

    def block_until_closed(self) -> None:
        """Turn off interactive mode and block until all windows are closed."""
        if self.enabled:
            plt.ioff()
            plt.show()
