"""
LoveBugs ABM using mesa‑frames + Polars
------------    def     def __init__(self, n: int, model: "LoveModel"):
        super().__init__(model)
        self += pl.DataFrame(
            {
                "genome": np.random.randint(0, 2**32, size=n, dtype=np.uint32),
                "energy": pl.Series([10.0] * n, dtype=pl.Float32),
                "age": pl.Series([0] * n, dtype=pl.UInt16),
            }
        )elf, n: int, model: "LoveModel"):
        super().__init__(model)
        self += pl.DataFrame(
            {
                "genome": np.random.randint(0, 2**32, size=n, dtype=np.uint32),
                "energy": pl.Series([10.0] * n, dtype=pl.Float32),
                "age": pl.Series([0] * n, dtype=pl.UInt16),
            }
        )-----------------

Minimal, vectorised starter scaffold for an evolutionary sexual‑selection model.

Genome layout (32‑bit unsigned int)
+------------+--------------+----------------+
| Bits 24‑31 | Bits 16‑23   | Bits 0‑15      |
+------------+--------------+----------------+
| Threshold  | Preference   | Display traits |
+------------+--------------+----------------+

Agents accept a mate if *Hamming similarity* between their preference mask and
partner display ≥ their threshold *and* vice‑versa. Offspring inherit a genome
via uniform crossover + point mutation.

Dependencies:
  pip install mesa-frames polars beartype numpy
"""

from __future__ import annotations

import numpy as np
import polars as pl
from mesa_frames import AgentSetPolars, ModelDF

# ── Bit masks & shifts ──────────────────────────────────────────────────────
DISPLAY_MASK = np.uint32(0x0000_FFFF)
PREF_MASK = np.uint32(0x00FF_0000)
BEHAV_MASK = np.uint32(0xFF00_0000)

DISPLAY_SHIFT = 0
PREF_SHIFT = 16
BEHAV_SHIFT = 24

# Probability that any particular bit flips during mutation
MUTATION_RATE = 1e-4
# Energy lost per tick & max lifespan
ENERGY_DECAY = 0.2
MAX_AGE = 100

# ── Helper: fast Hamming similarity on 16‑bit halves ────────────────────────


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # type: ignore
    """Return similarity (16 – Hamming distance) for lower 16 bits."""
    return 16 - np.bit_count((a ^ b) & DISPLAY_MASK)


# ── Agent container ─────────────────────────────────────────────────────────


class LoveBugs(AgentSetPolars):
    """All agents live inside a single Polars DataFrame (vectorised ops)."""

    def __init__(self, n: int, model: LoveModel):
        super().__init__(model)
        self += pl.DataFrame(
            {
                "genome": np.random.randint(0, 2**32, size=n, dtype=np.uint32),
                "energy": pl.Series([10.0] * n, dtype=pl.Float32),
                "age": pl.Series([0] * n, dtype=pl.UInt16),
            }
        )

    # ── Properties for lazy genome slices ──────────────────────────────────
    @property
    def display(self):
        return (pl.col("genome") & DISPLAY_MASK).alias("display")

    @property
    def preference(self):
        return ((pl.col("genome") & PREF_MASK) >> PREF_SHIFT).alias("preference")

    @property
    def threshold(self):
        return ((pl.col("genome") & BEHAV_MASK) >> BEHAV_SHIFT).alias("threshold")

    # ── Main timestep ──────────────────────────────────────────────────────
    def step(self) -> None:
        self.courtship()
        self.metabolism()
        self.age_and_die()

    # ── Vectorised sub‑routines ────────────────────────────────────────────
    def courtship(self) -> None:
        """Randomly pair agents, create offspring when mutual acceptance."""
        n = len(self)
        if n < 2:
            return

        partners = np.random.permutation(n)  # random partner for each agent

        genomes_self = self.agents["genome"].to_numpy(dtype=np.uint32)
        genomes_partner = genomes_self[partners]

        # Slice genomes into fields
        disp_self = genomes_self & DISPLAY_MASK
        pref_self = (genomes_self & PREF_MASK) >> PREF_SHIFT
        thr_self = (genomes_self & BEHAV_MASK) >> BEHAV_SHIFT

        disp_partner = genomes_partner & DISPLAY_MASK
        pref_partner = (genomes_partner & PREF_MASK) >> PREF_SHIFT
        thr_partner = (genomes_partner & BEHAV_MASK) >> BEHAV_SHIFT

        # Similarity scores (0‑16)
        sim_self = 16 - np.bit_count(disp_partner ^ pref_self)
        sim_partner = 16 - np.bit_count(disp_self ^ pref_partner)

        accepted = (sim_self >= thr_self) & (sim_partner >= thr_partner)
        if not accepted.any():
            return

        idx = np.where(accepted)[0]
        parents_a = genomes_self[idx]
        parents_b = genomes_partner[idx]

        # Uniform crossover via random mask
        mask = np.random.randint(0, 2**32, size=len(idx))
        offspring_genomes = (parents_a & mask) | (parents_b & ~mask)

        # Bit‑flip mutation
        if MUTATION_RATE > 0:
            flips = np.random.binomial(1, MUTATION_RATE, size=(len(idx), 32)).astype(bool)
            rows, bits = np.where(flips)
            if rows.size:
                offspring_genomes[rows] ^= np.left_shift(np.uint32(1), bits.astype(np.uint32))

        # Append offspring
        self += pl.DataFrame(
            {
                "genome": offspring_genomes,
                "energy": pl.Series([10.0] * len(idx), dtype=pl.Float32),
                "age": pl.Series([0] * len(idx), dtype=pl.UInt16),
            }
        )

    def metabolism(self) -> None:
        self["energy"] -= ENERGY_DECAY
        # remove depleted agents
        self.select(self.energy > 0)

    def age_and_die(self) -> None:
        self["age"] += 1
        self.select(self.age < MAX_AGE)


# ── Model container ─────────────────────────────────────────────────────────


class LoveModel(ModelDF):
    def __init__(self, n_agents: int = 1000):
        super().__init__()
        self.agents += LoveBugs(n_agents, self)

    def step(self):
        self.agents.do("step")

    def run(self, n_steps: int = 100):
        for _ in range(n_steps):
            self.step()


if __name__ == "__main__":
    model = LoveModel(5000)
    model.run(200)
    print("Final population:", len(model.agents))
