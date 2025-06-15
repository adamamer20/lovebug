"""LoveBugs ABM using Mesa-Frames + Polars.

Minimal, vectorised starter scaffold for an evolutionary sexual-selection
model.

Genome layout (32-bit unsigned int)
+------------+--------------+----------------+
| Bits 24-31 | Bits 16-23   | Bits 0-15      |
+------------+--------------+----------------+
| Threshold  | Preference   | Display traits |
+------------+--------------+----------------+

Agents accept a mate if the Hamming similarity between their preference mask
and their partner's display is at least their threshold, and vice versa.
Offspring genomes are produced via uniform crossover with per-bit mutation.

Example
-------
>>> from lovebug import LoveModel
>>> model = LoveModel(1000)
>>> model.run(10)

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

# ── Cultural evolution parameters ─────────────────────────────────────────
CULTURE_WEIGHT = 0.5
P_LEARN = 0.1
prestige_cut = 0.2
conformist_k = 3
USE_PRESTIGE = True  # False -> conformist rule

# ── Helper: fast Hamming similarity on 16‑bit halves ────────────────────────


def _bit_count_u32(values: np.ndarray) -> np.ndarray:
    """Return the number of set bits for each 32-bit integer."""
    return pl.Series(values.astype(np.uint32), dtype=pl.UInt32).bitwise_count_ones().to_numpy()


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # type: ignore
    """Return similarity (16 – Hamming distance) for lower 16 bits."""
    diff = (
        (pl.Series(a.astype(np.uint32), dtype=pl.UInt32) ^ pl.Series(b.astype(np.uint32), dtype=pl.UInt32))
        & DISPLAY_MASK
    ).bitwise_count_ones()
    return (16 - diff).to_numpy()


def get_effective_pref(df: pl.DataFrame) -> np.ndarray:
    """Return effective preference blending genetic and cultural layers."""
    gene = (df["genome"].to_numpy() & PREF_MASK) >> PREF_SHIFT
    culture = (
        df.get_column("pref_culture").to_numpy().astype(np.uint8)
        if "pref_culture" in df.columns
        else gene.astype(np.uint8)
    )
    if CULTURE_WEIGHT <= 0:
        return gene
    if CULTURE_WEIGHT >= 1:
        return culture
    choose = np.random.random(len(df)) < CULTURE_WEIGHT
    out = gene.copy()
    out[choose] = culture[choose]
    return out


# ── Agent container ─────────────────────────────────────────────────────────


class LoveBugs(AgentSetPolars):
    """All agents live inside a single Polars DataFrame (vectorised ops)."""

    def __init__(self, n: int, model: LoveModel):
        super().__init__(model)
        genomes = np.random.randint(0, 2**32, size=n, dtype=np.uint32)
        self.prev_matings = np.zeros(n, dtype=np.uint16)
        self += pl.DataFrame(
            {
                "genome": genomes,
                "energy": pl.Series([10.0] * n, dtype=pl.Float32),
                "age": pl.Series([0] * n, dtype=pl.UInt16),
                "pref_culture": ((genomes & PREF_MASK) >> PREF_SHIFT).astype(np.uint8),
            }
        )

    # ── Properties for lazy genome slices ──────────────────────────────────
    @property
    def display(self):
        return (pl.col("genome") & DISPLAY_MASK).alias("display")

    @property
    def preference(self):
        return ((pl.col("genome") & PREF_MASK) // (1 << PREF_SHIFT)).alias("preference")

    @property
    def threshold(self):
        return ((pl.col("genome") & BEHAV_MASK) // (1 << BEHAV_SHIFT)).alias("threshold")

    # ── Main timestep ──────────────────────────────────────────────────────
    def step(self) -> None:
        self.courtship()
        self.social_learning()
        self.metabolism()
        self.age_and_die()

    # ── Vectorised sub‑routines ────────────────────────────────────────────
    def courtship(self) -> None:
        """Randomly pair agents, create offspring when mutual acceptance."""
        n = len(self)
        if n < 2:
            return

        partners = np.random.permutation(n)  # random partner for each agent

        genomes_self = self.agents["genome"].to_numpy().astype(np.uint32)
        genomes_partner = genomes_self[partners]
        self.prev_matings = np.zeros(n, dtype=np.uint16)

        # Slice genomes into fields
        disp_self = genomes_self & DISPLAY_MASK
        pref_self = get_effective_pref(self.agents)
        thr_self = (genomes_self & BEHAV_MASK) >> BEHAV_SHIFT

        disp_partner = genomes_partner & DISPLAY_MASK
        pref_partner = pref_self[partners]
        thr_partner = (genomes_partner & BEHAV_MASK) >> BEHAV_SHIFT

        # Similarity scores (0‑16)
        sim_self = hamming_similarity(disp_partner, pref_self)
        sim_partner = hamming_similarity(disp_self, pref_partner)

        accepted = (sim_self >= thr_self) & (sim_partner >= thr_partner)
        if not accepted.any():
            return

        idx = np.where(accepted)[0]
        partner_idx = partners[idx]
        parents_a = genomes_self[idx]
        parents_b = genomes_partner[idx]

        np.add.at(self.prev_matings, idx, 1)
        np.add.at(self.prev_matings, partner_idx, 1)

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
                "pref_culture": ((offspring_genomes & PREF_MASK) >> PREF_SHIFT).astype(np.uint8),
            }
        )

    def social_learning(self) -> None:
        """Learners copy cultural preferences from teachers."""
        n = len(self)
        if n == 0 or P_LEARN <= 0:
            return

        learners = np.random.random(n) < P_LEARN
        if not learners.any():
            return

        culture = self.agents["pref_culture"].to_numpy().astype(np.uint8)

        if USE_PRESTIGE:
            top_n = max(1, int(prestige_cut * n))
            ranked = np.argsort(-self.prev_matings)[:top_n]
            teachers = np.random.choice(ranked, learners.sum(), replace=True)
            culture[learners] = culture[teachers]
        else:
            k = max(1, int(conformist_k))
            choices = np.random.randint(0, n, size=(learners.sum(), k))
            prefs = culture[choices]
            bits = (prefs[:, :, None] >> np.arange(8, dtype=np.uint8)) & 1
            sums = bits.sum(axis=1)
            majority = (sums >= (k / 2)).astype(np.uint8)
            teacher_pref = np.left_shift(majority, np.arange(8, dtype=np.uint8)).sum(axis=1)
            culture[learners] = teacher_pref

        self["pref_culture"] = pl.Series(culture, dtype=pl.UInt8)

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
        df = self.agents._agentsets[0].agents
        gene = ((df["genome"] & PREF_MASK) // (1 << PREF_SHIFT)).cast(pl.UInt32)
        culture = df["pref_culture"].cast(pl.UInt32)
        dist = ((gene ^ culture) & 0xFF).bitwise_count_ones().mean()
        print(f"Mean gene-culture Hamming distance: {float(dist):.2f}")

    def run(self, n_steps: int = 100):
        for _ in range(n_steps):
            self.step()


if __name__ == "__main__":
    model = LoveModel(5000)
    model.run(200)
    print("Final population:", len(model.agents))
