"""Sweepable case-config dataclass for GC-polyc (tier G1+).

All physical inputs per CONSTRAINTS.md §1 and §4. No model outputs here.
Units: c_b in mol/m³ (1 mM = 1.0), L in m, potentials in V, C_H in F/m².
"""
from dataclasses import dataclass
import numpy as np
from src import C_H_DEFAULT


@dataclass
class CaseConfig:
    """Sweepable inputs for one 2-D PB solve per CONSTRAINTS.md §4.

    E_pzc_1, E_pzc_2: PZC of facets 1 and 2 [V vs SHE].
    c_b: bulk concentration [mol/m³] (1 mM = 1.0 mol/m³).
    L: stripe period [m].
    C_H_1, C_H_2: Helmholtz capacitance [F/m²] (default 50 µF/cm²).
    x: fraction of stripe that is facet 1 (default 0.5).
    """
    E_M: float
    E_pzc_1: float
    E_pzc_2: float
    C_H_1: float = C_H_DEFAULT
    C_H_2: float = C_H_DEFAULT
    x: float = 0.5
    c_b: float = 0.1
    L: float = 10e-9

    def to_solve_dict(self) -> dict:
        """Return dict in the format expected by solve_pb_2d."""
        return {
            'E_M': self.E_M,
            'E_pzc': np.array([self.E_pzc_1, self.E_pzc_2]),
            'C_H': np.array([self.C_H_1, self.C_H_2]),
            'x': self.x,
            'c_b': self.c_b,
            'L': self.L,
        }

    @property
    def delta_E_pzc(self) -> float:
        """ΔE_pzc = E_pzc^1 − E_pzc^2."""
        return self.E_pzc_1 - self.E_pzc_2

    @property
    def E_pzc_avg(self) -> float:
        """Area-weighted average PZC: x·E_pzc^1 + (1−x)·E_pzc^2."""
        return self.x * self.E_pzc_1 + (1.0 - self.x) * self.E_pzc_2


def make_f2_case(E_M: float, delta_E_pzc: float, c_b: float, L: float,
                 x: float = 0.5, C_H: float = C_H_DEFAULT) -> CaseConfig:
    """Factory for F2 panel cases.

    E_pzc^1 = +ΔE_pzc/2, E_pzc^2 = −ΔE_pzc/2 (symmetric about 0 V)
    so that E_pzc_avg = 0 V at x=0.5, matching loop.txt TARGET METRICS.
    """
    return CaseConfig(
        E_M=E_M,
        E_pzc_1=+delta_E_pzc / 2.0,
        E_pzc_2=-delta_E_pzc / 2.0,
        C_H_1=C_H, C_H_2=C_H,
        x=x, c_b=c_b, L=L,
    )
