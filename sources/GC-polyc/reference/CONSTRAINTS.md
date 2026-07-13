# Article Constraints — Liu, Doblhoff-Dier, Koper (ACS Electrochem. 2026, 2, 995-1004)
# DO NOT EDIT — source of truth for implementation discipline in loop.txt
# DOI: 10.1021/acselectrochem.5c00544
# Reference PDFs: reference/source/{main.pdf, SI.pdf}
# Figure crops:    reference/crops/{fig1..fig7,figS1,tableS1}_*.png

## 0. No fitting parameters

The article relies entirely on parameters from Table S1 (no empirical
fits beyond literature-sourced values). DISCIPLINE: if an
implementation requires tuning a value beyond Table S1 → it is WRONG.
The double-layer capacitance C_dl(E_M), surface charge σ_M, PZ slope,
phase-diagram regime classification, and ψ_0(y) profiles are model
OUTPUTS computed by the equations of §2; they are NEVER adjusted to
match the article. Adjusting them is the artifact pattern (LESSON G3
forthcoming, mirrors rewriteB LESSON B3/B4).


## 1. Physical parameters — Table S1 (SI p11-12)

### General constants (universal — already in src/__init__.py)
k_B    = 1.381 × 10⁻²³ J/K        (Boltzmann)
T      = 298 K                     (absolute temperature)
e_0    = 1.6021 × 10⁻¹⁹ C          (elementary charge)
N_A    = 6.02 × 10²³ mol⁻¹         (Avogadro)
ε_0    = 8.85 × 10⁻¹² F/m          (vacuum permittivity)

### Solvent (water, bulk)
c_w     = 55.6 M                   (concentration of pure water)
ε_S     = 78.5 ε_0                 (bulk solution permittivity — used in GC)
ε_opt   = 1.8  ε_0                 (optical permittivity, ARTICLE EQ 3 ONLY — IGNORED at GC level)
ε_HP    = 30   ε_0                 (permittivity of metal-HP region — ENCAPSULATED IN C_H)
p       = 1.58 × 10⁻²⁹ C·m         (water dipole moment, ARTICLE EQ 3 ONLY — IGNORED at GC level)
d_t     = 3.1 Å                    (lattice size in solution, c_w·N_A^(-1/3) — used by Bikerman crowding ONLY — IGNORED at GC level)

### Helmholtz capacitance and electrolyte
C_H     = 50 μF/cm² = 0.5 F/m²     (per-facet default; sweepable)
c_b     = 0.1 mM (default base case); sweepable {0.1, 0.3, 1, 10} mM
γ_±     = 3                        (cation/anion size factor, Bikerman ONLY — IGNORED at GC level)

### Geometry / mosaic
L       = 10 nm (default base case); sweepable {1, 10, 30, 100} nm
x       = 0.5  (facet-1 fractional area; sweepable)
ΔE_pzc  ∈ {0, 0.2, 0.4, 0.6} V (sweep)

### Per-facet PZC (article default values, both vs SHE)
E_pzc_facet1 = +0.3 V  (article calls "T" — terrace / 111-like)
E_pzc_facet2 = -0.1 V  (article calls "S" — step    / 110-like)
                       (article notes: literature Au(111) ≈ 0.5 V SHE,
                        Au(110) ≈ 0.2 V SHE; chosen values reproduce
                        ΔE_pzc = 0.4 V representative of polycrystalline Au)

### Adsorption block — OUT OF SCOPE for GC-polyc (article §Adsorption only)
E_ads^0 = 0.45 V      (Langmuir adsorption potential)
θ_max   = 1 %         (max adsorption coverage)


## 2. Key equations — verbatim from main paper, with GC simplifications

### Article modified PB (full mean-field, eq 1-2)
ARTICLE  : -ε_eff ∇²ϕ_S = e_0 (n_+(ϕ) − n_−(ϕ))
           n_±(ϕ) = n_b · exp(∓βe₀ϕ_S) /
                    [1 + (v/2)(exp(βe₀ϕ_S) + exp(−βe₀ϕ_S) − 2)]
                    (Bikerman crowding in denominator)
GC SIMPL.: -ε₀ε_r ∇²ϕ = -2 F c_b sinh(F ϕ / RT)
           (set v=0 → standard Boltzmann distribution; ε_r constant = 78.5)

### Effective dielectric (article eq 3-4) — IGNORED at GC level
ARTICLE  : ε_eff = ε_opt + n_w(ϕ) p · L̂(βpE) / |E|
           n_w(ϕ) modulated by Bikerman denominator (eq 4)
           L̂(x) = coth(x) − 1/x     (Langevin function)
GC SIMPL.: ε_eff ≡ ε₀ ε_S = ε₀ · 78.5  (constant, no field-dependence)

### Helmholtz boundary condition (article eq 5)
At z = 0, for y in facet i:
  -ε_eff (∂ϕ_S/∂z) = C_H^i · (E_M − E_pzc^i − ϕ_HP)
where C_H^i = ε_HP^i / l_HP^i  (taken as 50 μF/cm² constant in base case).
GC retains this BC unchanged — the Helmholtz layer is encapsulated
in C_H^i and not resolved spatially.

### Bulk Dirichlet (article eq 6)
ϕ_S(z = ∞) = 0      (truncate at z = Z_max ≥ 5 λ_D in implementation)

### Periodicity (article eq 7)
ϕ_S(y = 0) = ϕ_S(y = L)

### Surface free charge (article eq 8)
σ_M(E_M) = q_free / L = (1/L) ∫₀^L  ε_eff (∂ϕ_S/∂z)|_{z=0}  dy
           (Gauss's law; SI convention positive σ → metal positive)
GC version uses ε_eff = ε₀ ε_S (constant).

### Differential double-layer capacitance (article eq 9)
C_dl(E_M) = dσ_M(E_M) / dE_M

### Single-facet Gouy-Chapman closed form (cross-check kernel)
For a uniform 1:1 symmetric electrolyte with no Bikerman crowding, the
diffuse-layer surface charge at electrode potential ψ_0 (vs bulk) is
  σ_GC(ψ_0) = sgn(ψ_0) · √(8 ε₀ ε_r R T c_b) · sinh( F ψ_0 / (2 R T) )
and the diffuse-layer capacitance is
  C_GC(ψ_0) = √(2 z² F² ε₀ ε_r c_b / (R T)) · cosh( F ψ_0 / (2 R T) )
With the Helmholtz layer in series:
  1/C_dl_single-facet(E_M) = 1/C_H + 1/C_GC(ψ_0(E_M))
ψ_0(E_M) is found by solving σ_GC(ψ_0) = C_H · (E_M − E_pzc − ψ_0).

### Debye length (article p2 footnote)
λ_D = √( ε_S k_B T / (2 e₀² c_b N_A) )
     = √( ε₀ ε_r R T / (2 F² c_b) )
For 1:1 electrolyte at T=298 K, ε_r=78.5: λ_D ≈ 9.6 nm at c_b = 1 mM.


## 3. Validation targets — figures the worker reproduces

The PNGs the worker writes under `output/` for F1-F5 are the
project deliverable. `reference/crops/figN_*.png` are the article
comparators — when the worker's PNG, viewed beside the corresponding
crop, is recognisable as the article's figure (correct panel layout,
sensible colour coding, axis ranges and labels matching the article),
that figure is "done." Quantitative bands below are how we know each
deliverable is honest, not a substitute for it.

Quantitative tolerances apply because classical GC drops Booth +
Bikerman, leading to expected absolute-magnitude offsets vs the
article (≤ 30 %) but identical qualitative trends.

### F1 — 2-D potential map ϕ(y, z)  (article Fig 1, right panel)
INPUTS:  E_M = 0 V vs SHE; E_pzc^1 = -E_pzc^2 = -0.3 V (so ΔE_pzc = 0.6 V);
         L = 10 nm; c_b = 10 mM (λ_D ≈ 3 nm); C_H^1 = C_H^2 = 50 μF/cm²;
         x = 0.5.
OUTPUTS: ϕ(y, z) on (Ny, Nz) tensor mesh; cmap diverging (red/blue).
ACCEPT:  no NaN; sign(ϕ_HP^i) follows -E_pzc^i as in article;
         periodic in y to machine precision (≤ 1e-12 V mismatch);
         |ϕ(z = Z_max)| < 1e-6 V.
ARTIFACT: reference/crops/fig1_potential_map.png

### F2 — Capacitance curves C_dl(E_M)  (article Fig 2, four panels)
INPUTS (4 panels × 3 lines/panel):
  panel(a): ΔE_pzc=0.6 V, c_b=0.1 mM (λ_D≈30 nm), L∈{1,10,100} nm
  panel(b): ΔE_pzc=0.6 V, c_b=10  mM (λ_D≈ 3 nm), L∈{1,10,100} nm
  panel(c): ΔE_pzc=0.2 V, c_b=0.1 mM,              L∈{1,10,100} nm
  panel(d): ΔE_pzc=0.2 V, c_b=10  mM,              L∈{1,10,100} nm
  Common:   x=0.5, C_H^1=C_H^2=50 μF/cm², E_M sweep over both PZCs ± 0.3 V.
OUTPUTS:  C_dl(E_M) per panel; solid = 2-facet model; dashed = weighted
          average of single-facet GC.
ACCEPT:   #-of-minima per panel matches article Fig 2;
          minimum positions within ±50 mV of article values;
          minimum magnitudes within 30 % of article values
          (classical GC underestimates because Bikerman crowding is
          neglected — KNOWN, not regression).
NOTE:     article main.pdf p3 has an internal contradiction on the
          minima rule. Sentence 1 (correct, matches SI Fig S1 +
          physics): "as L increases 1→100 nm, single minimum →
          two minima". Sentence 4 (typo, < / > swapped): "single
          when L > λD, two when L < λD". Trust sentence 1 + SI Fig S1:
          small L (L<λD) → 1 minimum (averaged); large L (L>λD) →
          2 minima (resolved). Worker reading the article alone
          may notice this inconsistency.
ARTIFACT: reference/crops/fig2_capacitance_panels.png

### F3 — Regime potential maps  (article Fig 3, four cases)
INPUTS:
  (a) Regime 1  : L=1 nm,  λ_D=3 nm, ΔE_pzc=0.4 V, x=0.5
  (b) Regime 2a : L=10 nm, λ_D=3 nm, ΔE_pzc=0.6 V, x=0.5, equal C_H
  (c) Regime 2b : L=30 nm, λ_D=3 nm, ΔE_pzc=0.2 V
  (d) Regime 2c : L=30 nm, λ_D=3 nm, ΔE_pzc=0.6 V, x=0.1 OR unequal C_H
OUTPUTS:  2-D ϕ(y,z) color maps per case.
ACCEPT:   (a) lateral averaging dominates (∂ϕ/∂y at z=0 small);
          (b) two distinct field zones per period;
          (c) similar to (a) — small ΔE_pzc averages out;
          (d) one facet visually dominates (small x or low C_H).
ARTIFACT: reference/crops/fig3_regime_maps.png

### F4 — Parsons-Zobel plots  (article Fig 7, panels a,b)
INPUTS:
  (a) effect of L ∈ {1, 10, 30, 100} nm at ΔE_pzc=0.2 V
  (b) effect of ΔE_pzc ∈ {0, 0.2, 0.4, 0.6} V at L=30 nm
  Concentrations c_b ∈ {0.1, 0.3, 1, 10} mM
   → λ_D ∈ {30.4, 17.6, 9.6, 3.0} nm
  C_H^1=C_H^2=50 μF/cm², x=0.5; PZ plot uses E_M = global E_pzc.
OUTPUTS: 1/C_dl(at PZC) vs 1/C_GC(c_b); linear fit per series; report
         slope + intercept.
ACCEPT:  PZ slopes monotone-decreasing with both L and ΔE_pzc;
         R² ≥ 0.95 of linear fit;
         article slopes (panel a — L sweep at ΔE_pzc=0.2 V):
            L=1 nm    → 0.99
            L=100 nm  → 0.42
         article slopes (panel b — ΔE_pzc sweep at L=30 nm):
            ΔE_pzc=0      → 1.00
            ΔE_pzc=0.6 V  → 0.47
         GC reproduction within ±20 % of article slopes per series.
ARTIFACT: reference/crops/fig7_parsons_zobel.png

### F5 — Phase diagram  (article SI Fig S1)
INPUTS:  scan L ∈ {1, 10, 30, 100} nm × ΔE_pzc ∈ multiple values such
         that eΔE_pzc/k_BT ∈ [0, 25]; c_b such that L/λ_D spans
         ~[0.03, 30].
OUTPUTS: classifier — for each (L/λ_D, eΔE_pzc/k_BT) point, count
         minima in C_dl(E_M); marker shape per L (circle=1nm,
         triangle=10nm, square=30nm, diamond=100nm); colour blue=1-min,
         red=2-min.
ACCEPT:  L/λ_D < 1 region is single-minimum across all ΔE_pzc;
         transition into two-minima at L/λ_D ≳ 1 AND eΔE_pzc/k_BT ≳ 4-6.
ARTIFACT: reference/crops/figS1_phase_diagram.png

### Stretch (LESSON-mint required to open scope)
Fig 5 — x and C_H asymmetry (article p5).
Fig 6 — E_dl,min vs global E_pzc (article p5/6).
Fig 8/9 — adsorption (OUT OF SCOPE; require Langmuir term).


## 4. Inputs vs outputs classification

ARTICLE INPUTS (sweepable):
  cation/anion identity (only the symmetric 1:1 case used here);
  c_b (bulk concentration);
  E_M (applied potential vs SHE);
  L (mosaic period);
  x (facet-1 fraction);
  ΔE_pzc, E_pzc^1, E_pzc^2;
  C_H^1, C_H^2.

MODEL OUTPUTS (computed — never tuned to fit):
  ϕ(y, z), ψ_0(y) = ϕ(y, z=0), σ_M(y);
  σ̄_M(E_M), C_dl(E_M);
  capacitance-minima count, minima positions, minima magnitudes;
  E_dl,min (global capacitance-minimum potential);
  PZ slope, PZ intercept;
  regime classification.

SOLVER INTERNALS (never reported as scientific result):
  mesh resolution Ny, Nz; mesh nonuniformity; tolerances atol/rtol;
  Newton iteration count; line-search choice; preconditioner.

OUT OF SCOPE — DO NOT IMPLEMENT in GC-polyc:
  - Booth-Langevin field-dependent ε_eff (article eq 3-4);
  - Bikerman / lattice-gas crowding terms (the v-denominator in eq 2);
  - Specific adsorption / Langmuir term (article §Adsorption,
    Figs 8-9);
  - Multi-electrolyte (asymmetric ions, mixed valence);
  - 3-D facet patterns (article uses stripe geometry, periodic in y;
    z-axis normal; perfect periodicity in the third in-plane axis is
    assumed — keep this assumption).

If a future LESSON G-mint requires lifting one of these restrictions,
it must be authorised by review and recorded in
loop.txt LESSONS section before implementation.
