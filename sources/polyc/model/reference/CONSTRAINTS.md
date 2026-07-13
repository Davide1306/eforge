# CONSTRAINTS.md — frozen ground truth (bootstrap 2026-06-17)

**FROZEN.** Any edit after bootstrap must halt and revert.
Authored once by reading the source inputs
artifacts (the CLAUDE.md `model-declaration` + the source PDFs); every statement
below cites the paper, never the model implementation.

**Paper:** Jinwen Liu, Katharina Doblhoff-Dier, Marc T. M. Koper, "Modelling the
Double Layer of Polycrystalline Electrodes: Capacitance, Potential of Zero Charge,
and Parsons–Zobel Plot", *ACS Electrochemistry* 2026, 2, 995–1004.
DOI `10.1021/acselectrochem.5c00544` (CC-BY-NC-ND 4.0). Data (on publication):
`10.4121/47abf0ac-8b80-4cca-a31e-945ae10d8959`.

**Frozen sources** (`model/reference/source/`):
- `main.pdf` — sha256 `6d0f48b573463ec6b1895081126afbac82e8c37140ead74c7ab99d114bde7909` (10 pp)
- `SI.pdf`  — sha256 `af5f57391793b2f9f93fc77d7e82f04219f856895bde5548e8a806ab182a8601`
  (13 pp; Figs S1–S8 on pp S3–S10, Table S1 on pp S11–S12)

Figure crops: `model/reference/crops/*.png` (B04). Digitized reference curves land
in `model/reference/extracted/` during the run (worker digitizes via the
extract-figures skill; accepted via digitization-audit + sha256).

## §0 — No-fitting discipline

1. **Zero free parameters.** Every numeric input is fixed by §1 (Table S1) or by
   a per-figure caption declaration in §3. Nothing is tuned, scaled, offset, or
   re-fit to move a model curve toward a target curve.
2. **Full physics per the declaration.** The article's complete mean-field model:
   a 2-D modified Poisson–Boltzmann equation with asymmetric ion sizes AND
   Langevin solvent polarization with a field-dependent effective permittivity
   (eqs 1–4), a facet-resolved Robin (Helmholtz) boundary condition (eq 5),
   y-periodicity (eq 7), Gauss-law charge extraction + differentiation (eqs 8–9),
   the analytic Gouy–Chapman / GCS relations (eqs 10–11) as cross-checks, and the
   Langmuir adsorption-capacitance extension (eqs 12–13). The ONLY admissible
   simplifications are the article's own (§4). A Gouy–Chapman-only or
   no-polarization variant is a diagnostic stepping-stone (tier ladder), never a
   deliverable.
3. **Deliverables are computed outputs.** F1–F5 (§3) are produced by solving §2 at
   §1/§3 inputs on the declared mesh — never by digitizing, interpolating, or
   replotting article data as model output.
4. **Solver re-implementation.** The article solved the PDE in COMSOL Multiphysics
   6.2. This project re-implements in Python (`python`
   only). The numerical method is free; the physics and parameters are not.
5. **Printed-equation primacy.** §2 transcribes the equations as printed, then
   records every printed inconsistency found on careful reading (errata E1–E7).
   Resolutions are derived consistency facts (dimensional analysis, bulk limits,
   Table S1 pins) — never curve fits. Where a genuine ambiguity remains (E2, E4,
   E6) BOTH readings are tracked and the resolution criterion is a named §3 figure,
   decided and recorded in the run docs; silently picking
   one is forbidden.

## §1 — Parameter table (SI Table S1 — base case; FROZEN inputs)

| Category | Symbol | Value | Source / note |
|---|---|---|---|
| General | k_B | 1.381e-23 J·K⁻¹ | constant |
| General | T | 298 K | constant |
| General | e_0 | 1.6021e-19 C | constant |
| General | N_A | 6.02e23 mol⁻¹ | constant |
| General | ε_0 | 8.85e-12 F·m⁻¹ | constant |
| General | c_w (pure water) | 55.6 M | constant |
| Electrolyte | ε_opt (optical) | 1.8 ε_0 | SI ref 4 |
| Electrolyte | ε_HP (metal↔HP gap) | 30 ε_0 | SI ref 5 (enters only via C_H^i = ε_HP^i/l_HP^i) |
| Electrolyte | ε_S (bulk solution) | 78.5 ε_0 | constant (bulk water) |
| Electrolyte | C_H (Helmholtz, default) | 50 μF/cm² | SI ref 1; per-figure overrides in §3 (Fig S3 uses 25) |
| Electrolyte | c_b (bulk conc., default) | 0.1 mM | experimental condition; per-figure overrides |
| Electrolyte | d_t (lattice size) | 3.1 Å | = (c_w N_A)^(−1/3) |
| Electrolyte | γ_± (ion size factor) | 3 | estimated; γ_± = (d_±/d_t)³, γ_+ = γ_− = 3 throughout |
| Electrolyte | p (solvent dipole) | 1.58e-29 C·m | = √(3 d_t³ (ε_S − ε_opt) k_B T) |
| Electrode | L (unit-cell length, default) | 10 nm | variable (1–100 nm swept) |
| Electrode | x (facet ratio, default) | 0.5 | variable |
| Electrode | E_pzc^T (111 facet) | +0.3 V_SHE | SI refs 3,6 |
| Electrode | E_pzc^S (110 facet) | −0.1 V_SHE | SI ref 7 |
| Electrode | N_tot (metal atom density) | 1.5029e19 m⁻² | = 4/(√3 a_Pt²), a_Pt = 3.92 Å |
| Adsorption | E_ads^0 (default) | 0.45 V | variable |
| Adsorption | θ_max (default) | 1% | variable; θ_max = x·θ_max^facet1 (eq 13) |

Derived quantities re-verified at bootstrap (computed from the table — no rounding
surprises):
- n_b = 1000·c_b[M]·N_A m⁻³; n_w^b = c_w N_A = 1/d_t³ = 3.35e28 m⁻³ (55.6 M ⇒ 3.347e28; 1/d_t³ ⇒ 3.357e28, agree ~0.3%).
- v = 2 d_t³ n_b (bulk volume fraction of solvated ions); β = 1/k_B T = 2.43e20 J⁻¹.
- p check: √(3·(3.1e-10)³·(78.5−1.8)·8.85e-12·k_B·298) = 1.580e-29 C·m ✓
  (equivalently the Langevin low-field closure ε_S = ε_opt + n_w^b p² β/3).
- N_tot check: 4/(√3·(3.92e-10)²) = 1.5029e19 m⁻² ✓
- λ_D = √(ε_S k_B T / (2 e_0² c_b N_A·1000·[M])):
  c_b = 0.1 / 0.3 / 1 / 10 mM ⇒ λ_D = 30.4 / 17.6 / 9.6 / 3.0 nm ✓
  (running text rounds 30.4→"30"; the Fig 7 caption lists 30.4/17.6/9.6/3.0 —
  use the Fig 7 values as canonical).
- The base-case facet pair {E_pzc^T, E_pzc^S} = {+0.3, −0.1} V gives ΔE_pzc = 0.4 V,
  average +0.1 V. Figures OVERRIDE this center per caption (§3): the paper varies
  ΔE_pzc ∈ {0.2,…,0.6} V and re-centers (often symmetric ∓ΔE_pzc/2 about 0).

## §2 — Equations VERBATIM (main.pdf Methods, eqs 1–13)

Geometry: 2-D. In-plane y ∈ [0, L] across the facets (periodic), with two facets
in stripes — facet 1 occupies fraction x of L, facet 2 the remainder; the in-plane
direction parallel to the facet edges is perfectly periodic (ignorable). Normal
z ∈ [0, ∞) from the Helmholtz plane (HP, z = 0) into the bulk. Each facet i carries
its own E_pzc^i and C_H^i.

- Debye length (monovalent, Methods inline): **λ_D = √(ε_S k_B T / (2 e_0² c_b N_A))**
  (c_b converted mol→m⁻³; see §1 check).
- **(1)** ∇·(ε_eff ∇ϕ_S) = −e_0 (n_+ − n_−)
- **(2)** n_± = n_b e^(∓βe_0ϕ_S) / D, with
  D = (v/2)(γ_+ e^(−βe_0ϕ_S) + γ_− e^(βe_0ϕ_S)) + [1 − (v/2)(γ_+ + γ_−)]·sinh(βpE)/(βpE)
- **(3)** ε_eff = ε_opt + (n_w p / E)·𝓛(βpE), 𝓛(u) = coth(u) − 1/u (Langevin),
  β = 1/k_B T, p = solvent dipole moment, E = |∇ϕ_S|
- **(4)** n_w = n_w^b·[sinh(βpE)/(βpE)] / D₄, where AS PRINTED
  D₄ = (v/2)(γ_+ e^(−βe_0ϕ_S) + γ_− e^(βe_0ϕ_S)) + [1 − (v/2)(γ_+ + γ_−)]·**n_w^b**·sinh(βpE)/(βpE)
  → see erratum E1 (the boxed n_w^b is spurious)
- **(5)** ∂ϕ_S/∂z |_(z=0) = −(1/ε_S)·C_H^i·(E_M − E_pzc^i − ϕ_HP),
  C_H^i = ε_HP^i / l_HP^i; ϕ_HP = ϕ_S(y, z=0); E_M and E_pzc^i on the same scale (SHE)
  → see ambiguity E2 (prefactor ε_S vs ε_eff(0))
- **(6)** ϕ_S(z = ∞) = 0
- **(7)** ϕ_S(y = 0) = ϕ_S(y = L)   (periodic in y)
- **(8)** q_free = ∫₀^L ε_eff(z=0)·∂ϕ_S/∂z |_(z=0) dy   ("according to Gauss's law")
  → see normalization/sign note E3 (printed integral has units C/m)
- **(9)** C_dl = dq_free / dE_M
- **(10)** C_GC(E) = (ε_S/λ_D)·cosh( e_0 (E − E_pzc) / (2 k_B T) )
  (single-facet Gouy–Chapman differential capacitance, monovalent electrolyte)
- **(11)** 1/C_dl = 1/C_H + 1/C_GC   (GCS series relation; in the ideal GCS frame
  the PZ plot of 1/C_dl vs 1/C_GC has slope 1, intercept 1/C_H)
- **(12)** ln( θ_ads / (1 − θ_ads) ) = e_0 (E_M − E_ads^0)/(k_B T) + ln c_b^(A⁻)
  (Langmuir isotherm for A⁻ + * ⇌ A_ad + e⁻; c_b^(A⁻) assumed = c_b)
  → see units convention E4
- **(13)** C_ads = θ_max·e_0·N_tot·(dθ_ads/dE_M), θ_max = x·θ_max^facet1; capacitance
  positive for oxidative adsorption. Total **C = C_dl + C_ads** (parallel addition;
  C_dl from the no-adsorbate model at the same inputs).

Exact consequences (derived cross-checks — not new physics):
- Bulk/low-field limit: as E→0, ϕ_S→0 ⇒ sinh(βpE)/(βpE)→1, n_±→n_b, D→1, and
  ε_eff→ε_opt + n_w p² β/3; with n_w→n_w^b this equals ε_S exactly (the §1 p-pin).
- Eq 12 closed form: θ_ads(E_M) = σ/(1+σ), σ = c_b^(A⁻)·e^(βe_0(E_M−E_ads^0));
  dθ_ads/dE_M = βe_0·θ_ads(1−θ_ads), so C_ads peaks at θ_ads = ½ with
  C_ads^peak = θ_max e_0² N_tot β/4 ≈ 23.4 μF/cm² at θ_max = 1%.

**Errata / printed inconsistencies (found on careful reading; resolution criteria bind):**
- **E1 (eq 4 denominator — resolved at bootstrap).** As printed, D₄'s last term
  carries an extra n_w^b (units m⁻³ added to an otherwise dimensionless D₄, so the
  bulk limit gives n_w ≠ n_w^b). The shared denominator D (eq 2's) restores
  dimensional consistency and the exact bulk limit n_w→n_w^b, and is pinned
  independently by Table S1's own p = √(3 d_t³(ε_S−ε_opt)k_BT) (⇔ ε_S = ε_opt +
  n_w^b p²β/3, which presumes n_w→n_w^b). IMPLEMENT D₄ ≡ D.
- **E2 (eq 5 prefactor — ambiguity; resolve at tier T3/T5 vs Fig S3/§3).** As printed
  the Robin BC divides by ε_S; Gauss-law displacement continuity through the
  charge-free Helmholtz gap would instead use ε_eff(z=0)
  (σ_M = C_H^i(E_M − E_pzc^i − ϕ_HP) = −ε_eff(0)·∂ϕ_S/∂z|₀). Default = as-printed (ε_S);
  test the ε_eff(0) reading if single-facet validation (Fig S3 camel→bell + GCS eq 11
  consistency) or §3 bands fail at large |E_M − E_pzc|. A9 call; record the verdict.
- **E3 (eq 8 normalization/sign).** As printed the integral has units C/m and, with
  eq 5's sign, returns the ionic countercharge. Convention adopted: q_free is
  per unit area (divide by L) with sign such that dq_free/dE_M reproduces the
  article's positive C_dl. Pure bookkeeping; no physics choice.
- **E4 (eq 12 c_b^(A⁻) units — ambiguity; resolve vs Fig S7).** ln c_b^(A⁻) needs a
  standard state. Candidate readings: c in M (θ=½ at E_M = E_ads^0 − (k_BT/e_0)ln c_b[M]
  ≈ E_ads^0 + 0.237 V at 0.1 mM) vs a normalization making θ=½ at E_M = E_ads^0.
  Resolve by matching Fig S7(a) coverage curves; A9 call; record the verdict.
- **E5 (text pointer typo).** Main text (PZ-plot paragraph) "the inverse C_GC given
  in eq 9" should read eq 10 (eq 9 is C_dl = dq/dE_M). No action.
- **E6 (Fig 9 parameter provenance — ambiguity).** Main text: Fig 9 "parameters
  identical to those in Figure 8" (L = 10 nm, x = 0.2); Fig 9 caption: "All other
  parameters are identical to those in Figure 7" (base L = 30 nm, x = 0.5). Resolve
  by whichever reproduces Fig 9's slope anchors (0.76 / 0.48, §3 F5); A9 call.
- **E7 (text typo, found at bootstrap — no action).** Main text p.997: "a single
  minimum appears when L > λ_D, while two minima emerge when L < λ_D" is SWAPPED —
  it contradicts the same paragraph ("curves evolve from single to two minima as L
  increases 1→100 nm" at λ_D = 30 nm), p.998 ("When L < λ_D … single minimum"),
  Fig 3 (Regime 1: L < λ_D ⇒ one minimum), and Fig S1. The model's own output
  governs: **single minimum for L < λ_D; two minima for L > λ_D** (at large enough ΔE_pzc).

## §3 — Deliverable figure targets F1–F5 + validation set + acceptance bands

Reference scale: potentials V vs SHE. Unless a row says otherwise: x = 0.5,
C_H^1 = C_H^2 = 50 μF/cm², parameters per §1. Facet-PZC placement is per-figure
(the paper moves the center): Fig 1 declares E_pzc^1 = −E_pzc^2 = −0.3 V (ΔE_pzc = 0.6,
centered at 0); Fig S6 declares global E_pzc = 0.2 V for a ΔE_pzc = 0.2, x = 0.5 case
(⇒ facet PZCs {0.1, 0.3}); Table S1's base pair is {−0.1, +0.3}. Where a figure
leaves the center undeclared, read it from the red-arrow / minima positions of the
digitized crop (digitization-audit gates acceptance) — never assume.

**Acceptance bands (article-grade; FULL physics ⇒ tight):**
- **B-pos** capacitance-minimum / feature positions: |ΔE| ≤ 20 mV.
- **B-mag** capacitance magnitudes, pointwise on digitized curves: rel. err ≤ 10%
  (digitization floor ~2–5%).
- **B-count** number of capacitance minima: EXACT (incl. Fig S1's stated convention:
  very shallow extra minima near the L ≈ λ_D transition count as "one minimum").
- **B-slope** Parsons–Zobel slopes: |Δslope| ≤ 0.05 absolute, linear-fit R² > 0.99
  (paper reports R² > 0.99); intercepts rel. err ≤ 10%.
- **B-dev** E_dl,min − E_pzc deviation curves (meV scale): |Δ| ≤ 10 meV.
- Diagnostic-tier work may exceed bands; FINAL/article-grade claims may not.

| ID | Target (main.pdf) | Inputs (caption-declared) | Quantitative anchors (text-cited) | Bands |
|---|---|---|---|---|
| **F1** | Fig 2a–d: C_dl(E) two-facet (solid) vs weighted-average (dashed), L ∈ {1,10,100} nm | (a) ΔE_pzc=0.6 V, c_b=0.1 mM (λ_D=30 nm); (b) 0.6 V, 10 mM (3 nm); (c) 0.2 V, 0.1 mM; (d) 0.2 V, 10 mM; x=0.5, C_H=50/50; red arrows = single-facet E_pzc | single minimum for L<λ_D; two minima for L>λ_D at ΔE_pzc=0.6; convergence to weighted average as L/λ_D grows; Fig 2a orange shallow extra minima = "one minimum" class | B-pos, B-mag, B-count |
| **F2** | Fig 5: C_dl(E) varying x and C_H^1 | L=10 nm, c_b=10 mM (λ_D=3 nm), ΔE_pzc=0.6 V; legends crop-read; C_H^2=50 fixed | second minimum suppressed as x↓ or C_H^1↓; surviving minimum near PZC of the dominant facet | B-pos, B-mag, B-count |
| **F3** | Fig 6a,b: E_dl,min vs global E_pzc over L/λ_D (log x-axis) | (a) ΔE_pzc=0.2 V, x=0.2 (single-min); (b) ΔE_pzc=0.6 V, x=0.5 (two-min, both facet branches); C_H=50/50; curves underlying = Fig S4 | (a) \|E_dl,min−E_pzc\| ≤ 10 meV for L/λ_D ≤ 3, growing to ~45–50 meV by L/λ_D ≳ 30. (b) minima near facet PZCs; deviations >70 meV, λ_D-dependent | B-dev (a); B-pos (b) |
| **F4** | Fig 7a,b: PZ plots (1/C_dl,min vs 1/C_GC) | c_b ∈ {0.1,0.3,1,10} mM ⇒ λ_D ∈ {30.4,17.6,9.6,3.0} nm; base L=30 nm, ΔE_pzc=0.2 V, x=0.5, C_H=50/50; (a) L swept; (b) ΔE_pzc swept (missing point at low 1/C_GC where two minima) | slope 0.99→0.42 as L:1→100 nm; slope 1.00→0.47 as ΔE_pzc:0→0.3 V; intercept ≈ const ≈ 1/C_H; R²>0.99 | B-slope |
| **F5** | Fig 8a–d + Fig 9a,b: adsorption (C = C_dl + C_ads) | C_dl case: L=10 nm, c_b=0.1 mM, ΔE_pzc=0.2 V, x=0.2, C_H=50/50; base E_ads^0=0.45 V, θ_max=1%; Fig 8a/c sweep E_ads^0/θ_max; Fig 9a: E_ads^0 sweep @ θ_max=1%; Fig 9b: θ_max sweep @ E_ads^0=0.40 V; Fig 9 base per erratum E6 | θ_max=1% already shifts E_min by >70 mV; two minima when adsorption peak overlaps C_dl,min; PZ slope 0.98 @ E_ads^0=0.25 V; 0.76 @ E_ads^0=0.40 V θ_max=1%; 0.48 @ θ_max=8%; all > exp Pt(111) 0.041 | B-pos, B-mag, B-slope |

**Validation set (gates tiers, not deliverable-numbered):**

| ID | Target | Inputs | Pass criterion |
|---|---|---|---|
| V0 | Fig S3: single-facet C_dl(E) vs c_b | one facet, C_H=25 μF/cm², c_b sweep, E−E_pzc ∈ [−1,1] V | GC minimum at E_pzc for dilute c_b; camel→bell transition with rising c_b; B-pos/B-mag vs digitized curves |
| V1 | Fig S1: phase diagram #minima | axes eΔE_pzc/k_BT vs L/λ_D; markers L ∈ {1,10,30,100} nm (circle/triangle/square/diamond); x=0.5, C_H=50/50 | B-count EXACT on every digitized marker (shallow-minima convention) |
| V2 | Fig 1: 2-D potential map | E_pzc^1,2 = ∓0.3 V, shown at E_M=0 V; L=10 nm, c_b=10 mM, C_H=50/50 | antisymmetric ϕ lobes over the two facets, decay ~λ_D; pointwise only if digitized |
| V3 | Fig S6: 2-D ϕ map + ϕ_HP(y) profile | c_b=0.1 mM (λ_D=30 nm), L=100 nm, ΔE_pzc=0.2 V, x=0.5, E_M=global E_pzc=0.2 V | facets carry equal/opposite charge at global PZC; ϕ_HP(y) matches digitized S6b within B-mag |
| V4 | Figs S2, S4, S5, S7, S8 | S2: ΔE_pzc ∈ {0.3,0.4,0.5} V, L/λ_D ∈ {1,3,10}, x=0.5; S4: x=0.2, ΔE_pzc=0.2 V, c_b ∈ {0.1,1,10} mM, L ∈ {1,10,100} nm; S5: L=100 nm, c_b ∈ {0.1,0.2} M (minimum→maximum inversion); S7/S8: adsorption sweeps | spot-check curves on demand; S7 doubles as the E4 resolution instrument; S8 at E_ads^0=0.40 V, θ_max=1% |

## §4 — Inputs vs outputs; scope

**Inputs (FROZEN):** §1 table; per-figure declarations in §3; geometry (two stripes,
fractions x / 1−x, period L); reference scale (SHE).
**Sweepable input tuple** (the ONLY knobs a run may vary, each over its article
range): c_b (0.1–10 mM; S5 extends to 0.1–0.2 M) × E_M (window per figure) × L
(1–100 nm) × ΔE_pzc (0–0.6 V incl. per-figure center) × x (0–1) × C_H^i (μF/cm² per
facet) × E_ads^0 × θ_max. Mesh/solver settings are numerical, not physical, knobs.

**Outputs (computed, never digitized):** ϕ_S(y,z), n_±, n_w, ε_eff fields; ϕ_HP(y);
q_free(E_M); C_dl(E_M); E_dl,min, C_min; #minima; global/facet PZC relations; PZ plots
+ fitted slopes/intercepts; θ_ads(E_M), C_ads(E_M), C = C_dl + C_ads. Deliverable
figures F1–F5 under `model/output/`.

**In scope (the article's full model — none of these may be dropped):** 2-D modified
PB (eqs 1–4: asymmetric ion size γ_±, volume fraction v, Langevin polarization,
field-dependent ε_eff, position-dependent n_w); facet-resolved Robin BC (eq 5);
periodicity (eq 7); GC/GCS analytics (eqs 10–11) as cross-checks; Langmuir
adsorption capacitance (eqs 12–13).

**Out of scope (the article's OWN exclusions — adopting them is not a simplification
of the article):** >2 facets / random facet distributions; grain boundaries; facet
spatial misalignment; atomic-level structure (facets are featureless stripes);
explicit adsorbate electronics (adsorbates enter ONLY via C_ads and, where a figure
says so, a reduced C_H); multiple co-adsorbing species; specific surface–ion/solvent
interactions (refs 56–58); DFT coupling.

**Tracked ambiguities:** §2 E2 (eq 5 ε_S vs ε_eff(0)), E4 (c_b^(A⁻) standard state),
E6 (Fig 9 base parameters) — each carries its resolution criterion in §2; resolutions
are review calls recorded in the run docs, never silent.
