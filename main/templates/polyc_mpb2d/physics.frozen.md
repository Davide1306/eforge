# physics.frozen.md — polyc_mpb2d (FROZEN — integrity-pinned; do not edit)

Source of truth: Jinwen Liu, Katharina Doblhoff-Dier, Marc T. M. Koper,
"Modelling the Double Layer of Polycrystalline Electrodes: Capacitance, Potential
of Zero Charge, and Parsons-Zobel Plot", *ACS Electrochemistry* 2026, 2, 995-1004
(DOI 10.1021/acselectrochem.5c00544). The full-physics modified-Poisson-Boltzmann
reference implementation transcribes the equations from the paper's Methods
(eqs 1-13) with numeric SI parameters from Table S1.

## Governing equations (modified Poisson-Boltzmann; SI units)

Full mean-field model: a 2-D modified PB equation with asymmetric ion sizes AND
Langevin solvent polarization with a field-dependent effective permittivity.
This is NOT the bare classical `sinh(phi)` form; eps_eff and the steric/Langevin
denominator are field- and position-dependent.

```
(1)  div(eps_eff grad(phi_S)) = -e_0 (n_+ - n_-)

(2)  n_± = n_b e^(∓ beta e_0 phi_S) / D
     D   = (v/2)(gamma_+ e^(- beta e_0 phi_S) + gamma_- e^(+ beta e_0 phi_S))
           + [1 - (v/2)(gamma_+ + gamma_-)] * sinh(beta p E) / (beta p E)

(3)  eps_eff = eps_opt + (n_w p / E) * L(beta p E),   L(u) = coth(u) - 1/u  (Langevin)
     with beta = 1/(k_B T), p = solvent dipole moment, E = |grad(phi_S)|

(4)  n_w = n_w^b * [ sinh(beta p E) / (beta p E) ] / D4
     IMPLEMENT D4 ≡ D  (erratum E1: the as-printed extra n_w^b term is spurious)
```

with bulk ion number density `n_b = c_b[M] * 1e3 * N_A`, bulk volume fraction
`v = 2 d_t^3 n_b`, bulk water density `n_w^b = 1/d_t^3` (canonical; makes the
eq-3 low-field limit eps_eff -> eps_S exact), and Debye length

```
lambda_D = sqrt( eps_S k_B T / (2 e_0^2 n_b) )
```

(c_b = 0.1 / 0.3 / 1 / 10 mM  =>  lambda_D = 30.4 / 17.6 / 9.6 / 3.0 nm — Fig 7 canonical).

## Boundary conditions

- **z = 0 (electrode, facet-resolved Robin BC, eq 5; encapsulates the Helmholtz layer):**
  `dphi_S/dz|_0 = -(1/eps_S) C_H^i (E_M - E_pzc^i - phi_HP)`, with
  `C_H^i = eps_HP^i / l_HP^i` and `phi_HP = phi_S(y, z=0)`. The ε_S prefactor is
  the as-printed reading and the LOCKED default (`bc = "eps_S"`, erratum E2); the
  ε_eff(0) reading is the documented fallback only. E_M and E_pzc^i are on the
  same SHE scale.
- **z = Z_max (eq 6):** Dirichlet `phi_S = 0` (bulk reference),
  `Z_max = z_max_over_lamD * lambda_D` (z_max_over_lamD = 25).
- **y (eq 7):** periodic with period `L`; `phi_S(y=0) = phi_S(y=L)`.

## Geometry

Periodic stripe: `y ∈ [0, L)`, facet 1 occupies `y < x·L`, facet 2 the rest;
each facet i carries its own E_pzc^i and C_H^i. Normal `z ∈ [0, Z_max]` from the
Helmholtz plane (HP, z = 0) into the bulk. tanh-stretched z-mesh refined at z = 0
(stretch = 7.0).

## Derived quantities (charge extraction + capacitance, eqs 8-9)

- Free (countercharge) per unit electrode area, by Gauss's law:
  `q_free = (1/L) ∫_0^L eps_eff(z=0) dphi_S/dz|_0 dy` (eq 8; per-area, sign per
  erratum E3 so that dq_free/dE_M is positive — bookkeeping only).
- Differential capacitance along an E_M sweep: `C_dl = dq_free / dE_M` (eq 9).

## Closed forms / cross-check oracle (single-facet Gouy-Chapman / GCS)

```
(10)  C_GC(E)  = (eps_S / lambda_D) * cosh( e_0 (E - E_pzc) / (2 k_B T) )
(11)  1/C_dl   = 1/C_H + 1/C_GC      (GCS series; ideal PZ plot slope 1, intercept 1/C_H)
```

Langmuir specific adsorption (opt-in, eqs 12-13; active only when theta_max > 0):

```
(12)  ln( theta_ads / (1 - theta_ads) ) = e_0 (E_M - E_ads^0)/(k_B T) + ln c_b^(A-)
      closed form: theta_ads = sigma/(1+sigma),  sigma = c_b^(A-) e^(beta e_0 (E_M - E_ads^0))
(13)  C_ads = theta_max e_0 N_tot (dtheta_ads/dE_M),  theta_max = x * theta_max^facet1
      dtheta_ads/dE_M = beta e_0 theta_ads (1 - theta_ads);  Total C = C_dl + C_ads
      C_ads^peak = theta_max e_0^2 N_tot beta / 4  (at theta = 1/2; ~23.4 uF/cm^2 at theta_max = 1%)
```

## Errata resolutions (bind; transcribed from CONSTRAINTS.md §2 errata)

- **E1** (eq 4 denominator): the as-printed extra `n_w^b` factor in D4's last term
  is spurious (breaks dimensional consistency and the bulk limit). Implement
  **D4 ≡ D** (eq 2's shared denominator), which restores the exact bulk limit
  n_w -> n_w^b.
- **E2** (eq 5 Robin prefactor): **default = as-printed (divide by eps_S)**, i.e.
  `bc = "eps_S"`; the eps_eff(0) displacement-continuity reading is the documented
  fallback only (tested if single-facet/GCS validation fails at large |E_M - E_pzc|).
- **E4** (eq 12 c_b^(A-) standard state): **c_b^(A-) = 1/c_b[M]**, giving theta = 1/2
  at `E_M = E_ads^0 - (k_B T / e_0) ln c_b[M] ≈ E_ads^0 + 0.237 V` at 0.1 mM.
- **E6** (Fig 9 parameter provenance): resolve to **base L = 30 nm, x = 0.5** (the
  Fig-7 lineage that reproduces Fig 9's Parsons-Zobel slope anchors 0.76 / 0.48),
  not the Fig-8 (L = 10 nm, x = 0.2) reading.
- **E7** (eq-region minima vs L direction): **single minimum for L < lambda_D; two
  minima for L > lambda_D** (at large enough ΔE_pzc); the printed p.997 swap is a
  typo (contradicts the same paragraph, p.998, Fig 3, and Fig S1).

(E3 and E5 are pure bookkeeping/text-pointer fixes with no physics choice;
E3's sign convention is folded into the eq-8 q_free definition above.)

## Out of scope (MUST NOT be added by any optimization step)

These are the article's OWN exclusions (CONSTRAINTS.md §4); adopting them is not a
simplification of the article and adding them is a physics fabrication:

- More than 2 facets / random facet distributions.
- Grain boundaries.
- Facet spatial misalignment / atomic-level facet structure (facets are
  featureless stripes).
- Explicit adsorbate electronics / DFT coupling (adsorbates enter ONLY via C_ads
  and, where a figure says so, a reduced C_H).
- Multiple co-adsorbing species / specific surface-ion-solvent interactions.
- Electrochemical kinetics / faradaic currents (CO2R / HER).
- Time dependence (steady-state only).
