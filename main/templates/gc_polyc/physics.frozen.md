# physics.frozen.md — gc_polyc_pb2d (FROZEN — integrity-pinned; do not edit)

Source of truth: Liu, Doblhoff-Dier, Koper, *ACS Electrochem.* 2026, 2, 995-1004
(DOI 10.1021/acselectrochem.5c00544). The classical Gouy-Chapman reference
implementation reproduces the paper and is covered by its own acceptance-test suite.

## Governing equation (dimensionless)

Poisson-Boltzmann, symmetric z:z electrolyte, classical Gouy-Chapman:

```
∂²φ/∂η² + ∂²φ/∂ζ² = sinh(φ)
```

with `φ = Fϕ/RT`, `ζ = z/λ_D`, `η = y/λ_D`, and Debye length

```
λ_D = sqrt(ε₀ ε_r R T / (2 F² c_b))
```

## Boundary conditions

- **z = 0 (electrode, per-facet Robin BC, encapsulates the Helmholtz layer):**
  `∂φ/∂ζ|₀ = ν_i (φ₀ − Φ_i)` with `ν_i = C_H^i λ_D / (ε₀ ε_r)` and
  `Φ_i = F (E_M − E_pzc^i) / (R T)` for the facet i under the y-position.
- **z = Z_max:** Dirichlet `φ = 0` (bulk reference), `Z_max = Z_max_factor · λ_D`.
- **y:** periodic with period `L` (dimensionless period `L/λ_D`).

## Geometry

Periodic stripe: `y ∈ [0, L)`, facet 1 occupies `y < x·L`, facet 2 the rest.
`z ∈ [0, Z_max]`, tanh-stretched mesh refined at z = 0 (stretch α = 3.0).

## Derived quantities

- Surface charge per facet strip: `σ_M(y) = C_H(y) · (E_M − E_pzc(y) − ψ_surf(y))` [C/m²]
- Area-averaged: `σ̄_M = ⟨σ_M(y)⟩_y`
- Differential capacitance along an E_M sweep: `C_dl = dσ̄_M/dE_M` (np.gradient)

## Single-facet closed forms (Chapman-Grahame; cross-check oracle)

```
σ_GC(ψ₀)  = sqrt(8 ε₀ ε_r R T c_b) · sinh(F ψ₀ / 2RT)
C_GC(ψ₀)  = sqrt(2 F² ε₀ ε_r c_b / RT) · cosh(F ψ₀ / 2RT)
C_DH      = C_GC(0)
ψ₀ solves:  σ_GC(ψ₀) = C_H (E_M − E_pzc − ψ₀)
1/C_dl    = 1/C_H + 1/C_GC(ψ₀)
```

## Out of scope (MUST NOT be added by any optimization step)

No Booth/field-dependent permittivity, no Bikerman/steric crowding, no specific
adsorption, no Stern slab meshing, no electrochemical kinetics. Classical GC only.
