# Contributing to PortfolioOpt.jl

Contributions are welcome: bug reports, new measures or ambiguity sets, examples and documentation. The project
follows [ColPrac](https://github.com/SciML/ColPrac).

## Workflow

1. Open an issue to discuss larger changes first.
2. Fork, create a branch, and keep each commit to one concern.
3. Run the tests and, if you changed the docs, build them:
   ```sh
   julia --project=. -e 'using Pkg; Pkg.test()'
   julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
   julia --project=docs docs/make.jl
   ```
   Static analysis with JET runs separately (Julia 1.12 or newer):
   ```sh
   julia --project=test/jet -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
   julia --project=test/jet test/jet/runtests.jl
   ```
4. Format with [JuliaFormatter](https://github.com/domluna/JuliaFormatter.jl) (BlueStyle, see `.JuliaFormatter.toml`).

## Adding a measure or an ambiguity set

- Subtype `PortfolioRiskMeasure` / `CenteredAmbiguitySet` and implement `calculate_measure!` (see the
  "Extending PortfolioOpt" docs page).
- Write the method as an optimization problem in the Theory pages, with its notation, assumptions and reference.
- State its claims (closed forms, bounds, monotonicity) in the docstring under "Guarantees", and add one named
  testset per claim in `test/guarantees.jl`.
- Document every exported symbol; the tests check it.
