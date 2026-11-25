# Code quality checks: Aqua.jl, ExplicitImports.jl and docstring coverage. JET.jl runs in a
# separate CI job (see test/jet/runtests.jl) because its results depend on the Julia version.

using Aqua
using ExplicitImports

@testset "Aqua" begin
    Aqua.test_all(PortfolioOpt;
        # known: `rand(::DeterministicSamples, n)` is ambiguous with methods of Random and
        # Distributions for `n::Integer` (not `Int`) and `n::Tuple`
        ambiguities=(broken=true,),
        # known: `mode` and `shape` are exported by both JuMP (re-exported) and Distributions
        undefined_exports=(broken=true,),
        # known: `rand(::AbstractRNG, ::Dirac, ::Int)` is defined for the Dirac distributions
        # used as point forecasts
        piracies=(broken=true,),
    )
end

@testset "ExplicitImports" begin
    @test check_no_stale_explicit_imports(PortfolioOpt) === nothing
    @test check_no_self_qualified_accesses(PortfolioOpt) === nothing
    # `rand` is extended through Distributions, which re-exports it from Random
    @test check_all_explicit_imports_via_owners(PortfolioOpt; ignore=(:rand,)) === nothing
    @test check_all_qualified_accesses_via_owners(PortfolioOpt; ignore=(:rand,)) === nothing
    # known: the package relies on `using JuMP` and `using Distributions` (JuMP is re-exported)
    @test_broken check_no_implicit_imports(PortfolioOpt) === nothing
end

@testset "every exported symbol is documented" begin
    undocumented = Symbol[]
    for name in names(PortfolioOpt)
        name === :PortfolioOpt && continue
        isdefined(PortfolioOpt, name) || continue
        # skip the names re-exported from JuMP and MathOptInterface
        Base.binding_module(PortfolioOpt, name) === PortfolioOpt || continue
        haskey(Base.Docs.meta(PortfolioOpt), Base.Docs.Binding(PortfolioOpt, name)) ||
            push!(undocumented, name)
    end
    @test isempty(undocumented)
end
