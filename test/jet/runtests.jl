# Static analysis of PortfolioOpt with JET.jl. Run from the repository root with
#
#     julia --project=test/jet -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#     julia --project=test/jet test/jet/runtests.jl
#
# It is not part of `Pkg.test()` because JET versions are tied to Julia minor versions.

using JET
using PortfolioOpt
using Test

# Reports that are known and harmless in practice. Each is identified by the function in which
# it is reported:
#  - `portfolio_model!`: `pointers` is `nothing` unless `record_measures=true`, and it is only
#    indexed in that case (JET does not correlate the two branches);
#  - `day_backtest_market!`: the `optimization_recorder` is only indexed when the strategy
#    returns the pointers of a formulation, and the default `state_recorders=Dict()` is never used
#    by `sequential_backtest_market`.
const KNOWN = ("portfolio_model!", "day_backtest_market!")

@testset "JET" begin
    result = JET.report_package(
        PortfolioOpt; target_modules=(PortfolioOpt,), toplevel_logger=nothing
    )
    reports = JET.get_reports(result)
    # name of the (keyword sorter of the) function in which the report was found
    reported_in(r) = string(first(r.vst).linfo.def.name)
    is_known(r) = any(f -> occursin(f, reported_in(r)), KNOWN)
    unknown = filter(!is_known, reports)
    isempty(unknown) || show(stdout, MIME"text/plain"(), result)
    @test isempty(unknown)
    @test length(reports) - length(unknown) <= 5
end
