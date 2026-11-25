using PortfolioOpt
using Documenter
using DocumenterMermaid
using Literate
using Test
using Random

const EXAMPLES_DIR = joinpath(@__DIR__, "src", "examples")

_sorted_files(dir, ext) = sort(filter(f -> endswith(f, ext), readdir(dir)))

function list_of_sorted_files(prefix, dir, ext = ".md")
    return Any["$(prefix)/$(file)" for file in _sorted_files(dir, ext)]
end

function _include_sandbox(filename)
    mod = @eval module $(gensym()) end
    return Base.include(mod, filename)
end

for dir in joinpath.(@__DIR__, "src", ("examples",))
    for jl_filename in list_of_sorted_files(dir, dir, ".jl")
        Random.seed!(12345)
        # `include` the file to test it before `#src` lines are removed. It is
        # in a testset to isolate local variables between files.
        Test.@testset "$jl_filename" begin
            _include_sandbox(jl_filename)
        end
        Random.seed!(12345)
        Literate.markdown(
            jl_filename,
            dir;
            documenter = true,
            postprocess = content -> replace(content, "nothing #hide" => ""),
        )
    end
end

DocMeta.setdocmeta!(PortfolioOpt, :DocTestSetup, :(using PortfolioOpt); recursive=true)

const EXAMPLES = [
    "Stochastic optimization" => "examples/stochastic_optimization.md",
    "Robust optimization" => "examples/robust_optimization.md",
    "Distributionally robust optimization" => "examples/distributionally_robust_optimization.md",
    "Model assessment" => "examples/model_assessment.md",
]
@assert sort(last.(EXAMPLES)) == list_of_sorted_files("examples", EXAMPLES_DIR)

makedocs(;
    modules=[PortfolioOpt],
    authors="Andrew Rosemberg",
    sitename="PortfolioOpt.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://andrewrosemberg.github.io/PortfolioOpt.jl",
        edit_link="master",
        assets=String[],
        size_threshold=400 * 1024,
    ),
    pages=[
        "Home" => "index.md",
        "Theory" => [
            "The portfolio problem" => "theory/problem.md",
            "Risk measures" => "theory/measures.md",
            "Ambiguity sets" => "theory/ambiguity.md",
            "Robust portfolio optimization" => "theory/robust_po.md",
            "Guarantees" => "theory/guarantees.md",
        ],
        "Package guide" => [
            "Getting started" => "guide/getting_started.md",
            "Concepts" => "guide/concepts.md",
            "Extending PortfolioOpt" => "guide/extending.md",
            "Breaking changes in v2.0" => "guide/migration.md",
        ],
        "Examples" => EXAMPLES,
        "API reference" => "api.md",
    ],
    doctest=true,
    checkdocs=:exports,
)

deploydocs(; repo="github.com/andrewrosemberg/PortfolioOpt.jl", devbranch="master", push_preview=true)
