using PortfolioOpt
using Documenter
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

for dir in joinpath.(@__DIR__, "src", ("examples", "tutorial", "explanation"))
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
            preprocess = content -> add_binder_links(
                replace(jl_filename, joinpath(@__DIR__, "src", "") => ""),
                content,
            ),
            postprocess = content -> replace(content, "nothing #hide" => ""),
        )
    end
end

makedocs(;
    modules=[PortfolioOpt],
    authors="Andrew Rosemberg",
    repo="https://github.com/andrewrosemberg/PortfolioOpt.jl/blob/{commit}{path}#L{line}",
    sitename="PortfolioOpt.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://andrewrosemberg.github.io/PortfolioOpt.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Robust PO" => "robust_po.md",
        "Examples" => list_of_sorted_files("examples", EXAMPLES_DIR),
        "API" => "api.md",
    ],
    strict=true,
    checkdocs=:exports,
)

deploydocs(; repo="github.com/andrewrosemberg/PortfolioOpt.jl")
