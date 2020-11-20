using PortfolioOpt
using Documenter

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
    pages=["Home" => "index.md"],
    strict=true,
    checkdocs=:exports,
)

deploydocs(; repo="github.com/andrewrosemberg/PortfolioOpt.jl")
