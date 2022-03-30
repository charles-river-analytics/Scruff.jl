push!(LOAD_PATH,"../src/")

using Documenter
using Scruff

makedocs(
    sitename = "Scruff.jl",
    modules = [Scruff, Scruff.SFuncs, Scruff.Algorithms, Scruff.Models, 
        Scruff.Operators, Scruff.Utils, Scruff.RTUtils],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Avi Pfeffer, Michael Harridon, Joe Campolongo, Sanja Cvijic, and contributors.",
    pages = [
        "Getting Started" => "index.md",
	    "Tutorial" => "tutorial/tutorial.md",
        "Examples" => "examples.md",
        "Library" => Any[
            "Core" => "lib/core.md",
            "Stochastic Functions" => "lib/sfuncs.md",
            "Operators" => "lib/operators.md",
            "Models" => "lib/models.md",
            "Algorithms" => "lib/algorithms.md",
            "Utilities" => "lib/utilities.md",
            "Runtime Utilities" => "lib/rtutils.md"
        ]
    ]
)

cp("$(@__DIR__)/examples/", "$(@__DIR__)/build/examples/"; force=true)

deploydocs(
    repo="github.com/p2t2/Scruff.jl", 
    branch_previews = "develop")