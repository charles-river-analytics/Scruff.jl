# Scruff.jl

Scruff is an AI framework to build agents that sense, reason, and learn in the world using a variety of models.  It aims to integrate many different kinds of models in a coherent framework, provide flexibility in spatiotemporal modeling, and provide tools to compose, share, and reuse models and model components.

Scruff is provided as a [Julia](https://julialang.org/) package and is licensed under the BSD-3-Clause License.

> *Warning*: Scruff is rapidly evolving beta research software. Although the software already has a lot of functionality, we intend to expand on this in the future and cannot promise stability of the code or the APIs at the moment.

## Download and Installation

To download the package, from the Julia package manager, run

```julia-repl
(v1.7) pkg> add https://github.com/p2t2/Scruff.jl
```

## Building the documentation

Scruff uses [Documenter.jl](https://juliadocs.github.io/Documenter.jl/stable/) to generate its documentation.  To build, navigate to the `docs` folder and run

```julia
Scruff.jl\docs> julia --project=.. --color=yes make.jl
```

This will create a `docs/build` directory with an `index.html` file, which will contain the documentation.

## Running tests

To run the tests, activate the project as above and just run `test` from the `pkg` prompt.  From the `julia` prompt, `include("test/runtests.jl")` can be used to run the tests.

## Development

The source can be cloned from https://github.com/p2t2/Scruff.jl.git.

The Scruff packages are split into four (4) main modules:  `Models`, `Algorithms`, `SFuncs`, and `Operators`.

- To add to the `Models` module, add a `.jl` file to the `src/models/` directory and `include` it in the `src/models.jl` file
- To add to the `Algorithms` module, add a `.jl` file to the `src/algorithms/` directory and `include` it in the `src/algorithms.jl` file
- To add to the `SFuncs` module, add a `.jl` file to the `src/sfuncs/` directory and `include` it in the `src/sfuncs.jl` file
- To add to the `Operators` module, add a `.jl` file to the `src/operators` directory and `include` it in the `src/operators.jl` file
