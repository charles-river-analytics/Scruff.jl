
[![][docs-main-img]][docs-main-url][![][docs-dev-img]][docs-dev-url]&nbsp;&nbsp;[![][CI-img]][CI-url]&nbsp;&nbsp;[![][codecov-img]][codecov-url]

# Scruff.jl

Scruff is an AI framework to build agents that sense, reason, and learn in the world using a variety of models.  It aims to integrate many different kinds of models in a coherent framework, provide flexibility in spatiotemporal modeling, and provide tools to compose, share, and reuse models and model components.

Scruff is provided as a [Julia](https://julialang.org/) package and is licensed under the BSD-3-Clause License.  It should be run using Julia v1.6 or v1.7.

> *Warning*: Scruff is rapidly evolving beta research software. Although the software already has a lot of functionality, we intend to expand on this in the future and cannot promise stability of the code or the APIs at the moment.

## Download and Installation

To download the package, from the Julia package manager, run

```julia-repl
(v1.7) pkg> add Scruff
```

## Scruff Tutorial and Examples

The Scruff tutorial can be found in the [tutorial](https://p2t2.github.io/Scruff.jl/stable/tutorial/tutorial/) section of the documentation.

Scruff examples can be found in the [examples/](docs/examples/) directory.

## Building the documentation

Scruff uses [Documenter.jl](https://juliadocs.github.io/Documenter.jl/stable/) to generate its documentation.  To build, navigate to the `docs` folder and run

```julia
Scruff.jl\docs> julia --project=.. --color=yes make.jl
```

This will create a `docs/build` directory with an `index.html` file, which will contain the documentation.

## Running tests

To run the tests, activate the project as above and just run `test` from the `pkg` prompt.  From the `julia` prompt, `include("test/runtests.jl")` can be used to run the tests.

## Development

Development against the Scruff codebase should _only_ be done by branching the `develop` branch.

### Scruff module layout

The Scruff packages are split into four (4) main modules:  `Models`, `Algorithms`, `SFuncs`, and `Operators`.

- To add to the `Models` module, add a `.jl` file to the `src/models/` directory and `include` it in the `src/models.jl` file
- To add to the `Algorithms` module, add a `.jl` file to the `src/algorithms/` directory and `include` it in the `src/algorithms.jl` file
- To add to the `SFuncs` module, add a `.jl` file to the `src/sfuncs/` directory and `include` it in the `src/sfuncs.jl` file
- To add to the `Operators` module, add a `.jl` file to the `src/operators` directory and `include` it in the `src/operators.jl` file

[docs-main-img]: https://img.shields.io/badge/docs-main-blue.svg
[docs-main-url]: https://charles-river-analytics.github.io/Scruff.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://charles-river-analytics.github.io/Scruff.jl/dev

[CI-img]: https://github.com/p2t2/Scruff.jl/actions/workflows/ci.yml/badge.svg
[CI-url]: https://github.com/p2t2/Scruff.jl/actions/workflows/ci.yml

[codecov-img]: https://codecov.io/gh/p2t2/Scruff.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/p2t2/Scruff.jl

