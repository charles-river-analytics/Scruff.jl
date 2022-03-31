# Scruff

Scruff is an AI framework to build agents that sense, reason, and learn in the world using a variety of models.  
It aims to integrate many different kinds of models in a coherent framework, provide flexibility in spatiotemporal modeling, and provide tools to compose, share, and reuse models and model components.

Warning: Scruff is rapidly evolving beta research software. Although the software already has a lot of functionality, we intend to expand on this in the future and cannot promise stability of the code or the APIs at the moment.

## Installation

First, [download Julia 1.6.0 or later](https://julialang.org/downloads/).

Then, install the Scruff package with the Julia package manager.  From the Julia REPL, type `]` to enter the Pkg REPL mode and then run:

```julia-repl
pkg> add https://github.com/p2t2/Scruff#main
```

## Developing Scruff

To develop Scruff, first pull down the code

```bash
$ git clone https://github.com/p2t2/Scruff.git
```

## Learning about Scruff

Please read the [The Scruff Tutorial](@ref), which describes most of the language features through examples.
The library documentation contains detailed information about most of the data structures
and functions used in the code.

## Contributing to Scruff

We welcome contributions from the community. Please see the issues in Github for some of the improvements 
we would like to make, and feel free to add your own suggestions. 