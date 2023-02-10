include("..//src//utils//logplots.jl")

using Scruff
using Scruff.SFuncs
using Scruff.Utils
using Scruff.RTUtils
using Scruff.Models
import Scruff.Algorithms

pyplot()

x1m = Cat([1,2], [0.1, 0.9])()
x2m = Cat([1,2,3], [0.2, 0.3, 0.5])()
cpd2 = Dict((1,1) => [0.3, 0.7], (1,2) => [0.6, 0.4], (2,1) =>[0.4, 0.6],
            (2,2) => [0.7, 0.3], (3,1) => [0.5, 0.5], (3,2) => [0.8, 0.2])
x3m = DiscreteCPT([1,2], cpd2)()
x4m = DiscreteCPT([1,2], Dict((1,) => [0.15, 0.85], (2,) => [0.25, 0.75]))()
x5m = DiscreteCPT([1,2], Dict((1,) => [0.35, 0.65], (2,) => [0.45, 0.55]))()
x6m = DiscreteCPT([1,2], Dict((1,) => [0.65, 0.35], (2,) => [0.75, 0.25]))()

x1 = x1m(:x1)
x2 = x2m(:x2)
x3 = x3m(:x3)
x4 = x4m(:x4)
x5 = x5m(:x5)
x6 = x6m(:x6)

fivecpdnet = InstantNetwork(Variable[x1,x2,x3,x4,x5], VariableGraph(x3=>[x2,x1], x4=>[x3], x5=>[x3]))

run = Runtime(fivecpdnet)
# default_initializer(run)

logger = BPLogger([:x1,:x2])
@info "logger=$logger"
# run em
with_logger(logger) do
    Scruff.Algorithms.three_pass_BP(run)
end

