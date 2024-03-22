using Test
using Scruff
using Scruff.SFuncs
using Scruff.Operators

@testset "net" begin
    @testset "Build NetworkSFunc" begin
        x = NetworkInput{Int}()

        struct AddC <: SFunc{Tuple{Int},Int}
            c::Int
        end

        addc = AddC(2)

        struct MulC <: SFunc{Tuple{Int},Int}
            c::Int
        end

        struct Add <: SFunc{Tuple{Int,Int},Int}
        end

        mulc = MulC(-1)

        add = Add()

        sfuncs = (addc, mulc, add)
        parents = Dict{SFunc,Any}(addc=>[x], mulc=>[x], add=>[addc, mulc])
        outputs = (add,)

        net = NetworkSFunc((x,),
                           sfuncs,
                           parents,
                           outputs)

        @test 1 == 1
    end

    @testset "Netwalk Ops" begin
        x = NetworkInput{Int}()

        x_y = DiscreteCPT([1,2], Dict((1,) => [0.9, 0.1], (2,) => [0.1, 0.9]))
        y_z = DiscreteCPT([1,2], Dict((1,) => [0.1, 0.9], (2,) => [0.9, 0.1]))

        sfuncs = (x_y, y_z)
        parents = Dict{SFunc,Any}(x_y=>[x],y_z=>[x_y])
        outputs = (y_z,)
        
        net = NetworkSFunc((x,),
                           sfuncs,
                           parents,
                           outputs)

        y1 = sample(net, (1,)) 
        y2 = sample(net, (1,))
        cpdf2 = logcpdf(net, (1,), y2)
        # cpdf2, y2 = sample_logcpdf(net, 1)

        N = 100000
        correct = [0.18 0.82
                   0.82 0.18]
        for x in 1:2
            for z in 1:2
                cpdf3 = 0.0
                for i in 1:N
                    cpdf3 += exp(logcpdf(net, (x,), z))
                end
                cpdf3 = cpdf3/N
                # println("P(z=$z|x=$x) = $cpdf3")
                @test isapprox(correct[x,z], cpdf3, rtol=5e-2)
            end
        end
    end
end
