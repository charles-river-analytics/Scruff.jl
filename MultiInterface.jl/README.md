# MultiInterface.jl
Defining and selecting alternative parameterized implementations of an interface in Julia

```
@interface a(x::Int)::Int

@impl function a(x::Int)
    struct A1 end
    return 1
end

@impl function a(x::Int)
    struct A2
        time::AbstractFloat = 0.01
    end
    sleep(time)
    return 2
end

struct Policy1 <: Policy end
get_imp(policy::Policy1, ::Type{A}, args...) = A1()

struct Policy2 <: Policy end
get_imp(policy::Policy2, ::Type{A}, args...) = A2()

with_policy(Policy1()) do
    @test a(0)==1
end
with_policy(Policy2()) do
    @test a(0)==2
end
```

See tests for more examples. See test/demo.jl for an in-line example of the macro expanded representations. This may not be exactly up-to-date.


## Debugging
Debugging implementations may be a bit tricky right now. Currently we don't have a `NotImplemented` fallthrough call for reasons similar to outlined here: https://www.oxinabox.net/2020/04/19/Julia-Antipatterns.html. This would also preclude certain sophisticated usage of `hasmethod` by complex Policies. `methods(f)` can help demonstrate issues with calling implementations.
