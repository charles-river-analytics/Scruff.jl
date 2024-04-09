module MultiInterface

export Interface,
       Policy,
       @interface,
       @impl,
       impl,
       __policy,
       get_policy,
       set_policy,
       with_policy,
       @unpack,
       list_impls,
       get_method,
       get_imp,
       get_impl

using MacroTools
import Parameters: with_kw, @unpack

__policy = nothing
num_imps = Dict()

display = false

# This is used to give each impl struct a unique name
function get_and_inc_imp_number(imp_type)
    global num_imps
    if ~(imp_type in keys(num_imps))
        num_imps[imp_type] = 0
    end

    num_imps[imp_type] += 1
    return num_imps[imp_type]
end

function get_policy()
    global __policy
    return __policy
end

function set_policy(new_policy)
    global __policy
    old_policy = __policy
    __policy = new_policy
    return old_policy
end

abstract type Interface end

abstract type Policy end
get_imp(policy::Policy, args...) = nothing

OptionalPolicy = Union{Policy, Nothing}

function list_impls(imp_type, args_type=nothing)
    if isnothing(args_type)
        args_type = Tuple{<:Interface, Vararg{Any}}
    else
        args_type = Tuple{<:Interface, args_type.parameters...}
    end
    return [get_impl(m.sig) for m in methods(imp_type, args_type)]
end
function get_impl(t::UnionAll)
    return get_impl(t.body)
end
function get_impl(t::DataType)
    return t.parameters[2]
end

function firstcaps_to_lowerunder(s)
    return lowercase(strip(replace(s,r"[A-Z]"=>s"_\0"),['_']))
end

function lowerunder_to_firstcaps(s)
    return replace(titlecase(s),"_"=>"")
end


"""
    macro interface(interface_exp)
Called
```
@interface a(x::I)::O where {I,O}
```
Generates the following code:
```
abstract type A <: Interface end
function (a(x::I; )::O) where {I,O}
    policy = get_policy()
    return a(policy, x::I)
end
function (a(policy::Policy, x::I; )::O) where {I,O}
    imp = get_imp(policy, A, x::I)
    return a(imp, x::I)
end
```
"""
macro interface(interface_exp)
    return esc(interface_macro(__module__, __source__, interface_exp))
end

function interface_macro(mod, source, interface_exp)
    f, args, R, W = interface_capture(interface_exp)
    if display
        println("name: ", f, ": ", typeof(f))
        println("args: ", args, ": ", typeof(args))
        println("returntype: ", R, ": ", typeof(R))
        println("where: ", W, ": ", typeof(W))
    end
    println(stderr, f)

    W = isnothing(W) ? () : W

    interface_type_name = Symbol(lowerunder_to_firstcaps(String(f)))

    body = quote
        policy = get_policy()
        return $f(policy, $(remove_types_from_args(args)...))
    end

    use_policy_body = quote
        imp = get_imp(policy, $interface_type_name, $(remove_types_from_args(args)...))
        return $f(imp, $(remove_types_from_args(args)...))
    end

    bare_func_sig_dict = Dict(:name => f,
                              :args => args,
                              :kwargs => Any[],
                              :body => body,
                              :rtype => R,
                              :whereparams => W,
                              )

    default_func_sig_dict = Dict(:name => f,
                                 :args => [:(policy::Policy); args],
                                 :kwargs => Any[],
                                 :body => use_policy_body,
                                 :rtype => R,
                                 :whereparams => W,
                                 )

    result = quote
        abstract type $interface_type_name <: Interface end
        $(combinedef(bare_func_sig_dict))
        $(combinedef(default_func_sig_dict))
        export $interface_type_name, $f
    end
    if display
        println(prettify(result))
    end

    return result
end

"""
    macro impl(expr)
Called
```
@impl begin
    struct MyA
        precision::Int64
    end
    function a(x::Int)
      c = 1
      return x + c + precision
    end
end
```
The `MyA` can be considered an identifier for this particular method, as opposed to different implementations of 
this function.
This macro generates the following code:
```
begin
    Base.@__doc__ struct MyA <: A
            precision::Int64
            MyA(; precision = error("Field '" * "precision" * "' has no default, supply it with keyword.")) = MyA(precision)
            MyA(precision) = new(precision)
        end
    ()
    ()
    MyA(pp::MyA; kws...) = (Parameters).reconstruct(pp, kws)
    MyA(pp::MyA, di::(Parameters).AbstractDict) = (Parameters).reconstruct(pp, di)       
    MyA(pp::MyA, di::Vararg{Tuple{Symbol, Any}}) = (Parameters).reconstruct(pp, di)      
    nothing
    macro unpack_MyA(ex)
        esc((Parameters)._unpack(ex, Any[:precision]))
    end
    macro pack_MyA()
        esc((Parameters)._pack_new(MyA, Any[:precision]))
    end
    MyA
    function a(impl::MyA, x::Int; )
        @unpack (precision,) = impl
        c = 1
        return x + c + precision
    end
    function a(policy::Nothing, x::Int; )
        return a(MyA(), x::Int)
    end
    @generated function get_method(imp::Type{MyA})
            m = ([m for m = methods(a, Tuple{MyA, Vararg})])[1]
            return m
        end
    (s::MyA)(args...) = a(s, args...)
end
```
"""

macro impl(expr, interface_module)
    #=
    interface_name = get_interface_name(__module__, __source__, expr)
    interface = A
    interface = eval(:A)
    interface = eval(interface_name)
    interface_module = eval(:(parentmodule($interface_name)))
    println("Module: $(interface_module)")
    =#
    return esc(impl(__module__, __source__, expr, interface_module))
end

macro impl(expr)
    return esc(impl(__module__, __source__, expr, nothing))
end

synth_arg_if_none(s::Symbol) = s
function synth_arg_if_none(exp)
    if length(exp.args)==2
        return exp
    else
        x = gensym()
        return :($x::$(exp.args[1]))
    end
end

function def_args_to_call_args(def_args)
    # Mostly handle edge case where def args has no variable placeholder
    # E.g. given defined f(x::Int, ::Vector) need to call f(x::Int, _::Vector)
    return [synth_arg_if_none(exp) for exp in def_args]
end

just_arg_var(s::Symbol) = s
function just_arg_var(exp)
    return exp.args[1]
end

function remove_types_from_args(call_args)
    return [just_arg_var(exp) for exp in call_args]
end

function get_interface_name(mod, source, implement_expr)
    impl_parts = collect([(implement_expr.args[2*i - 1], implement_expr.args[2*i]) for i in 1:Int(length(implement_expr.args) / 2)])

    local impl_type_name, fields, op_name
    if last(impl_parts[1]).head == :struct
        struct_part = impl_parts[1]
        #println("struct part: $struct_part")
        impl_type_name, fields = struct_capture(last(struct_part))
        impl_parts = impl_parts[2:end]

        op_name = splitdef(last(impl_parts[1]))[:name]
        interface_name = Symbol(lowerunder_to_firstcaps(String(op_name)))
    else
        op_name = splitdef(last(impl_parts[1]))[:name]
        interface_name = Symbol(lowerunder_to_firstcaps(String(op_name)))
    end

    return interface_name
end

function impl(mod, source, implement_expr, interface_module)
    #println("impl_expr: $(implement_expr.head), $(implement_expr.args)")
    #impl_parts::Vector{Tuple{LineNumberNode, <:Expr}} = collect([(implement_expr.args[2*i - 1], implement_expr.args[2*i]) for i in 1:Int(length(implement_expr.args) / 2)])
    impl_parts = collect([(implement_expr.args[2*i - 1], implement_expr.args[2*i]) for i in 1:Int(length(implement_expr.args) / 2)])

    #=
    line_number_node, block = implement_expr.args[1], implement_expr.args[2]
    if block.head == :block
        impl_parts = [(block.args[i], block.args[i + 1]) for i in range(length(block.args) / 2)]
    else
        impl_parts = [(line_number_node, block)]
    end
    =#

    #println("impl_parts: $(impl_parts)")
    first_ln_node = first(impl_parts[1])
    local impl_type_name, fields, op_name
    if last(impl_parts[1]).head == :struct
        struct_part = impl_parts[1]
        #println("struct part: $struct_part")
        impl_type_name, fields = struct_capture(last(struct_part))
        impl_parts = impl_parts[2:end]

        op_name = splitdef(last(impl_parts[1]))[:name]
        interface_name = Symbol(lowerunder_to_firstcaps(String(op_name)))
    else
        op_name = splitdef(last(impl_parts[1]))[:name]
        interface_name = Symbol(lowerunder_to_firstcaps(String(op_name)))

        impl_type_name = Symbol(string(interface_name, get_and_inc_imp_number(interface_name)))
        fields = []
    end

    # TODO: Get mutable from declaration?
    local struct_exp
    if length(fields) >= 1
        struct_exp = with_kw(:(mutable struct $impl_type_name <: $interface_name
                                   $(fields...)
                               end),
                             mod,
                             false)
    else
        struct_exp = quote
            mutable struct $impl_type_name <: $interface_name
                $(fields...)
            end
        end
    end

    if interface_module isa Nothing
        qualified_name = op_name
    else
        qualified_name = :($interface_module.$op_name)
    end

    function create_impl_defs(func_sig_dict)
        # Add impl as first argument, then assign each field in the struct to a local variable at beginning of def
        specific_imp_func_dec = deepcopy(func_sig_dict)
        specific_imp_func_dec[:name] = qualified_name
        specific_imp_func_dec[:args] = [:(impl::$impl_type_name); specific_imp_func_dec[:args]]
        if length(fields) > 0
            fieldnames = [namify(f) for f in fields]
            # Have to do some wonky thing here to get this to parse right
            unpack_expr = :(@unpack $(fieldnames[1:(end-1)]...), $(fieldnames[end]) = impl) 
        else
            unpack_expr = :(begin end)
        end
        specific_imp_func_dec[:body] = quote
            $unpack_expr
            $(specific_imp_func_dec[:body])
        end

        default_imp_func_dec = deepcopy(func_sig_dict)
        call_args = def_args_to_call_args(default_imp_func_dec[:args])
        #call_args_no_types = remove_types_from_args(call_args)
        call_args_no_types = call_args
        default_imp_func_dec[:name] = qualified_name
        default_imp_func_dec[:args] = [:(impl::Nothing); call_args]
        default_imp_func_dec[:body] = :(return $(default_imp_func_dec[:name])($impl_type_name(),
                                                                              $(call_args_no_types...)))

        if ~(:rtype in keys(func_sig_dict))
            func_sig_dict[:rtype] = Any
        end

        return (specific_imp_func_dec, default_imp_func_dec)
    end

    impl_defs = [(linenum, create_impl_defs(splitdef(funcdef))) for (linenum, funcdef) in impl_parts]

    function make_block(impl_def)
        (line_num, (specific_imp_func_dec, default_imp_func_dec)) = impl_def
        return quote
            $line_num
            $(combinedef(specific_imp_func_dec))
            $(combinedef(default_imp_func_dec))
        end 
    end

    func_blocks = [make_block(impl_def) for impl_def in impl_defs]

    result = quote
        $first_ln_node
        $struct_exp
        # Constructor
        (s::$impl_type_name)(args...) = $(op_name)(s, args...)
        $(func_blocks...)
        @generated function get_method(imp::Type{$impl_type_name})
            m = [m for m in methods($(op_name), Tuple{$impl_type_name, Vararg{Any}})][1]
            return m
        end
        export $impl_type_name #, $op_name, $interface_name
    end
    if display
        println(prettify(result))
    end
    return result
end

function interface_capture(interface_exp)
    @capture(interface_exp, (f_(args__)::R_ where W__) |
                            (f_(args__) where W__) |
                            (f_(args__)::R_) |
                            (f_(args__)))
    return f, args, R, W
end


function struct_capture(struct_exp::Expr)
    @capture(struct_exp, struct T_
                             fields__
                         end |
                         mutable struct T_
                             fields__
                         end |
                         struct T_ end |
                         mutable struct T_ end)
    if isa(fields, Nothing)
      fields = []
    end
    return T, fields
end

###
### USING POLICIES
###
function with_policy(f, new_policy)
    last_policy = set_policy(new_policy)
    result = nothing
    try
        result = f()
    finally
        _ = set_policy(last_policy)
    catch err
        rethrow()
    end
    return result
end

end # module
