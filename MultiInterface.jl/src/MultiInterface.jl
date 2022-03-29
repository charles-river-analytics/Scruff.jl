module MultiInterface

export Interface,
       Policy,
       @interface,
       @impl,
       enable_multiinterface_validation,
       get_policy,
       set_policy,
       with_policy,
       @unpack,
       list_impls,
       get_imp,
       get_method

using MacroTools
import Parameters: with_kw, @unpack

SE = Union{Symbol,Expr}

__policy = nothing
__num_imps = Dict()
__modules = Dict{Symbol,Module}()
# __interface = Dict{Symbol, Signature}()

enable_display() = global display = true
disable_display() = global display = false
display = false

gen_argname() = Symbol(replace(string(gensym("arg")), "#"=>""))

mi_types(e::Expr) = length(e.args) == 1 ? e.args[1] : e.args[2]
mi_types(::Symbol) = :Any

mi_argname(arg::Expr) = length(arg.args) == 1 ? gen_argname() : arg.args[1]
mi_argname(arg::Symbol) = arg

mi_getname(f::Symbol) = f
mi_getname(f::Expr) = f.args[2].value

function register_module(str, mod) 
    global __modules
    push!(__modules, str=>mod)
end

function get_module(mod, str)
    global __modules
    get(__modules, str, mod)
end

# TODO do we want to allow the same interface name in different modules?
function register_interface(source, str, mod, args, retval, wparams)
    global __interface
    haskey(__interface, str) && @warn "reregistering interface for $str at $source" was___=get_interface(str) willbe=Signature(mod, args, retval, wparams)
    push!(__interface, str=>Signature(mod, args, retval, wparams))
end

function get_interface(str)
    get(__interface, str, nothing)
end

function getvars(t::Type)
    v = []

    getvars(t) = nothing
    getvars(t::DataType) = append!(v, collect(t.parameters))
    getvars(t::UnionAll) = begin
        push!(v, t.var)
        getvars(t.body)
    end
    
    unique(v)
end

function get_and_inc_imp_number(imp_type)
    global __num_imps
    if ~(imp_type in keys(__num_imps))
        __num_imps[imp_type] = 0
    end

    __num_imps[imp_type] += 1
    return __num_imps[imp_type]
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

function strip_module(s)
    l = findlast('.', s)
    l === nothing || return s[(l+1):end]
    s
end

function firstcaps_to_lowerunder(s)
    s = strip_module(s)
    return lowercase(strip(replace(s,r"[A-Z]"=>s"_\0"),['_']))
end

function lowerunder_to_firstcaps(s)
    s = strip_module(s)
    return replace(titlecase(s),"_"=>"")
end

nothing2any(::Nothing) = :Any
nothing2any(x::Union{Symbol,Expr}) = x

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
    args = add_missing_argnames(args)
    if display
        println("name: ", f, ": ", typeof(f))
        println("args: ", args, ": ", typeof(args))
        println("returntype: ", R, ": ", typeof(R))
        println("where: ", W, ": ", typeof(W))
    end

    W = isnothing(W) ? () : W

    interface_type_name = Symbol(lowerunder_to_firstcaps(String(f)))

    body = quote
        policy = get_policy()
        return $f(policy, $(get_arg_vars(args)...))
    end
    
    use_policy_body = quote
        imp = get_imp(policy, $interface_type_name, $(get_arg_vars(args)...))
        return $f(imp, $(get_arg_vars(args)...))
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

    register_module(interface_type_name, mod)
    # WIP
    # register_interface(source, interface_type_name, mod, get_arg_types(args), nothing2any(R), W)
    
    result = quote
        abstract type $interface_type_name <: Interface end
        $(combinedef(bare_func_sig_dict))
        $(combinedef(default_func_sig_dict))
    end
    if display
        println(prettify(result))
    end

    return result
end

function get_arg_vars(args)
    tovar(s::Symbol) = s
    tovar(expr::Expr) = tovar(Val(expr.head), expr)
    tovar(::Val{:(::)}, expr) = expr.args[1]
    
    map(tovar, args)
end

function add_missing_argnames(args)
    map(args) do arg
        m = match(r"^::\s*(.*)", string(arg))
        if m !== nothing
            arg = copy(arg)
            insert!(arg.args, 1, gen_argname())
        end
        return arg
    end
end

function get_arg_type_sigs(args)
    map(args) do arg
        m = match(r".*?::\s*(.*)", string(arg))
        s = m === nothing ? :Type : Meta.parse(string("Type{<:", m.captures[1],"}"))
        return Expr(:(::), s)
    end
end

function get_arg_types(args)::Vector{Union{Symbol, Expr}}
    map(args) do arg
        m = match(r".*?::\s*(.*)", string(arg))
        return m === nothing ? :Any : Meta.parse(m.captures[1])
    end
end


function mi_join_args(func_sig_dicts)
    zippedargs = zip([[mi_types(a) for a in fsd[:args]] for fsd in func_sig_dicts]...)
    argnames = [mi_argname(a) for a in first(func_sig_dicts)[:args]]

    function mi_merge_into!(s::Expr, t)
        t in s.args && return s
        push!(s.args, t)
        return s
    end

    do_union(s,t) = :(Union{$s,$t})
    mi_merge(::Val, s::Expr, t::Symbol) = do_union(s,t)
    mi_merge(::Val, s::Expr, ::Val, t::Expr) = do_union(s,t)

    function mi_merge(::Val{:Union}, s::Expr, ::Val{:Union}, t::Expr)
        v = deepcopy(s)
        append!(v.args, t.args)
        unique!(v.args)
        return v 
    end
    
    mi_merge(::Val{:Union}, s::Expr, t::Symbol) = mi_merge_into!(deepcopy(s), t)
    mi_merge(::Val{:Union}, s::Expr, ::Val, t::Expr) = mi_merge_into!(deepcopy(s), t)
    mi_merge(::Val, s::Expr, ::Val{:Union}, t::Expr) = mi_merge(t, s)
    
    mi_merge(s::Symbol, t::Symbol) = s === t ? s : do_union(s,t)
    mi_merge(s::Expr, t::Symbol) = mi_merge(Val(s.args[1]), s, t)
    mi_merge(s::Symbol, t::Expr) = mi_merge(t,s)

    mi_merge(s::Expr, t::Expr) = mi_merge(Val(s.args[1]), s, Val(t.args[1]), t)

    args = [reduce(mi_merge, arg) for arg in zippedargs]
    exprs = map(argnames, args) do n,arg
        :($(n)::$(arg))
    end
    collect(exprs)
end

parameter_names(e::Symbol) = e
parameter_names(e::Expr) = length(e.args) == 1 ? gen_argname() : e.args[1]

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
macro impl(expr)
    return esc(implement_macro(__module__, __source__, expr))
end

function implement_macro(mod, source, implement_expr)
    stripped = MacroTools.prewalk(rmlines, implement_expr)
    T, fields, fname, func_sig_dicts = struct_capture(stripped)

    interface_name = Symbol(lowerunder_to_firstcaps(string(fname)))
    impl_type_name = isnothing(T) ?
                         Symbol(string(interface_name, get_and_inc_imp_number(interface_name))) :
                         T

    first_func_sig_dict = first(func_sig_dicts)
    specific_imp_func_decs = Dict[]
    for func_sig_dict in func_sig_dicts
        specific_imp_func_dec = deepcopy(func_sig_dict)
        specific_imp_func_dec[:args] = [:(impl::$impl_type_name); specific_imp_func_dec[:args]]
        if length(fields) > 0
            fieldnames = [namify(f) for f in fields]
            # Have to do some wonky thing here to get this to parse right
            unpack_expr = :(@unpack $(fieldnames[1:(end-1)]...), $(fieldnames[end]) = impl) 
        else
            unpack_expr = :(begin end)
        end
        specific_imp_func_dec[:body] = quote
            try
                $unpack_expr
                $(specific_imp_func_dec[:body])
            catch
                src = $(string(source))
                @error "error at $(src)"
                rethrow()
            end
        end
        push!(specific_imp_func_decs, specific_imp_func_dec)
    end

    joined_args = mi_join_args(func_sig_dicts)
    joined_params = [parameter_names(e) for e in joined_args]
    default_imp_func_dec = deepcopy(first_func_sig_dict)
    default_imp_func_dec[:args] = [:(impl::Nothing); joined_args]
    default_imp_func_dec[:whereparams] = unique(reduce((x,fsd)->append!(x,get(fsd,:whereparams,[])), func_sig_dicts;init=[]))
    default_imp_func_dec[:body] = quote
        try
            return $(default_imp_func_dec[:name])($impl_type_name(), $(joined_params...))
        catch
            src = $(string(source))
            @error "error at $(src)"
            rethrow()
        end
    end
    # TODO should do a typejoin on all retvals
    default_imp_func_dec[:rtype] = :Any

    if ~(:rtype in keys(first_func_sig_dict))
        first_func_sig_dict[:rtype] = :Any
    end

    interface_module = get_module(mod, interface_name)

    if length(fields) >= 1
        struct_exp = with_kw(:(struct $impl_type_name <: $(interface_module).$(interface_name)
                                   $(fields...)
                               end),
                             @__MODULE__,
                             false)
    else
        struct_exp = quote
            struct $impl_type_name <: $(interface_module).$(interface_name)
                $(fields...)
            end
        end
    end

    wparams = reduce(append!, [get(fsd, :whereparams, []) for fsd in func_sig_dicts]; init=[])
    wparams = unique(wparams)
    argtypes = [[get(fsd,:rtype,:Any);get_arg_types(fsd[:args])] for fsd in func_sig_dicts]
    source_str = string(source)
    
    result = quote
        $struct_exp
        $([combinedef(sifd) for sifd in specific_imp_func_decs]...)
        $(combinedef(default_imp_func_dec))
        @generated function get_method(imp::Type{$impl_type_name})
            m = [m for m in methods($(first_func_sig_dict[:name]), Tuple{$impl_type_name, Vararg{Any}})][1]
            return m
        end
        (s::$impl_type_name)(args...) = $(first_func_sig_dict[:name])(s, args...)
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

function struct_capture(expr)
    if expr.args[1].head === :struct
        @capture(expr.args[1], (struct T_ fields__ end) | (mutable struct T_ fields__ end))
        fs = [splitdef(a) for a in expr.args[2:end]]
    else
        T = nothing
        fields = []
        fs = [splitdef(a) for a in expr.args]
    end
    fname = first(fs)[:name]
    T,fields,fname,fs
end

###
### USING POLICIES
###
function with_policy(f, new_policy)
    last_policy = set_policy(new_policy)
    result = nothing
    try
        result = f()
    catch err
        rethrow()
    finally
        _ = set_policy(last_policy)
    end
    return result
end

end # module
