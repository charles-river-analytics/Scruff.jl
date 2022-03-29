# WIP
import Base: ==, hash

"""
struct Scope

The `Scope` tracks a typevar and its 'rules' or 'constraints'.  For example, 
it represents `:(T where T <: AbstractFloat)`.
"""
struct Scope
    var::Symbol
    rule::Union{Nothing, Expr}
end
Scope(var::Symbol) = Scope(var, nothing)

as_scope_expr(s::Scope) = s.rule === nothing ? s.var : s.rule

"""
function rule_symbols(s::Scope)

From a [`Scope`](@ref) extract all typevar symbols
"""
function rule_symbols(s::Scope)
    extract_symbols(e::Symbol) = [e]
    extract_symbols(e::Expr) = extract_symbols(Val(e.head), e)
    extract_symbols(::Val{:(.)}, e) = extract_symbols(strip_module(e))
    extract_symbols(::Val{:curly}, e) = mapreduce(extract_symbols,append!, e.args[2:end];init=Symbol[])
    extract_symbols(::Val{:where}, e) = extract_symbols(e.args[1])
    function extract_symbols(::Union{Val{:(>:)},Val{:(<:)}}, e)
        if length(e.args) == 1 
            extract_symbols(e.args[1])
        else
            if s.var === e.args[1] 
                extract_symbols(e.args[2])
            else
                [extract_symbols(e.args[1]); extract_symbols(e.args[2])...]
            end
        end 
    end

    s.rule === nothing && return Symbol[]
    extract_symbols(s.rule)
end

"""
function replace_rule_symbol!(expr::Expr, s, t)

In the given expression, replace all references to the symbol `s` with the
symbol `t`.
"""
function replace_rule_symbol!(expr::Expr, s, t)
    rrs(e::Symbol) = e === s ? t : e
    rrs(e::Expr) = (rrs(Val(e.head), e); e)
    rrs(::Val{:where}, e) = (rrs(e.args[1]); e)
    function rrs(::Val{:curly}, e)
        for (i,a) in enumerate(e.args[2:end])
            e.args[i+1] = rrs(a)
        end
        e
    end
    function rrs(::Union{Val{:(>:)},Val{:(<:)}}, e)
        e.args[1] = rrs(e.args[1])
        if length(e.args) > 1 
            e.args[2] = rrs(e.args[2])
        end 
        e
    end

    rrs(expr)
end

Base.hash(obj::Scope, h::UInt) = hash(obj.var)
==(o1::Scope, o2::Scope) = o1.var === o2.var
==(o1::Scope, o2::Symbol) = o1.var === o2
==(o1::Symbol, o2::Scope) = o2.var === o1

"""
struct Dependencies

`Dependencies` tracks the set of implementation's expressions that are bound to a 
given interface's [`Scope`](@ref), and the dependencies on those bindings, so
every `Scope` has a related `Dependencies` struct (perhaps empty).  

The `bindings` field contains the set of implementation's expressions dependent on 
a `Scope`, while the `dependencies` field maps the specific interface's scope to the 
implementation's signature's expression.
"""
struct Dependencies
    bindings::Vector
    dependencies::Vector{<:Pair}

    Dependencies() = new(Any[], Pair[])
end

"""
    Signature

A structure that holds information about the signature of a method; it includes the 
expression for every argument and the return value, along with the evaluated types
for each, and includes the type parameters following the method (the `where` clause)
"""
struct Signature
    mod::Module
    args::Vector{<:SE}
    args_evald::Vector
    retval::SE
    retval_evald
    wparams

    function Signature(mod::Module, args::Vector{<:SE}, retval::SE, wparams)
        newargs,newretval = add_wparams_to_args(wparams, args, retval)

        # @debug "Signature" args=args retval=retval wparams=wparams

        # here we evaluate the args, and then recreate the Expressions
        # for each arg; the idea is to normalize type expressions
        args_evald = [mod.eval(a) for a in newargs]
        newargs_from_types = map(zip(args_evald, args)) do (arg,oldarg)
            e = strip_module(Meta.parse(string(arg)))
            replace_vector_with_array(e)
            e
        end

        newarg_exprs = map(zip(args_evald, args, newargs, newargs_from_types)) do (t,old,a,b)
            e = decide_between_exprs(t,a,b,wparams)
            e = strip_added_where(old, e, wparams)
        end

        # @debug "Signature" newargs=newargs newargs_from_types=newargs_from_types evald=args_evald newarg_exprs=newarg_exprs
        
        retval_evald = mod.eval(newretval)
        newretval_from_type = strip_module(Meta.parse(string(retval_evald)))
        replace_vector_with_array(newretval_from_type)
        
        newretval_expr = decide_between_exprs(retval_evald, newretval, newretval_from_type, wparams)

        newretval_expr = strip_module(newretval_expr)
        newretval_expr = strip_added_where(retval, newretval_expr, wparams)

        new(mod,
            SE[na for na in newarg_exprs], 
            args_evald, 
            newretval_expr, 
            retval_evald,
            wparams)
    end
end

function add_wparams_to_args(wparams, args, retval)
    if isempty(wparams)
        newargs = args
        newretval = retval
    else
        newargs = map(args) do arg 
            # wparams will include all typevars, but it works
            Expr(:where, [arg; wparams...]...)
        end
        newretval = Expr(:where, [retval; wparams...]...) 
    end
    (newargs, newretval)
end

function strip_added_where(old, new::Symbol, wparams) 
    # @info "strip_added_where(Any,Symbol)" old=old new=new
    new
end

function strip_added_where(old::Symbol, new::Expr, wparams)
    if new.head === :where
        try_replace_symbol!(new, wparams)
    end
    new
end

#=
    old = :Score
    new = :(Score{I} where I)
    transform to :(Score{I} where I)

    old = :(Score{<:O})
    new = :(Score{var"#s24"} where {O, var"#s24" <: O})
    transform to :(Score{var"#s24"} where {var"#s24" <: O})
=#
function strip_added_where(old::Expr, new::Expr, wparams)
    inwhere(s::Symbol) = Any[s]
    inwhere(e::Expr) = inwhere(Val(e.head), e)
    inwhere(::Union{Val{:(<:)}, Val{:(>:)}}, e) = mapreduce(arg->inwhere(arg), append!, e.args; init=Any[])
    inwhere(::Union{Val{:where},Val{:curly}}, e) = mapreduce(arg->inwhere(arg), append!, e.args[2:end]; init=Any[])

    incurly(x) = Any[]
    incurly(s::Symbol) = Any[s]
    incurly(e::Expr) = incurly(Val(e.head), e)
    incurly(e::QuoteNode) = incurly(e.value)
    incurly(::Val{:(.)}, e) = incurly(e.args[2])
    incurly(::Val{:curly}, e) = mapreduce(arg->incurly(arg), append!, e.args[2:end]; init=Any[])
    incurly(::Union{Val{:(<:)}, Val{:(>:)}}, e) = incurly(e.args[1])
    
    isrelated(::Val, e, tokeep) = false
    function isrelated(::Union{Val{:(<:)}, Val{:(>:)}}, e, tokeep) 
        isrelated(e.args[1], tokeep) || (length(e.args) > 1 && isrelated(e.args[2], tokeep))
    end
    isrelated(::Val{:curly}, e, tokeep) = any(arg->isrelated(arg,tokeep), e.args[2:end])
    isrelated(s::Symbol, tokeep) = s in tokeep
    isrelated(e, tokeep) = isrelated(Val(e.head), e, tokeep)

    extract_related(s::Symbol) = s
    extract_related(e::Expr) = extract_related(Val(e.head), e)
    extract_related(::Union{Val{:(<:)}, Val{:(>:)}}, e::Expr) = e.args[1]

    old == new && return new

    inwhere = old.head === :where ? inwhere(Val(old.head), old) : Any[]
    incurly = old.head === :curly ? incurly(Val(old.head), old) : Any[]

    if new.head === :where
        toremove = Int64[]
        for (i,arg) in enumerate(new.args[2:end])
            if !(isrelated(arg, inwhere)) && (arg in incurly)
                push!(toremove, i+1)
            end
        end
        deleteat!(new.args, toremove)
        if length(new.args) == 1
            new = new.args[1]
        else
            try_replace_symbol!(new, wparams)
        end
    end

    new
end

function try_replace_symbol!(new, wparams) # new is a :where Expr
    # We skip replacing :(T where T) constructs; we
    # only handle :(X{I,O} where O) constructs
    new.args[1] isa Expr || return
    new.args[1].head === :curly || return

    shouldreplace(arg, p::Expr) = shouldreplace(arg, p.args[1])
    shouldreplace(arg, p::Symbol) = arg === p

    function replace_symbol(::Val, e::Expr, arg, sym) # :curly, :(<:), :(>:)
        for i in 1:length(e.args)
            e.args[i] = replace_symbol(e.args[i], arg, sym)
        end
        e
    end
    function replace_symbol(::Val{:curly}, e::Expr, arg, sym)
        for i in 2:length(e.args)
            e.args[i] = replace_symbol(e.args[i], arg, sym)
        end
        e
    end
    replace_symbol(e, arg, sym) = e
    replace_symbol(e::Symbol, arg, sym) = e === arg ? sym : e
    replace_symbol(e::Expr, arg, sym) = replace_symbol(Val(e.head), e, arg, sym)

    for p in wparams
        for i in 2:length(new.args)
            arg = new.args[i]
            if shouldreplace(arg, p)
                replace_symbol(new, arg, gensym())
            end
        end
    end
end


#= We prefer to use the parse(string(eval(expr))) version (the b parameter), 
   since it tends to be the most complete.  In certain cases,
   though, (e.g. (T where T)) it removes our typevars and
   convert it to `Any`.  In the first case, see below where the `C` is
   not specified.

   ```
   struct Foo{A,B,C} end

   foo(f::Foo{A,B}) where {A,B} = (A,B)
   ```

   Once we eval and reparse the type, we get the `C` back.

   We have to handle all special cases, such as Vectors, which
   are aliases, and 'decompile' differently depending on whether
   there is a typevar:  e.g. Vector{T} where T => Array{T,1} where T
   while Vector{Int64} => Vector{Int64}.  We convert all Vector
   references to Array references.
=# 
function decide_between_exprs(t, a, b, wparams)
    # this handles the case where we have Q where Q <: Tuple, so b == :Tuple but
    # we want to keep Q where Q <: Tuple
    inwparams = in(a isa Expr ? a.args[1] : a, map(x->x isa Expr ? x.args[1] : x, wparams))

    # @debug "mapping newargs" a=a b=b t=t inwparams=inwparams
    if t == Vector
        b
    elseif t <: Vector
        x = copy(b)
        replace_vector_with_array(x)
        x
    elseif t == Any || inwparams
        a
    else
        b
    end

end

replace_vector_with_array(curr) = nothing

function replace_vector_with_array(expr::Expr)
    for (i,arg) in enumerate(expr.args)
        if arg === :Vector
            expr.args[i] = :Array
            if i == 1
                push!(expr.args, 1)
            end
        else 
            replace_vector_with_array(arg)
        end
    end
end

body(t::UnionAll) = body(t.body)
body(t::Type) = t

extract_vars(t::Type, a::Vector) = nothing
function extract_vars(t::UnionAll, a::Vector)
    push!(a, t.var.name)
    extract_vars(t.body, a)
end


function extract_vars(::Val{:where}, t, a)
    append!(a, t.args[2:end])
    extract_vars(t.args[1], a)
end

function extract_vars(::Union{Val{:(<:)}, Val{:(>:)}}, t, a)
    for arg in t.args[1:end]
        extract_vars(arg, a)
    end
end

function extract_vars(::Val{:curly}, t, a)
    for arg in t.args[2:end]
        extract_vars(arg, a)
    end
end

extract_vars(t::Expr, a::Vector) = extract_vars(Val(t.head), t, a)
extract_vars(t::Symbol, a::Vector) = begin
    if startswith(string(t), "#")
        push!(a, t)
    end
end
extract_vars(::Number, a::Vector) = nothing
extract_vars(t) = (a = []; extract_vars(t, a); a)

function is_number(s)
    rx = r"^\d+(?:\.\d*)?$"
    match(rx, s) !== nothing
end

function wrap_in_where(expr, wparams)
    isempty(wparams) ? expr : Expr(:where, [expr;unique(wparams)...]...)
end

wheres(scope, sig) = unique(mapreduce(d->[as_scope_expr(x) for x in keys(d)], append!, scope; init=Any[sig.wparams...])) 

function eval_expr(mod, expr, params)
    is_number(string(expr)) && return mod.eval(expr)
    if expr isa Expr && length(expr.args) == 1 && (expr.head == :(<:) || expr.head == :(>:))
        s = gensym()
        pushfirst!(expr.args, s)
        whered = wrap_in_where(expr, [s])
    end 
    whered = wrap_in_where(expr, params)

    @debug "eval_expr" whered=whered

    evald = try
        mod.eval(whered)
    catch e
        if e isa UndefVarError
            @debug "Handling eval $whered; returning Any"
            Any
        else
            rethrow()
        end
    end

    return evald
end

function eval_in_union(mod, i_expr, s_expr, i_params, s_params, errors)
    i_eval = eval_expr(mod, i_expr, i_params)
    s_eval = eval_expr(mod, s_expr, s_params)

    @debug "eval_in_union" i_eval=i_eval s_eval=s_eval
    if !(s_eval isa Core.TypeofVararg || i_eval isa Core.TypeofVararg) && !(s_eval <: i_eval)
        push!(errors, """Invalid Union:
            \tdefined :  $(s_eval)
            \texpected:  $(i_eval)
            \tdefined :  $(s_expr)
            \texpected:  $(i_expr)""")
        return (false, Nothing, i_expr, s_expr)
    end
    
    # new_iexpr = Meta.parse(string(i_eval))
    # replace_vector_with_array(new_iexpr)
    # new_sexpr = Meta.parse(string(s_eval))
    # replace_vector_with_array(new_sexpr)

    return (true, s_eval, i_expr, s_expr)
end


ntharg(sig::Signature, n) = n > length(sig.args_evald) ? sig.retval_evald : sig.args_evald[n]

# we added the first where clause to every argument
stripwheres(s::Symbol, wheres) = s
function stripwheres(e::Expr, wheres) 
    gf(a) = a
    gf(a::Expr) = a.args[1]

    todel = Int64[]
    e.head === :where || return e
    c = copy(e)
    for (i,arg) in enumerate(c.args[2:end])
        for w in wheres
            if gf(arg) == gf(w)
                push!(todel, i+1)
            end
        end
    end

    deleteat!(c.args, todel)
    length(c.args) == 1 ? c.args[1] : c
end

find_scope(::Val, expr, v) = (expr, v)

function find_scope(::Val{:where}, expr, v)
    push!(v, expr.args[2:end]...)
    find_scope(expr.args[1], v)
end

find_scope(expr) = (expr, [])
find_scope(expr::Expr) = find_scope(Val(expr.head), expr, [])

find_scope(expr, v) = (expr, v)
find_scope(expr::Expr, v) = find_scope(Val(expr.head), expr, v)

to_scope(::Val{:(>:)}, expr) = Scope(expr.args[1], expr)
to_scope(::Val{:(<:)}, expr) = Scope(expr.args[1], expr)
to_scope(expr::Symbol) = Scope(expr)
to_scope(expr::Expr) = to_scope(Val(expr.head), expr)

strip_module(expr::Expr, args1::Expr) = begin
    expr.args[1] = strip_module(Val(args1.head), args1)
    expr
end 
strip_module(expr::Expr, args1::Symbol) = expr 

strip_module(::Union{Val{:curly}, Val{:where}}, expr) = strip_module(expr, expr.args[1])
strip_module(::Val{:(.)}, expr) = strip_module(expr.args[2])

strip_module(expr::QuoteNode) = expr.value
strip_module(expr::Symbol) = expr
strip_module(expr::Expr) = strip_module(Val(expr.head), expr)

in_curly(::Val, expr) = (expr, [])
in_curly(::Val{:curly}, expr) = (expr.args[1], collect(expr.args[2:end]))
in_curly(expr::Symbol) = (expr, [])
in_curly(expr::Expr) = in_curly(Val(expr.head), expr)

create_scope_dicts(list) = Dict(map(x->to_scope(x)=>Dependencies(), list))

function check_supertype_sig(ifc_expr::Expr, ifc_scope, sig_scope, ifc, sig, sigarg, errors, source)
        # take a look at the supertype
        sigarg === Any && return

        @debug "check_supertype_sig" ifc_expr=ifc_expr sigarg=sigarg

        super_sig = supertype(sigarg)
        super_expr = strip_module(Meta.parse(string(super_sig)))

        @debug "check_supertype_sig" super_expr=super_expr
        # update the wparam context for the signature
        d = last(sig_scope)
        sig_expr, sig_sub_scope = find_scope(super_expr)
        replace_vector_with_array(sig_expr)
        merge!(d, create_scope_dicts(sig_sub_scope))

        bind_sig_symbols(ifc_expr, sig_expr, ifc_scope, sig_scope, ifc, sig, super_sig, errors, source)
end

function collect_unions(expr)
    base, typevars = in_curly(expr)
    base === :Union && return mapreduce(collect_unions, append!, typevars; init=[])
    [expr]
end

function collect_orset_to_check(ifc_expr, sig_expr, errors)
    ifc_unions = collect_unions(ifc_expr)
    sig_unions = collect_unions(sig_expr)

    # handle the `Union{}` case
    if xor(isempty(ifc_unions), isempty(sig_unions))
        return [(ifc_expr, sig_expr)]
    end

    collect(Iterators.product(ifc_unions, sig_unions))
end

function collect_orset_to_check(ifc_expr::Number, sig_expr, errors)
    if ifc_expr != sig_expr
        push!(errors, """Invalid numeric type paramter $sig_expr:
            \tdefined :  $(sig_expr)
            \texpected:  $(ifc_expr)""")
    end
    return []
end

function bind_sig_symbols(ifc_expr::Number, sig_expr, ifc_scope, sig_scope, ifc, sig, sigarg, errors, source)
    if ifc_expr != sig_expr
        push!(errors, """Invalid numeric type paramter $sig_expr:
            \tdefined :  $(sig_expr)
            \texpected:  $(ifc_expr)""")
    end
end

function bind_sig_symbols(ifc_expr::Expr, sig_expr, ifc_scope, sig_scope, ifc, sig, sigarg, errors, source)
    base_ifc_expr, ifc_typevars = in_curly(ifc_expr)
    base_sig_expr, sig_typevars = in_curly(sig_expr)

    @debug "bind_sig_symbols(Expr, Expr)" ifc=ifc_expr sig=sig_expr base_ifc=base_ifc_expr base_sig=base_sig_expr iscope=ifc_scope sscope=sig_scope

    if base_ifc_expr === base_sig_expr
        if length(ifc_typevars) != length(sig_typevars)
            sigparms = [sig.wparams...;[as_scope_expr(x) for x in keys(last(sig_scope))]...]
            ifcparms = [ifc.wparams...;[as_scope_expr(x) for x in keys(last(ifc_scope))]...]
            if !(eval_expr(sig.mod, sig_expr, sigparms) <: eval_expr(ifc.mod, ifc_expr, ifcparms))
                push!(errors, """Invalid typevar length for paramter $sig_expr:
                    \tdefined :  $(sig_expr)
                    \texpected:  $(ifc_expr)""")
                return
            end
        end
        for (ifc_typevar, sig_typevar) in zip(ifc_typevars, sig_typevars)
            parms = [sig.wparams...;[as_scope_expr(x) for x in keys(last(sig_scope))]...]
            evald_sigtypevar = eval_expr(sig.mod, sig_typevar, parms)
            walk_arg_depth(ifc_typevar, sig_typevar, ifc_scope, sig_scope, ifc, sig, evald_sigtypevar, source)
        end
    else
        check_supertype_sig(ifc_expr, ifc_scope, sig_scope, ifc, sig, sigarg, errors, source)
    end
end

function bind_sig_symbols(ifc_symbol::Symbol, sig_expr, ifc_scope, sig_scope, ifc, sig, sigarg, errors, source)
    @debug "bind_sig_symbols(Symbol)" ifc=ifc_symbol sig=sig_expr ifc_scope=ifc_scope
    # this handles the case where the interface is `:(I where I)`
    haskey(ifc_scope[end], ifc_symbol) && return

    for (i, iscope) in reverse(collect(enumerate(ifc_scope)))
        for (k,v) in iscope
            if k.var === ifc_symbol
                # push! binds symbols (the var itself)
                push!(v.bindings, sig_expr)
                # push_dependencies_into_scope! binds the variables in the dependencies (sub/supertyping)
                push_dependencies_into_scope!(k, sig_expr, ifc_scope[i:-1:1])
                @goto jump_out_of_loop;
            end
        end
    end
    @label jump_out_of_loop

    @debug "bind_sig_symbols(Symbol)" iscope=ifc_scope sscope=sig_scope
end

function push_dependencies_into_scope!(scope::Scope, binding, ifc_scope)
    symbols = rule_symbols(scope)
    @debug "push_dependencies_into_scope!" scope=scope symbols=symbols binding=binding ifc_scope=ifc_scope
    !isempty(symbols) || return

    for scp in ifc_scope
        for (k,v) in scp
            for s in symbols
                if s === k.var
                    push!(v.dependencies, scope=>binding)
                end
            end
        end
    end

end

function check_for_mismatched_scoping(mod, ifc, sig, ifc_scope, sig_scope, errors)
    scope_wheres = wheres(ifc_scope, sig)

    scope = pop!(ifc_scope)
    pop!(sig_scope)

    @debug "check_for_mismatched_scoping" scope=scope
    
    for (k,v) in scope
        types = unique(v.bindings)
        if length(types) > 1
            push!(errors, """Invalid typing--interface typevar bound to multiple types
                \ttypevar:  $(k.var)
                \ttypes  :  $types""")
        end
    end

    # TODO must also check for invalid subtyping based on scoping
    # e.g. var"#s10" <: T => Float64 and T => AbstractFloat
    vars = Symbol[k.var for k in keys(scope)]
    for (_,v) in scope
        if isempty(v.dependencies)
            continue
        end

        for dep in v.dependencies
            scp::Scope = first(dep)
            rule_symbs = rule_symbols(scp)
            rule_binding = last(dep)

            if !all(in(vars), rule_symbs)
                @warn """Cannot currently check depencency $dep in current scope;
                        only $vars are available.  In the future, we must collect
                        bindings and push them into the parent scope."""
                continue
            end

            d = Dict()
            for v in rule_symbs
                try
                    vals = get(scope, v, nothing).bindings
                    if !isempty(vals)
                        push!(d, v=>vals[1])
                    else
                        # TODO push error and continue
                    end
                catch e
                    @debug "$e" scope=scope v=v d=d
                    rethrow()
                end
            end

            new_rule = copy(scp.rule)
            replace_rule_symbol!(new_rule, scp.var, rule_binding)
            for (k, rep) in d 
                replace_rule_symbol!(new_rule, k, rep)
            end

            error_rule = copy(new_rule)
            if new_rule.head === :(<:) || new_rule.head === :(>:)
                new_rule.args[1] = wrap_in_where(new_rule.args[1], scope_wheres)
                new_rule.args[2] = wrap_in_where(new_rule.args[2], scope_wheres)
            end

            @debug "check_for_mismatched_scoping" new_rule=new_rule
            # TODO try/catch around this; cannot judge
            try
                if !mod.eval(new_rule)
                    push!(errors, """Invalid typing--interface typevars misbound
                    \trule        :  $(scp.rule)
                    \tbound rule  :  $(error_rule)""")
                end
            catch e
                @debug """Cannot currently check unrealized dependency scopes
                \t$(new_rule)
                """
                # occurs when we are checking against type parameters 
                # when we do not have a where for them;
                # FIXME add where clause to dependencies
            end
        end
    end
end

function walk_arg_depth(ifc_expr, sig_expr, ifc_scope, sig_scope, ifc, sig, sig_evald_arg, source)
    bare_ifc_expr, sub_ifc_scope = find_scope(ifc_expr)
    bare_sig_expr, sub_sig_scope = find_scope(sig_expr)

    last_ifc_scope = [as_scope_expr(x) for x in keys(last(ifc_scope))]
    last_sig_scope = [as_scope_expr(x) for x in keys(last(sig_scope))]
    @debug "walk_arg_depth" ifc=ifc_expr sig=sig_expr iscope=ifc_scope sscope=sig_scope

    push!(ifc_scope, create_scope_dicts(sub_ifc_scope))
    push!(sig_scope, create_scope_dicts(sub_sig_scope))

    allerrors = String[]
    orset = collect_orset_to_check(bare_ifc_expr, bare_sig_expr, allerrors)
    @debug "walk_arg_depth" orset=orset sis=sub_ifc_scope sss=sub_sig_scope
    for (i_expr, s_expr) in orset
        errors = String[]
        i_scope = copy(ifc_scope)
        s_scope = copy(sig_scope)

        evald, s_eval, i_expr, s_expr = eval_in_union(sig.mod, i_expr, s_expr, 
                            [ifc.wparams...;last_ifc_scope...;sub_ifc_scope...], 
                            [sig.wparams...;last_sig_scope...;sub_sig_scope...], 
                            allerrors)
        evald || continue

        bind_sig_symbols(i_expr, s_expr, i_scope, s_scope, ifc, sig, s_eval, errors, source)
        check_for_mismatched_scoping(sig.mod, ifc, sig, i_scope, s_scope, errors)
        isempty(errors) || push!(allerrors, join(unique(errors), "\n"))
    end

    pop!(ifc_scope)
    pop!(sig_scope)

    @debug "walk_arg_depth" lenerrors=length(allerrors)
    (length(allerrors) < length(orset)) || isempty(allerrors) || error(string(source, "\n", join(allerrors, "\n")))
end

function check_signature_type_params_against_interface(ifc, sig, source)
    # if the interface has no type parameters, then the previous subtyping checks cover everything
    isempty(ifc.wparams) && return nothing
    ifc_scope = Any[create_scope_dicts(ifc.wparams)]
    sig_scope = Any[create_scope_dicts(sig.wparams)]
    
    @debug "check_signature_type_params_against_interface" allsig=sig.args allifc=ifc.args sigrv=sig.retval ifcrv=ifc.retval

    for (n, (sig_expr, ifc_expr)) in enumerate(zip([sig.args;sig.retval], [ifc.args;ifc.retval]))
        # check both expressions against each other, and strip only when sig_params does not match ifcparams?
        ifc_expr = stripwheres(ifc_expr, ifc.wparams)
        sig_expr = stripwheres(sig_expr, sig.wparams)

        @debug "check_signature_type_params_against_interface" ifcexpr=ifc_expr sigexpr=sig_expr 

        walk_arg_depth(ifc_expr, sig_expr, ifc_scope, sig_scope, ifc, sig, ntharg(sig, n), source)
    end

    @debug "check_signature_type_params_against_interface" ifc_scope=ifc_scope
    errors = []
    check_for_mismatched_scoping(sig.mod, ifc, sig, ifc_scope, sig_scope, errors)
    isempty(errors) || error(string(source, "\n", join(errors, "\n")))
end

function __validate_signature(fkey, fname, sig::Signature, source)
    ifc = get_interface(fkey)
    ifc !== nothing || error("Interface not defined for $fname (impl at $source)")

    length(sig.args_evald) == length(ifc.args_evald) || error("""Invalid argument length for $fname at $source:
    \tdefined :  $(sig.args_evald)
    \texpected:  $(ifc.args_evald)""")
    
    subtype = map(<:, sig.args_evald, ifc.args_evald)
    all(subtype) || error("""Invalid definition of $fname at $source:
    \tdefined :  $(sig.args_evald)
    \texpected:  $(ifc.args_evald)
    \tdefined :  $(sig.args)
    \texpected:  $(ifc.args)
    \tat      :  $subtype""")

    (sig.retval_evald <: ifc.retval_evald) || error("""Invalid return type for $fname at $source:
    \tdefined :  $(sig.retval_evald)
    \texpected:  $(ifc.retval_evald)
    \tdefined :  $(sig.retval)
    \texpected:  $(ifc.retval)""")

    @debug "__validate_signature" f=fname ifc=ifc sig=sig

    check_signature_type_params_against_interface(ifc, sig, source)
end

function __validate_signature(mod, source, fkey::Symbol, f, args::Vector{<:Vector}, wparams)
    body(t::UnionAll) = body(t.body)
    body(t::Type) = t

    for arglist in args
        @debug "__validate_signature" f=string(f) args=arglist wparams=wparams
        sig = Signature(mod, arglist[2:end], arglist[1], wparams)
        try
            __validate_signature(fkey, Symbol(f), sig, source)
        catch e
            if !(e isa ErrorException)
                @error("Error in function $f at $source")
            end
            rethrow()
        end
    end
end

