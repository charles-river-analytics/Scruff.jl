export
    @op_perf,
    @make_c,
    runtime

# All TODO

macro op_perf(op_perf_body)
    return esc(op_perf_macro(op_perf_body))
end

function op_perf_macro(op_perf_body::Expr)
    return quote 
        $op_perf_body
        export $(op_perf_body.args[1].args[1])
    end
end

macro make_c()
    return (:1)
end

function runtime end
