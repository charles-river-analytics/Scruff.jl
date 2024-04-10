# MultipleScore lets you assert multiple evidence on the same variable
export MultipleScore

struct MultipleScore{I} <: Score{I}
    components :: Vector{<:Score{I}}
end

@impl begin
  function get_log_score(ms::MultipleScore{I}, i::I) where I
      tot = 0.0
      for m in ms.components
          tot += get_log_score(m, i)
      end
      tot
  end
end
