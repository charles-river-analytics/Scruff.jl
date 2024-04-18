using Profile

include("runtests.jl")
@profile include("runtests.jl")

open("output/profile_tree.txt", "w") do s
    Profile.print(IOContext(s, :displaysize => (24, 500));
                  format=:tree,
                  maxdepth=100,
                  noisefloor=2,
                  mincount=2)
end

open("output/profile_flat.txt", "w") do s
    Profile.print(IOContext(s, :displaysize => (24, 500));
                  format=:flat,
                  sortedby=:count,
                  noisefloor=1,
                  mincount=1)
end
