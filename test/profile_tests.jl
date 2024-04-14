using Profile

include("runtests.jl")
@profile include("runtests.jl")

open("output/profile_tree.txt", "w") do s
    Profile.print(IOContext(s, :displaysize => (24, 500));
                  format=:tree,
                  maxdepth=60,
                  noisefloor=4,
                  mincount=4)
end

open("output/profile_flat.txt", "w") do s
    Profile.print(IOContext(s, :displaysize => (24, 500));
                  format=:flat,
                  sortedby=:count,
                  maxdepth=60,
                  noisefloor=4,
                  mincount=4)
end
