################################################################################
# DEFINE a projection onto multidimensional bilateral constraints
################################################################################
function projection_ab(x::Vector{Float64},a::Vector{Float64},b::Vector{Float64})
    proj = zeros(length(x))

    for i in 1:length(x)
        if x[i] >= b[i]
            proj[i] = b[i]
        elseif x[i] <= a[i]
            proj[i] = a[i]
        else
            proj[i] = x[i]
        end
    end

    return proj
end
################################################################################

################################################################################
# One possibility for the Projection type
# P_X = Projection(projection_ab)
# P_X.Proj(randn(10),-1*ones(10),ones(10))

# This is a partial function application of the projection_ab
function pa_projection_ab(a::Vector{Float64},b::Vector{Float64})
    pa_proj_ab(x::Vector{Float64}) = projection_ab(x,a,b)

    return pa_proj_ab
end

# Another possibility for the Projection type. Here, we fix the bounds
# so that P_X.Proj only takes the argument x.
P_X = Projection(pa_projection_ab(-1*ones(10),ones(10)))
P_X.Proj(randn(10))
################################################################################
