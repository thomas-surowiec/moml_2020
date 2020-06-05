mutable struct Projection
    Proj::Function
end

mutable struct Objective
     Obj::Function
    dObj::Function
end

mutable struct StepSize
    StepRule::Function
end

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

P_X = Projection(projection_ab)
P_X.Proj(randn(10),-1*ones(10),ones(10))
