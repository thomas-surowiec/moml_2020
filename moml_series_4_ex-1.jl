
################################################################################
# Example
################################################################################
function stoch_grad(A::Diagonal{Float64,Array{Float64,1}},
                    b::Vector{Float64},
                    alpha::Float64,
                    x::Vector{Float64})
    g = A*x - b
    g = A'*g + alpha*x
    return g
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

function pa_projection_ab(a::Vector{Float64},b::Vector{Float64})
    pa_proj_ab(x::Vector{Float64}) = projection_ab(x,a,b)

    return pa_proj_ab
end

function gamma_t(k::Float64,i::Int64)
    return k/i
end

function pa_gamma_t(k::Float64)
    pgt(i::Int64) = gamma_t(k,i)
    return pgt
end
################################################################################
