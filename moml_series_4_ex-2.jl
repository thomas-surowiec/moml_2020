
################################################################################
# Example
################################################################################
function subdiff_ell_1(x::Vector{Float64})
    q = zeros(length(x))
    for i in 1:length(x)
        if x[i] > 0
            q[i] = 1.0
        elseif x[i] < 0
            q[i] = -1.0
        else
            q[i] = 2*rand()-1
        end
    end

    return q
end

function stoch_grad(A::Diagonal{Float64,Array{Float64,1}},
                    b::Vector{Float64},
                    alpha::Float64,
                    x::Vector{Float64})

    z = A*x - b
    q = subdiff_ell_1(z)

    return A'*q + alpha*x
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
