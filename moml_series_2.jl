# MOML Series 2 Problem 3
function ell_lp_norm(x::Vector{Float64},p::Int64)
    # p = 1,2
    if p == 1
        elp = 0.0
        for i in 1:length(x)
            elp = elp + abs(x[i])
        end
    elseif p == 2
        elp = 0.0
        for i in 1:length(x)
            elp = elp + x[i]^2
        end
    else
        elp = 0.0
        for i in 1:length(x)
            elp = max(elp,abs(x[i]))
        end
    end
    return elp
end
ell_lp_norm(randn(1),3)

function quadratic_objective(A::Matrix{Float64},
                             b::Vector{Float64},
                             x::Vector{Float64},
                             p::Int64)
    z = A*x - b
    f = ell_lp_norm(z,p)
    return 0.5*f^2
end
quadratic_objective(rand(2,2),rand(2),rand(2),1)

function grad_f_ell_2(A::Matrix{Float64},b::Vector{Float64},x::Vector{Float64})
    temp = A*x
    temp = A'*temp
    return temp - A'*b
end
@time grad_f_ell_2(rand(2,2),rand(2),rand(2))
