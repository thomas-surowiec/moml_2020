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
grad_f_ell_2(rand(2,2),rand(2),rand(2))

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
subdiff_ell_1(randn(10))

function ell_l1_norm_nsq(A::Matrix{Float64},
                         b::Vector{Float64},
                         x::Vector{Float64})
    z = A*x - b
    q = subdiff_ell_1(z)

    return A'*q
end
ell_l1_norm_nsq(rand(20,20),rand(20),randn(20))

function ell_l1_norm_sq(A::Matrix{Float64},
                        b::Vector{Float64},
                        x::Vector{Float64})
    z = A*x - b
    q = subdiff_ell_1(z)

    return ell_lp_norm(z,1)*A'*q
end
ell_l1_norm_sq(rand(20,20),rand(20),randn(20))


function projection_Rn_plus(x::Vector{Float64})
    proj = zeros(length(x))

    for i in 1:length(x)
        if x[i] >= 0
            proj[i] = x[i]
        else
            proj[i] = 0.0
        end
    end

    return proj
end
projection_Rn_plus(randn(10))

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
a = -1*ones(10)
b = ones(10)
projection_ab(randn(10),a,b)

function projection_hp(x::Vector{Float64},w::Vector{Float64},b::Float64)
    c_1 = w'*x + b
    c_2 = ell_lp_norm(w,2)
    return x - (c_1/c_2)*w
end
w  = rand(3)
bb = 1.0
xx = randn(3)
projection_hp_1(xx,w,bb)
