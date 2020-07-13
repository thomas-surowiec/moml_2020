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
# P_X = Projection(pa_projection_ab(-1*ones(10),ones(10)))
# P_X.Proj(randn(10))
################################################################################

################################################################################
# DEFINE an objective function
################################################################################
# We use the norm both in the definition of the objective as well as
# for optimality criteria in smooth nlo optimization
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

function ns_objective(A::Matrix{Float64},b::Vector{Float64},x::Vector{Float64})

    z = A*x - b
    return ell_lp_norm(z,1)
end


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

function subgrad_ns_obj(A::Matrix{Float64},b::Vector{Float64},x::Vector{Float64})
    z = A*x - b
    q = subdiff_ell_1(z)

    return A'*q
end
################################################################################

################################################################################
# One possibility for the Objective type
# f = Objective(ns_objective,subgrad_ns_obj)
# f.Obj(rand(2,2),rand(2),rand(2))
# f.dObj(rand(2,2),rand(2),rand(2))

# This is a partial function application of ns_objective
function pa_ns_objective(A::Matrix{Float64},b::Vector{Float64})

    pa_ns_obj(x::Vector{Float64}) = ns_objective(A,b,x)

    return pa_ns_obj
end

# This is a partial function application of subgrad_ns_obj
function pa_subgrad_ns_obj(A::Matrix{Float64},b::Vector{Float64})

    pa_d_ns_obj(x::Vector{Float64}) = subgrad_ns_obj(A,b,x)

    return pa_d_ns_obj
end

# Another possibility for the Objective type
# A = rand(2,2)
# b = rand(2)
# p = 2
# f = Objective(pa_ns_objective(A,b,p),pa_subgrad_ns_obj(A,b))
# f.Obj(rand(2))
# f.dObj(rand(2))
################################################################################

################################################################################
# DEFINE a stepsize rule
################################################################################
# Since f is a general nonsmooth function, we cannot expect fixed step sizes
# to generate a rapidly convergent sequence.

# We need to estimate
#   M: Lipschitz constant of f
# D_X: Diameter of the set

# For M we have
# | f(x) - f(y)| \le ||Ax - b - (Ay - b)||_{1}
#                 =  ||A(x - y)||_{1}
#                \le ||A||_{op,1}||x - y||_{1}
#                \le \|A||_{op,1} \sqrt{n} ||x - y||_{2}
#
# This yields M = opnorm(A, 1) * sqrt(size(A,2)).

# For D_X we have for any x,y \in X that
# ||x - y||^2_2 =    ||x||^2_2 + 2 x^T y + ||y||^2_2
#             \le  2||x||^2_2 + 2 ||y||^2_2
#             \le  4 n ||d||^2_{\inf} <<<<<<<<<<<<< ERROR! Needs to include c
#
# This yields an upper bound for D_X
# D_X^2 :=  2 sqrt(size(A,2)) norm(d, Inf)^2

# The optimal step size is given by
# gamma_t = sqrt(2 * D_X^2/(k M^2)) where t = 1,...,k
################################################################################
function gamma_t(A::Matrix{Float64},upper_bound::Vector{Float64},k::Int64)

        M  = opnorm(A, 1)*sqrt(size(A,2))
     D_X_2 = 2*sqrt(size(A,2))*norm(upper_bound, Inf)^2

    return sqrt((2*D_X_2)/(k * M^2))
end

function pa_gamma_t(A::Matrix{Float64},upper_bound::Vector{Float64})

    pa_g_t(k::Int64) = gamma_t(A,upper_bound,k)

    return pa_g_t
end
# g_t = StepSize(gamma_t,0.1)
# g_t.StepRule(rand(50,50),rand(50),10)
# g_t.FixedStep
