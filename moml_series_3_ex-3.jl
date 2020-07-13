################################################################################
# DEFINE the proximal map for \lambda  || x \|_{1}
################################################################################
function prox_ell_one(x::Vector{Float64},lambda::Float64)
    prox = zeros(length(x))

    for i in 1:length(x)
        if x[i] >= lambda
            prox[i] = x[i] - lambda
        elseif x[i] <= -lambda
            prox[i] = x[i] + lambda
        else
            prox[i] = 0.0
        end
    end

    return prox
end
################################################################################

################################################################################
# One possibility for the Proximal type
# P_X = Proximal(prox_ell_one)
# P_X.Prox(randn(10),0.1)

# This is a partial function application of the prox_ell_one
function pa_prox_ell_one(lambda::Float64)
    pa_prox_l1(x::Vector{Float64}) = prox_ell_one(x,lambda)

    return pa_prox_l1
end

# Another possibility for the Proximal type. Here, we fix lambda
# so that P_X.Prox only takes the argument x.
# P_X = Proximal(pa_prox_ell_one(0.1))
# P_X.Prox(randn(10))
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


function quadratic_objective(A::Matrix{Float64},
                             b::Vector{Float64},
                             x::Vector{Float64})

    z = A*x - b
    f = ell_lp_norm(z,2)
    return 0.5*f^2
end


function grad_f_ell_2(A::Matrix{Float64},b::Vector{Float64},x::Vector{Float64})
    temp = A*x
    temp = A'*temp
    return temp - A'*b
end
################################################################################

################################################################################
# One possibility for the Objective type
# f = Objective(quadratic_objective,grad_f_ell_2)
# f.Obj(rand(2,2),rand(2),rand(2),2)

# This is a partial function application of quadratic_objective
function pa_quadratic_objective(A::Matrix{Float64},b::Vector{Float64})

    pa_quad_obj(x::Vector{Float64}) = quadratic_objective(A,b,x)

    return pa_quad_obj
end

# This is a partial function application of grad_f_ell_2
function pa_grad_f_ell_2(A::Matrix{Float64},b::Vector{Float64})

    pa_d_quad_obj(x::Vector{Float64}) = grad_f_ell_2(A,b,x)

    return pa_d_quad_obj
end

# Another possibility for the Objective type
# A = rand(2,2)
# b = rand(2)
# p = 2
# f = Objective(pa_quadratic_objective(A,b,p),pa_grad_f_ell_2(A,b))
# f.Obj(rand(2))
# f.dObj(rand(2))
################################################################################

################################################################################
# DEFINE a stepsize rule
################################################################################
# Since f is smooth and convex with a Lipschitz gradient, we can pick a fixed
# step size, if we can determine or estimate the Lipschitz constant at least
# over the feasible set X
#
#
# Here, grad f(x) = A^T A x - A^T b. On X we have the bound
# grad f(x) - grad f(y) = A^T A (x - y). Taking the Euclidean norm of both sides
# and bounding from above using the Cauchy-Schwarz inequality we have
# | grad f(x) - grad f(y)| \le ||A^T A|| ||x - y||
#
# Since we only need it once, we compute the Lipschitz constant using
# the built-in Julia function for the induced matrix norm: opnorm(A'*A, 2)
# You need the LinearAlgebra package for this.
################################################################################
function gamma_t(A::Matrix{Float64})
    return opnorm(A'*A, 2)
end
