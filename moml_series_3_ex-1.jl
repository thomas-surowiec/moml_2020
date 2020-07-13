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


function quadratic_objective(A::Matrix{Float64},
                             b::Vector{Float64},
                             x::Vector{Float64},
                             p::Int64)
    z = A*x - b
    f = ell_lp_norm(z,p)
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
function pa_quadratic_objective(A::Matrix{Float64},b::Vector{Float64},p::Int64)

    pa_quad_obj(x::Vector{Float64}) = quadratic_objective(A,b,x,p)

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
# || grad f(x) - grad f(y)|| \le ||A^T A||_{op,2} ||x - y||_{2}
#
# Since we only need it once, we compute the Lipschitz constant using
# the built-in Julia function for the induced matrix norm: opnorm(A'*A, 2)
# You need the LinearAlgebra package for this.
################################################################################
function gamma_t(A::Matrix{Float64})
    return opnorm(A'*A, 2)
end
# g_t = StepSize(A -> gamma_t(A),1/opnorm(A'*A, 2))
# g_t.StepRule(A)
# g_t.FixedStep
