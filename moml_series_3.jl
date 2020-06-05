#
# This is a version of homework problem 3.i and 3.iii
# The code is written in a way that it most closely mirrors
# the discussion in the text.
#

################################################################################
# DEFINE structs: We define three mutable structures that comprise
# the main components of the projected subgradient algorithms
# These are mainly introduced for readability
################################################################################
mutable struct Projection
    Proj::Function
end#

mutable struct Objective
     Obj::Function
    dObj::Function
end#

mutable struct StepSize
    StepRule::Function
end
################################################################################

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
A = rand(2,2)
b = rand(2)
p = 2
f = Objective(pa_quadratic_objective(A,b,p),pa_grad_f_ell_2(A,b))
f.Obj(rand(2))
f.dObj(rand(2))
################################################################################
