#
# This is a version of homework problem 3.i and 3.iii
# The code is written in a way that it most closely mirrors
# the discussion in the text.
#
using LinearAlgebra
include("ExampleOpt-1.jl")

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
    FixedStep::Float64
end
################################################################################




################################################################################
# main function: A projected subgradient algorithm
################################################################################
function proj_sub_grad(f::Objective,
                       Proj::Projection,
                       g_t::StepSize,
                       maxit::Int64,
                       x_0::Vector{Float64},
                       step::String)

    it = 0
    if step == "fixed"
        while it < maxit
            x_1 = P_X.Proj(x_0 - g_t.FixedStep*f.dObj(x_0))

            # Behavior of iterates
            # Progress:
            # println("||x_0 - x_1|| = ", norm(x_0-x_1))
            # Optimality:
            # println("x - Proj_X(x - grad f(x)) = ", norm(x_1 - P_X.Proj(x_1 - f.dObj(x_1))))
            # Objective Function:
            println("f(x_1) - f(x_0) = ", f.Obj(x_1) - f.Obj(x_0))

            x_0 = x_1
            it += 1
        end
    else
        while it < maxit
            x_1 = P_X.Proj(x_0 - g_t.FixedStep*f.dObj(x_0))

            # Behavior of iterates
            # Progress:
            # println("||x_0 - x_1|| = ", norm(x_0-x_1))
            # Optimality:
            # println("x - Proj_X(x - grad f(x)) = ", norm(x_1 - P_X.Proj(x_1 - f.dObj(x_1))))
            # Objective Function:
            # println("f(x_1) - f(x_0) = ", f.Obj(x_1) - f.Obj(x_0))

            x_0 = x_1
            it += 1
        end
    end

    return x_0
end

# A problem instance for ExampleOpt-1.jl
A = randn(50,50)
b = rand(50)
c = -ones(50)
d = ones(50)

P_X = Projection(pa_projection_ab(c,d))
f   = Objective(pa_quadratic_objective(A,b,2),pa_grad_f_ell_2(A,b))
g_t = StepSize(A -> gamma_t(A),1/opnorm(A'*A, 2))
x_0 = rand(50)

x_sol = proj_sub_grad(f,P_X,g_t,100,x_0,"fixed")
