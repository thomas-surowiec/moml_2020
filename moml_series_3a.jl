#
# This is a version of homework problem 3
# The code is written in a way that it most closely mirrors
# the discussion in the text.
#
using LinearAlgebra
using Plots

include("ExampleOpt-1a.jl")

################################################################################
# DEFINE structs: We define three mutable structures that comprise
# the main components of the projected subgradient algorithms
# These are mainly introduced for readability
################################################################################
mutable struct Proximal
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
    f_vec    = zeros(maxit)
    f_vec[1] = f.Obj(x_0)

    if step == "fixed"
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
            f_vec[it] = f.Obj(x_0)
        end
    else
        while it < maxit
            x_1 = P_X.Proj(x_0 - g_t.StepRule(maxit)*f.dObj(x_0))

            # Behavior of iterates
            # Progress:
            # println("||x_0 - x_1|| = ", norm(x_0-x_1))
            # Optimality:
            # println("x - Proj_X(x - grad f(x)) = ", norm(x_1 - P_X.Proj(x_1 - f.dObj(x_1))))
            # Objective Function:
            # println("f(x_1) - f(x_0) = ", f.Obj(x_1) - f.Obj(x_0))
            x_0 = x_1
            it += 1
            f_vec[it] = f.Obj(x_0)
        end
    end

    return x_0, f_vec
end
################################################################################
