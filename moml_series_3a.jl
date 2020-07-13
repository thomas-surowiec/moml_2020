#
# This is a version of homework problem 3
# The code is written in a way that it most closely mirrors
# the discussion in the text.
#
using LinearAlgebra
using Plots

include("moml_series_3_ex-3.jl")

################################################################################
# DEFINE structs: We define three mutable structures that comprise
# the main components of the projected subgradient algorithms
# These are mainly introduced for readability
################################################################################
mutable struct Proximal
    Prox::Function
end

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
function prox_sub_grad(f::Objective,
                       P_X::Proximal,
                       g_t::StepSize,
                       maxit::Int64,
                       x_0::Vector{Float64},
                       step::String)

    it = 0
    f_vec    = zeros(maxit)
    f_vec[1] = f.Obj(x_0)

    if step == "fixed"
        while it < maxit
            x_1 = P_X.Prox(x_0 - g_t.FixedStep*f.dObj(x_0))

            # Behavior of iterates
            # Progress:
            # println("||x_0 - x_1|| = ", norm(x_0-x_1))
            # Objective Function:
            # println("f(x_1) - f(x_0) = ", f.Obj(x_1) - f.Obj(x_0))

            x_0 = x_1
            it += 1
            f_vec[it] = f.Obj(x_0)
        end
    else
        while it < maxit
            x_1 = P_X.Prox(x_0 - g_t.StepRule(maxit)*f.dObj(x_0))

            # Behavior of iterates
            # Progress:
            # println("||x_0 - x_1|| = ", norm(x_0-x_1))
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

################################################################################
# A problem instance for ExampleOpt-1.jl
A      = randn(5,5)
b      = rand(5)
lambda = 1/opnorm(A'*A, 2) # According to the theory, we need to pick lambda = 1/L

P_X = Proximal(pa_prox_ell_one(lambda))
f   = Objective(pa_quadratic_objective(A,b),pa_grad_f_ell_2(A,b))
g_t = StepSize(A -> gamma_t(A),1/opnorm(A'*A, 2))
x_0 = rand(5)

@time x_sol, f_vec = prox_sub_grad(f,P_X,g_t,10000,x_0,"fixed")

# log-log plot
plot(f_vec, xaxis=:log, yaxis=:log)

# plot
plot(f_vec)
################################################################################
