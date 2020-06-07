#
# This is a version of homework problem 3
# The code is written in a way that it most closely mirrors
# the discussion in the text.
#
using LinearAlgebra
using Plots
# include("ExampleOpt-1.jl")
include("ExampleOpt-2.jl")

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

################################################################################
# A problem instance for ExampleOpt-1.jl
A = randn(50,50)
b = rand(50)
c = -ones(50)
d = ones(50)

P_X = Projection(pa_projection_ab(c,d))
f   = Objective(pa_quadratic_objective(A,b,2),pa_grad_f_ell_2(A,b))
g_t = StepSize(A -> gamma_t(A),1/opnorm(A'*A, 2))
x_0 = rand(50)

x_sol, f_vec = proj_sub_grad(f,P_X,g_t,100,x_0,"fixed")

# log-log plot
plot(f_vec, xaxis=:log, yaxis=:log)

# plot
plot(f_vec)
################################################################################

################################################################################
# A problem instance for ExampleOpt-2.jl
A = randn(50,50)
b = rand(50) # test accuracy with b = zeros(50)
c = -ones(50)
d = ones(50)

P_X = Projection(pa_projection_ab(c,d))
f   = Objective(pa_ns_objective(A,b),pa_subgrad_ns_obj(A,b))
g_t = StepSize(pa_gamma_t(A,d),0.1)
x_0 = rand(50)

x_sol, f_vec = proj_sub_grad(f,P_X,g_t,1000,x_0,"var")

# log-log plot
plot(f_vec, xaxis=:log, yaxis=:log, leg=false)

# plot
plot(f_vec, leg=false)
################################################################################

################################################################################
# This is a work in progress intended to illustrate the theoretical
# behavior more clearly 
#
# function aggregate(f::Objective,Proj::Projection,g_t::StepSize,k::Int64,
#                    x_0::Vector{Float64},step::String)
#
#     scaling = 0
#     gam_t   = g_t.StepRule(k)
#
#     for t in 1:k
#         scaling += gam_t
#     end
#
#     # println(gam_t)
#     x_o   = x_0
#     x_bar = zeros(length(x_o))
#
#     for t in 1:k
#         x_n, f_vec = proj_sub_grad(f,P_X,g_t,1,x_o,"var")
#         x_o = x_n
#
#         # p = plot(x_o)
#         # display(p)
#
#         x_bar += gam_t*x_n
#     end
#     # p = plot(x_bar/scaling)
#     # display(p)
#
#     return x_bar/scaling
# end
#
# x_sol_bar, f_vec = aggregate(f,P_X,g_t,100,x_0,"var")
#
# # log-log plot
# plot(f_vec, xaxis=:log, yaxis=:log, leg=false)
#
# # plot
# plot(f_vec, leg=false)
################################################################################
