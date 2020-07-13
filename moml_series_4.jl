#=
moml_series_4

A stochastic mirror descent algorithm

In its most general form, we assume to have a:
- prox-function
- stochastic oracle
- step size rule
- prox-operator
- start value x_0
- stopping criterion
in order to generate a sequence x_k via stochastic mirror descent

This would require, e.g., several types:
- mutable struct Objective
- mutable struct FeasibleSet
- mutable struct StochasticOracle
- mutable struct Proximal
- mutable struct StepSize

This might not be able to make use of known explicit formulae and be less
efficient. For specific examples, we can define less types and have an even
simpler algorithm. This requires a bit of pen and paper work first.
=#
cd(dirname(@__FILE__()))
using LinearAlgebra
using Plots

include("moml_series_4_ex-1.jl")

mutable struct Proximal
    prox::Function
end

mutable struct Oracle
    stoch_grad::Function
    func_data::Array{Float64}
end

function (G_t::Oracle)(x::Vector{Float64})
    return G_t.stoch_grad(Diagonal(randn(n)),randn(n),G_t.func_data[1],x)
end

mutable struct StepSize
    fixd_step::Float64
    step_rule::Function
end

################################################################################
# main function: A projected subgradient algorithm
################################################################################
function prox_sub_grad(G::Oracle,
                       P_X::Proximal,
                       g_t::StepSize,
                       maxit::Int64,
                       x_0::Vector{Float64})

    it = 0
    err_vec = zeros(maxit)
    while it < maxit
        x_1 = P_X.prox(x_0 - g_t.step_rule(it)*G(x_0))

        err_vec[it+1] = norm(x_0 - x_1)
        println(norm(x_0 - x_1))

        x_0 = x_1
        it += 1
    end

    return x_0, err_vec
end
################################################################################



################################################################################
# TEST
################################################################################
x_0, err_vec = prox_sub_grad(G_t,Proj,g_t,10000,zeros(dim_x))

ep  = plot(err_vec,lw = 2,label=false, yaxis = :log)
savefig(ep,"diff_of_iterates_smd.pdf")
