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

using LinearAlgebra

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
    f_vec    = zeros(maxit)
    f_vec[1] = f.Obj(x_0)

    while it < maxit
        x_1 = P_X.Prox(x_0 - g_t.StepRule(maxit)*f.dObj(x_0))
        x_0 = x_1
        it += 1
        f_vec[it] = f.Obj(x_0)
    end

    return x_0, f_vec
end
################################################################################
function stoch_grad(A::Diagonal{Float64,Array{Float64,1}},
                    b::Vector{Float64},
                    alpha::Float64,
                    x::Vector{Float64})
    g = A*x - b
    g = A'*g + alpha*x
    return g
end

G_t = Oracle(stoch_grad,[1.0])
G_t(rand(100))
