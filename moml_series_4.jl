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

This would require several types, e.g.:
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

# include("moml_series_4_ex-1.jl")
include("moml_series_4_ex-2.jl")

################################################################################
# Constructors for special types
################################################################################
# Proximal type for prox-operator
mutable struct Proximal
    prox::Function
end

# Stochastic Oracle: A callable type
mutable struct Oracle
    stoch_grad::Function
    func_data::Array{Float64}
end

function (G_t::Oracle)(x::Vector{Float64})
    n = length(x)
    return G_t.stoch_grad(Diagonal(randn(n)),randn(n),G_t.func_data[1],x)
end

# Step Size Rule: Allows fixed or adaptable steps
mutable struct StepSize
    fixd_step::Float64
    step_rule::Function
end
################################################################################

################################################################################
# main function: A proximal subgradient algorithm
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

        x_0 = copy(x_1)
        it += 1
    end

    return x_0, err_vec
end
################################################################################

################################################################################
# TEST
################################################################################
# Select size of problem
dim_x = 100

# Construct oracle
G_t = Oracle(stoch_grad,[1.0])
G_t(rand(dim_x))

# Construct prox-operator
c    = -ones(dim_x)
d    = ones(dim_x)
Proj = Proximal(pa_projection_ab(c,d))

# Construct prox-operator
g_t = StepSize(0.1,pa_gamma_t(10.0))

# Run example
x_0, err_vec = prox_sub_grad(G_t,Proj,g_t,10000,zeros(dim_x))

# Plot behavior of consecutive iterates
ep  = plot(err_vec,lw = 2,label=false, yaxis = :log)

# Save figure
# savefig(ep,"diff_of_iterates_smd_ex-1.pdf")
savefig(ep,"diff_of_iterates_smd_ex-2.pdf")
