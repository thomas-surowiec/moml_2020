cd(dirname(@__FILE__()))
using LinearAlgebra
using Plots

include("moml_series_4_ex-3.jl")
# include("moml_series_4_ex-4.jl")

################################################################################
# Constructors for special types
################################################################################
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
# main function: A proximal subgradient algorithm (entropic dgf)
################################################################################
function entrop_sub_grad(G::Oracle,
                         g_t::StepSize,
                         maxit::Int64,
                         x_0::Vector{Float64})

    it = 0
    err_vec = zeros(maxit)
    x_1 = zeros(length(x_0))
    while it < maxit
        norm_const = 0.0
        for k in 1:length(x_0)
            norm_const += x_0[k]*exp(-g_t.step_rule(it+1)*G(x_0)[k])
        end

        for i in 1:length(x_0)
            x_1[i] = x_0[i]*exp(-g_t.step_rule(it+1)*G(x_0)[i])/norm_const
        end
        # println(norm(x_1))

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
x_sv  = ones(dim_x)/dim_x # feasible x

# Construct oracle
G_t = Oracle(stoch_grad,[0.1])
G_t(rand(dim_x))

# Construct prox-operator
g_t = StepSize(0.1,pa_gamma_t(1.0))

# Run example
x_0, err_vec = entrop_sub_grad(G_t,g_t,500,x_sv)

# Plot behavior of consecutive iterates
ep  = plot(err_vec,lw = 2,label=false, yaxis = :log)

# Save figure
# savefig(ep,"diff_of_iterates_smd_ex-1.pdf")
# savefig(ep,"diff_of_iterates_smd_ex-2.pdf")
# savefig(ep,"diff_of_iterates_smd_ex-3.pdf")
# savefig(ep,"diff_of_iterates_smd_ex-4.pdf")
