# 0. Introduce the course briefly, ask about programming capabilities and academic background
# 1. Show how to install Julia and use juliaup
# 2. Show how to use Julia REPL and set up environment with packages.

x = 3
arr = [1, 2, 3]
arr[1]
x *= 3  # x = 9

for k = 1:4
    @show k
end

my_square(x) = x^2
my_square(3)
my_square.(arr)

# 3. Show how to use JuMP and Convex to solve a least-squares problem.

# y_true = a*x_true + b
# we get noisy measurements y_meas = y_true + noise

using JuMP
using HiGHS

x_true = -10:0.4:10
a, b = 1.3, 2.0
y_true = a * collect(x_true) .+ b
y_meas = y_true + 0.6 * randn(size(y_true))

using Plots
scatter(x_true, y_meas)

# We want to find the best line that fits the data
model = Model()
@variable(model, â)
@variable(model, b̂)

@objective(
    model,
    Min,
    sum((y_meas - (â * x_true .+ b̂)).^2)
)

set_optimizer(model, HiGHS.Optimizer)

# show the ! on reverse/reverse!
optimize!(model)

plot!(x_true, value(â) * x_true .+ value(b̂), linewidth=2)

a_b_linalg = [x_true ones(length(x_true))] \ y_meas
plot!(x_true, a_b_linalg[1] * x_true .+ a_b_linalg[2], linewidth=2)
a_b_linalg ≈ [value(â), value(b̂)]

## We know how to do this with linear algebra, but how about constraints?

@constraint(model, â ≤ 0)
optimize!(model)
plot!(x_true, value(â) * x_true .+ value(b̂), linewidth=2)

## How about a different criterion?
x2_true = [-10.1; collect(x_true); 10.1]
y2_meas = [-130; y_meas; 130]

scatter(x2_true, y2_meas)

model = Model()
@variable(model, â)
@variable(model, b̂)
@objective(
    model,
    Min,
    sum((y2_meas - (â * x2_true .+ b̂)).^2 )
)
set_optimizer(model, HiGHS.Optimizer)
optimize!(model)
plot!(x2_true, value(â) * x2_true .+ value(b̂), linewidth=2)

residuals = y2_meas - (â * x2_true .+ b̂)
@variable(model, abs_residuals[1:length(residuals)] ≥ 0)
@constraint(model, abs_residuals .≥ residuals)
@constraint(model, abs_residuals .≥ -residuals)

@objective(model, Min, sum(abs_residuals))

optimize!(model)

plot!(x2_true, value(â) * x2_true .+ value(b̂), linewidth=2, color=:yellow)

## Convex.jl
using Convex

â = Variable()
b̂ = Variable()

# Array of constraints to be filled incrementally
constraints = []
push!(constraints, â ≤ 15)

# Various convex functions available compared to JuMP
# https://jump.dev/Convex.jl/stable/reference/atoms/#huber
# Convex.jl keeps track of 'vexity' of the functions (https://dcp.stanford.edu/rules)
residuals = y2_meas - (â * x2_true + b̂)
objective = huber(residuals, 10) |> sum
problem = minimize(objective, constraints)
solve!(problem, HiGHS.Optimizer)    # throws error
using SCS
solve!(problem, SCS.Optimizer)
plot!(x2_true, evaluate(â) * x2_true .+ evaluate(b̂), linewidth=2, )