using OSQP
using SparseArrays

# Define the problem data and build the problem description
P = sparse([4. 1.; 1. 2.])
q = [1.; 1.]
A = sparse([1. 1.; 1. 0.; 0. 1.])
l = [1.; 0.; 0.]
u = [1.; 0.7; 0.7]

problem_OSQP = OSQP.Model()
OSQP.setup!(problem_OSQP; P=P, q=q, A=A, l=l, u=u, alpha=1, verbose=false)

# Solve the problem
results_OSQP = OSQP.solve!(problem_OSQP)
results_OSQP.x


using COSMO
using SparseArrays

# Define the problem data and build the problem description
P = sparse([4. 1.; 1. 2.])
q = [1.; 1.]
A = sparse([1. 1.; 1. 0.; 0. 1.])
l = [1.; 0.; 0.]
u = [1.; 0.7; 0.7]

Aa = [-A; A]
ba = [u; -l]

problem_COSMO = COSMO.Model()
constraint = COSMO.Constraint(Aa, ba, COSMO.Nonnegatives)
settings = COSMO.Settings(verbose=false)
assemble!(problem_COSMO, P, q, constraint, settings = settings)

# Solve the problem
results_COSMO = COSMO.optimize!(problem_COSMO)
results_COSMO.x


using JuMP
using SparseArrays
using OSQP, COSMO

# Define the problem data and build the problem description

model_JuMP = Model(OSQP.Optimizer) # or COSMO.Optimizer
@variable(model_JuMP, x[1:2])
@objective(model_JuMP, Min, 0.5*x'*[4. 1.; 1. 2.]*x + [1.; 1.]'*x)
@constraint(model_JuMP, [1. 1.; 1. 0.; 0. 1.]*x .<= [1.; 0.7; 0.7])
@constraint(model_JuMP, [1. 1.; 1. 0.; 0. 1.]*x .>= [1.; 0.; 0.])
print(model_JuMP)

# Solve the problem
optimize!(model_JuMP)
termination_status(model_JuMP)
value.(x)


## SOCP

using Plots

# Define the function
f(x, y) = sqrt(x^2 + y^2)

# Generate x, y coordinates
x = -10:0.5:10
y = -10:0.5:10

# Generate z values for each (x, y) pair
z = [f(i, j) for i in x, j in y]

# Create a 3D surface plot
p = surface(x, y, z, xlabel="x", ylabel="y", zlabel="", title="")


using CairoMakie

# Define the function
f(x, y) = sqrt(x^2 + y^2)

# Generate x, y coordinates in a circular pattern
θ = LinRange(0, 2π, 100)
r = LinRange(0, 10, 100)
x = [r_i*cos(θ_j) for r_i in r, θ_j in θ]
y = [r_i*sin(θ_j) for r_i in r, θ_j in θ]

# Generate z values for each (x, y) pair
z = f.(x, y)

# Create a 3D surface plot
fig = Figure(resolution = (600, 400))
ax = Axis3(fig[1, 1])
surface!(ax, x, y, z, color = z, colormap = :viridis)

fig


using CairoMakie
f = Figure(size = (600, 400))
ax = Axis3(f[1, 1], xlabel="x_1", ylabel="x_2", zlabel="x_3")
lower = fill(Point3f(0,0,0), 100)
upper = [Point3f(sin(x), cos(x), 1.0) for x in range(0,2pi, length=100)]
col = repeat([1:50;50:-1:1],outer=2)
band(lower, upper, color=col, axis=(type=Axis3,xlabel="x₁",ylabel="x₂",zlabel="x₃"))


using JuMP
using SCS

A = [1.0 2.0 3.0;
     4.0 5.0 6.0]
b = [1.0, 2.0]
c = [3.0 4.0 5.0]

model = Model(SCS.Optimizer)
set_silent(model) 
@variable(model, x[1:3])
@variable(model, t) 
@constraint(model, A*x == b) 
@constraint(model, [t;x] in SecondOrderCone())
@objective(model, Min, t) 
optimize!(model)
value(t), value.(x)



using Convex, SCS

# Define the problem data and build the problem description
P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

# Create a vector variable of size n
x = Variable(2)

# Define the objective 
objective = 1/2*quadform(x,P) + dot(q,x)

# Define the constraints
constraints = [l <= A*x, A*x <= u]

# Define the overal description of the optimization problem
problem = minimize(objective, constraints)

# Solve the problem by calling solve!
solve!(problem, SCS.Optimizer; silent_solver = true)

# Check the status of the problem
problem.status # :Optimal, :Infeasible, :Unbounded etc.

# Get the optimum value
problem.optval

# Get the optimal x
x.value


Q = [1 1; 1 2]
r = [0, 1]

using LinearAlgebra

@show x_stationary = -Q\r
@show eigvals(Q)

f(x) = 1/2*dot(x,Q*x)+dot(x,r)
x1_data = x2_data = -4:0.1:4;  
f_data = [f([x1,x2]) for x2=x2_data, x1=x1_data];

using Plots
display(surface(x1_data,x2_data,f_data))
contour(x1_data,x2_data,f_data)
display(plot!([x_stationary[1]], [x_stationary[2]], marker=:circle, label="Stationary point"))



using LinearAlgebra         # For dot() function.
using Printf                # For formatted output.

x0 = [2, 3]                 # Initial vector.
Q = [1 0; 0 3]              # Positive definite matrix defining the quadratic form.
c = [1, 2]                   # Vector defining the linear part.

xs = -Q\c                   # Stationary point, automatically the minimizer for posdef Q. 

ϵ  = 1e-5                   # Threshold on the norm of the gradient.
N  = 100;                   # Maximum number of steps .

function gradient_descent_quadratic_exact(Q,c,x0,ϵ,N)
     x = x0
     iter = 0
     f = 1/2*dot(x,Q*x)+dot(x,c)
     ∇f = Q*x+c
     while (norm(∇f) > ϵ)
          f = 1/2*dot(x,Q*x)+dot(x,c)
          ∇f = Q*x+c
          α = dot(∇f,∇f)/dot(∇f,Q*∇f)
          x = x - α*∇f
          iter = iter+1
          @printf("i = %3d   ||∇f(x)|| = %6.4e   f(x) = %6.4e\n", iter, norm(∇f), f)
          if iter >= N
               return f,x
          end
     end
     return f,x
end

fopt,xopt = gradient_descent_quadratic_exact(Q,c,x0,ϵ,N)