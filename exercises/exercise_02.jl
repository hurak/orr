using ForwardDiff

# Define the function
f(x) = x^2 + 3x + 2

# Compute the derivative at a specific point (e.g., x = 5)
f′ = ForwardDiff.derivative(f, 5)

using Zygote # Reverse-mode automatic differentiation

# Compute the derivative at a specific point (e.g., x = 5)

∇f = Zygote.gradient(f, 5)[1]


# Vector function

g(x) = [x[1]^2 + x[2]^2, x[1]*x[2]]

# Compute the derivative at a specific point (e.g., x = [1, 2])

∇g = Zygote.gradient(g, [1.0, 2.0])[1] # fails 

∇g = Zygote.jacobian(g, [1.0, 2.0])[1] # works


## Reverse vs Forward mode

using BenchmarkTools

# Function with many inputs and one output

f(x) = sum(x.^2)

x = rand(1000)

@btime ForwardDiff.gradient(f, $x)

@btime Zygote.gradient(f, $x)

# Function with one input and many outputs

α = rand(1000)

g(x) =  (α .* x).^2 

x = rand(1)

@btime ForwardDiff.jacobian(g, $x)

@btime Zygote.jacobian(g, $x)[1]


## Mutation is no friend of automatic differentiation

using Zygote

function f(x)
    x[1] += 1
    return x[1]^2
end

x = [1.0]

∇f = Zygote.gradient(f, x)[1] # fails


## Enzyme - Another AD with more complex interface

using Enzyme

# Function with many inputs and one output - Reverse mode

f(x) = sqrt.(sum(x.^2))

x = rand(1000)

∇f = zeros(1000)

Enzyme.autodiff(Enzyme.Reverse, f, Duplicated(x, ∇f)) # gradient gets added to ∇f

@show ∇f

# Calling it again will add the gradient again

Enzyme.autodiff(Enzyme.Reverse, f, Duplicated(x, ∇f))

@show ∇f


## Common interface for automatic differentiation
using DifferentiationInterface

import Zygote
import Enzyme


x = rand(1000)

∇f_enzyme = DifferentiationInterface.gradient(f, AutoEnzyme(), x) # Enzyme call

∇f_zygote = DifferentiationInterface.gradient(f, AutoZygote(), x) # Zygote call

∇f_enzyme ≈ ∇f_zygote