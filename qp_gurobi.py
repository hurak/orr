#!/usr/bin/env python

import gurobipy as gp
import numpy as np

# Define the data for the model

P = np.array([[4.0, 1.0], [1.0, 2.0]])
q = np.array([1.0, 1.0])
A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
l = np.array([1.0, 0.0, 0.0])
u = np.array([1.0, 0.7, 0.7])

# Create a new model
m = gp.Model("qp")

# Create a vector variable
x = m.addMVar((2,))

# Set the objective
obj = 1/2*(x@P@x + q@x)
m.setObjective(obj)

# Add the constraints
m.addConstr(A@x >= l, "c1")
m.addConstr(A@x <= u, "c2")

m.optimize()

for v in m.getVars():
    print(f"{v.VarName} {v.X:g}")

print(f"Obj: {m.ObjVal:g}")


