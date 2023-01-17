#「最適化問題のシミュレータ」アプリを作成するためのサンプルコード
import streamlit as st
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

st.title("Optimization Problem Simulator")

# Define the objective function
def objective_function(x):
    return x[0]**2 + x[1]**2

# Define the constraints
def constraint1(x):
    return x[0] + x[1] - 1

def constraint2(x):
    return 1 - x[0] - x[1]

# Set the optimization bounds
bounds = [(0, 1), (0, 1)]

# Set the initial guess
x0 = [0.5, 0.5]

# Run the optimization
result = minimize(objective_function, x0, bounds=bounds, constraints={"type": "eq", "fun": "constraint1"},
                  constraints={"type": "ineq", "fun": constraint2})

# Plot the results
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

plt.contour(X, Y, Z, levels=np.logspace(0, 2, 20))
plt.scatter(result.x[0], result.x[1])

st.pyplot()