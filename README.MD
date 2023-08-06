# 2D Heat Equation Solver using Finite Difference Method

This repository contains a Python script that implements a numerical solver for the 2D heat equation using the Finite Difference Method. The heat equation describes the diffusion of heat in a 2D domain over time. The script simulates the temperature distribution on a square domain and plots the final temperature distribution at a specified time.

The 2D heat equation describes the diffusion of heat in a 2D domain over time and is given by:

- ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)

![Figure_1](https://github.com/Deivison-Costa/Heat-Equation-Solver/assets/112001225/5da8be19-a49c-4f11-8581-a4b13ae1e651)

# Installation and Setup
- Clone the repository or download the source code.
- Make sure you have Python installed on your system (Python 3.6 or higher is recommended).
## Create a virtual environment (optional, but recommended):
### On Windows:
    pip install virtualenv
    virtualenv venv
    .\venv\Scripts\Activate.ps1
### On Linux:
    sudo apt-get install python3-venv
    python3 -m venv venv
    source meu_env/bin/activate
### Install the required dependencies:
    pip install numpy matplotlib

# Usage
- Ensure you are in the project directory.
## Run the heat_equation_solver.py file to start the Flask application:
    python heat_equation_solver.py

## Parameters:
- *Lx, Ly*: Size of the domain in the x and y directions, respectively.
- *Nx, Ny*: Number of grid points in the x and y directions, respectively.
- *alpha*: Thermal diffusivity, a material-specific property that describes how quickly heat diffuses through the material.
- *T_final*: Final simulation time, at which the distribution of temperature will be plotted.
- *dt*: Time step, or the size of the time intervals for the simulation.
- *dx, dy*: Spacing of grid points in the x and y directions, respectively.

## Functions:
- *initial_condition(x, y)*: This function defines the initial temperature distribution in the domain. It is used to set the initial values of the temperature matrix u.

- *heat_equation_solver()*: This function solves the 2D heat equation using the method of finite differences. It iteratively updates the temperature distribution over time steps until it reaches the final simulation time T_final.

- The core of the numerical solver is in the nested loops that update the temperature values using the finite difference method. The finite difference method approximates the derivatives in the heat equation with the differences between neighboring grid points.

# Technology Stack
- Python.
- NumPy.
- Matplotlib.
