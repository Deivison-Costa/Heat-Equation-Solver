import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
Lx = Ly = 1.0      # Tamanho do domínio (comprimento e largura)
Nx = Ny = 50       # Número de pontos da malha nas direções x e y
alpha = 0.01       # Difusividade térmica
T_final = 0.1      # Tempo de simulação final
dt = 0.001         # Espaço de tempo
dx = Lx / (Nx - 1) # Espaçamento da malha em x
dy = Ly / (Ny - 1) # Espaçamento da malha em y

# Função de condição inicial
def initial_condition(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# Função de simulação principal usando o Método das Diferenças Finitas
def heat_equation_solver():
    u = np.zeros((Nx, Ny))  # Matriz de temperatura (inicialmente definida como zero)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # Definindo a condição inicial
    u[:, :] = initial_condition(X, Y)

    # Loop do espaço temporal
    num_steps = int(T_final / dt)
    for n in range(num_steps):
        u_new = u.copy()

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                u_new[i, j] = u[i, j] + alpha * dt * (
                    (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx**2 +
                    (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy**2
                )

        u = u_new

    return X, Y, u

# Resolvendo a equação do calor e obtendo os resultados
X, Y, u_final = heat_equation_solver()

# Plotando a distribuição de temperatura no tempo final
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, u_final, shading='auto')
plt.colorbar(label='Temperature')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'2D Heat Equation - Time: {T_final} seconds')
plt.show()