import numpy as np
import matplotlib.pyplot as plt

# Given values for inductance (L), capacitance (C), and target resonance frequency
L = 0.5  # Inductance in Henrys
C = 10e-6  # Capacitance in Farads
target_frequency = 1000  # Desired frequency in Hz
precision = 1e-3  # Precision for methods

# Function to calculate resonance frequency for a given resistance R
def resonance_frequency(R):
    sqrt_term = 1 / (L * C) - (R ** 2) / (4 * L ** 2)
    if sqrt_term <= 0:
        return None  # Invalid if negative inside square root
    return (1 / (2 * np.pi)) * np.sqrt(sqrt_term)

# Derivative of the resonance function for Newton-Raphson
def resonance_derivative(R):
    sqrt_term = 1 / (L * C) - (R ** 2) / (4 * L ** 2)
    if sqrt_term <= 0:
        return None
    return -R / (4 * np.pi * L ** 2 * np.sqrt(sqrt_term))

# Newton-Raphson method to find the resistance that results in target frequency
def newton_raphson(initial_R, tolerance):
    R = initial_R
    while True:
        freq_value = resonance_frequency(R)
        if freq_value is None:
            return None
        error = freq_value - target_frequency
        derivative = resonance_derivative(R)
        if derivative is None:
            return None
        new_R = R - error / derivative
        if abs(new_R - R) < tolerance:
            return new_R
        R = new_R

# Bisection method to find resistance within a given range
def bisection_method(a, b, tolerance):
    while (b - a) / 2 > tolerance:
        midpoint = (a + b) / 2
        freq_mid = resonance_frequency(midpoint) - target_frequency
        if freq_mid is None:
            return None
        if abs(freq_mid) < tolerance:
            return midpoint
        if (resonance_frequency(a) - target_frequency) * freq_mid < 0:
            b = midpoint
        else:
            a = midpoint
    return (a + b) / 2

# Initial parameters for finding resistance
initial_guess = 50
range_a, range_b = 0, 100

# Find resistance using both methods
resistance_newton = newton_raphson(initial_guess, precision)
frequency_newton = resonance_frequency(resistance_newton) if resistance_newton is not None else "Not found"
resistance_bisection = bisection_method(range_a, range_b, precision)
frequency_bisection = resonance_frequency(resistance_bisection) if resistance_bisection is not None else "Not found"

# Display results
print("Newton-Raphson Method:")
print(f"R: {resistance_newton} ohm, Resonance Frequency: {frequency_newton} Hz")
print("\nBisection Method:")
print(f"R: {resistance_bisection} ohm, Resonance Frequency: {frequency_bisection} Hz")

# Plotting the results
plt.figure(figsize=(10, 5))
plt.axhline(target_frequency, color="red", linestyle="--", label="Target Frequency 1000 Hz")

if resistance_newton is not None:
    plt.scatter(resistance_newton, frequency_newton, color="blue", label="Newton-Raphson")
    plt.text(resistance_newton, frequency_newton + 30, f"NR: R={resistance_newton:.2f}, f={frequency_newton:.2f} Hz", color="blue")

if resistance_bisection is not None:
    plt.scatter(resistance_bisection, frequency_bisection, color="green", label="Bisection")
    plt.text(resistance_bisection, frequency_bisection + 30, f"Bisection: R={resistance_bisection:.2f}, f={frequency_bisection:.2f} Hz", color="green")

plt.xlabel("Resistance R (Ohm)")
plt.ylabel("Resonance Frequency f(R) (Hz)")
plt.title("Comparison of Newton-Raphson and Bisection Methods")
plt.legend()
plt.grid(True)
plt.show()

# Gaussian and Gauss-Jordan elimination for solving a linear system
A = np.array([[1, 1, 1],
              [1, 2, -1],
              [2, 1, 2]], dtype=float)
b = np.array([6, 2, 10], dtype=float)

# Gaussian elimination function
def gaussian_elimination(A, b):
    n = len(b)
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])
    for i in range(n):
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:n])) / augmented_matrix[i, i]
    return x

# Gauss-Jordan elimination function
def gauss_jordan_elimination(A, b):
    n = len(b)
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])
    for i in range(n):
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]
        for j in range(n):
            if i != j:
                augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j, i]
    return augmented_matrix[:, -1]

# Solving the system using both methods
solution_gauss = gaussian_elimination(A, b)
solution_gauss_jordan = gauss_jordan_elimination(A, b)

print("Solution using Gaussian Elimination:")
print(f"x1 = {solution_gauss[0]}, x2 = {solution_gauss[1]}, x3 = {solution_gauss[2]}")
print("\nSolution using Gauss-Jordan Elimination:")
print(f"x1 = {solution_gauss_jordan[0]}, x2 = {solution_gauss_jordan[1]}, x3 = {solution_gauss_jordan[2]}")
