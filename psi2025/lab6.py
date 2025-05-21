import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
sys.stdout.reconfigure(encoding='utf-8')

def function(x, y):
    return (1.5 - x - x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

## min funkcji to około (2.51799, -0.37463) i warttość funkcji to około 0.0135145 
def numpy_min():
    def func(vars):
        x, y = vars
        return (1.5 - x - x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
    
    bounds = [(-4.5, 4.5), (-4.5, 4.5)]
    x0 = [0.0, 0.0]
    result = minimize(func, x0, bounds=bounds)
    return result.x, result.fun

class Particle:
    def __init__(self, x, y, w, c1, c2):
        self.position = np.array([x, y])
        self.v = np.random.rand(2) * 0.1
        self.best_position = self.position.copy()
        self.best_value = function(x, y)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def update_velocity(self, global_best_position, current_w):
        r1 = np.random.rand(2)
        r2 = np.random.rand(2)

        # own_component = self.c1 * r1 * (self.position - self.best_position)
        # global_component = self.c2 * r2 * (self.position - global_best_position)
        own_component = self.c1 * r1 * (self.best_position - self.position)
        global_component = self.c2 * r2 * (global_best_position - self.position)

        self.v = current_w * self.v + own_component + global_component
        self.v = np.clip(self.v, -1, 1)


    def update_position(self):
        new_position = self.position + self.v
        if new_position[0] < -4.5 or new_position[0] > 4.5:
            self.v[0] = -self.v[0]
        if new_position[1] < -4.5 or new_position[1] > 4.5:
            self.v[1] = -self.v[1]
        
        self.position = new_position

        position_value = function(self.position[0], self.position[1])
        if position_value < self.best_value:
            self.best_value = position_value
            self.best_position = self.position.copy()


def pso_algorithm(num_particles, max_iter, w, c1, c2, size=[-4.5, 4.5]):
    def actualize_w(iteration, max_iterations, w_max=1.0, w_min=0.1):
        return w_max - ((w_max - w_min) * (iteration / max_iterations))
    
    optimal_result = numpy_min()
    
    min_history = []
    particles = [Particle(random.uniform(size[0], size[1]), random.uniform(size[0], size[1]),
                          w, c1, c2) for _ in range(num_particles)]
    global_best_position = particles[0].best_position
    global_best_value = function(global_best_position[0], global_best_position[1])

    for iter in range(max_iter):
        current_w = actualize_w(iter, max_iter, w_max=w, w_min=0.1)
        for particle in particles:
            particle.update_velocity(global_best_position, current_w)
            particle.update_position()

            if particle.best_value < global_best_value:
                min_history.append(particle.best_value)
                global_best_value = particle.best_value
                global_best_position = particle.best_position
        
        if global_best_value < optimal_result[1]:
            break
    # for particle in particles:
    #     print(f"Particle Position: {particle.position}, Value: {particle.best_value}")

    return global_best_position, global_best_value, min_history, iter

if __name__ == "__main__":
    configs = [
        {"w": 1.0, "c1": 1.0, "c2": 2.0}, ## najlepszy w grupie
        {"w": 1.0, "c1": 2.0, "c2": 2.0}, ## równe 
        {"w": 1.0, "c1": 2.0, "c2": 1.0}, ## najlepszy indywidualnie
    ]

    res_x, res_y = numpy_min()
    print(f"Optymalne rozwiązanie z użyciem scipy: x={res_x}, y={res_y:.6f}")
    print(70*"-")

    max_iter = 0
    y_lim = 0
    for config in configs:
        best_pos, best_val, min_history, iters = pso_algorithm(
            num_particles=30,
            max_iter=100,
            w=config["w"],
            c1=config["c1"],
            c2=config["c2"]
        )
        if len(min_history) > max_iter:
            max_iter = len(min_history)
        if max(min_history) > y_lim:
            y_lim = max(min_history)

        print(f"Różnica między najlepszym rozwiązaniem a PSO: {abs(best_val - res_y):.6f}")
        print(f"Config {config} => [X, Y]: {best_pos}, Minimum: {best_val:.6f} po {iters} iteracjach\n")
        plt.plot(min_history, label=f"w={config['w']}, c1={config['c1']}, c2={config['c2']}")

    plt.title("PSO dla różnych parametrów")
    plt.xlabel("liczba iteracji")
    plt.ylabel("najlepsza wartość")
    plt.ylim(0, y_lim)
    plt.xlim(0, max_iter)
    plt.legend()
    plt.grid()
    plt.show()
