import random
import numpy as np
import matplotlib.pyplot as plt


def function(x, y):
    return (1.5 - x - x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

class Particle:
    def __init__(self, x, y, w, c1, c2):
        self.position = np.array([x, y])
        self.v = np.random.rand(2) * 0.1
        self.best_position = self.position.copy()
        self.best_value = function(x, y)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def update_velocity(self, global_best_position):

        r1 = np.random.rand(2)
        r2 = np.random.rand(2)

        own_component = self.c1 * r1 * (self.position - self.best_position)
        global_component = self.c2 * r2 * (self.position - global_best_position)

        self.velocity = self.w * self.v + own_component + global_component

    def update_position(self):
        self.position += self.v

        position_value = function(self.position[0], self.position[1])
        if position_value < self.best_value:
            self.best_value = position_value
            self.best_position = self.position.copy()


def pso(num_particles, max_iter, w, c1, c2, size=[-4.5, 4.5]):
    min_history = []
    particles = [Particle(random.uniform(size[0], size[1]), random.uniform(size[0], size[1]),
                          w, c1, c2) for _ in range(num_particles)]
    global_best_position = np.array([0, 0])
    global_best_value = np.array([float("inf")])

    for _ in range(max_iter):
        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.update_position()

            if particle.best_value < global_best_value:
                min_history.append(particle.best_value)
                global_best_value = particle.best_value
                global_best_position = particle.best_position

    return global_best_position, global_best_value, min_history

# print(pso(30, 1000))

configs = [
    {"w": 0.5, "c1": 1.5, "c2": 1.5},
    {"w": 0.7, "c1": 2.0, "c2": 2.0},
    {"w": 0.9, "c1": 2.5, "c2": 0.5},
]

for config in configs:
    best_pos, best_val, min_history = pso(
        num_particles=30,
        max_iter=1000,
        w=config["w"],
        c1=config["c1"],
        c2=config["c2"]
    )
    print(f"Config {config} => Best Pos: {best_pos}, Best Val: {best_val:.6f}")
    plt.plot(min_history, label=f"w={config['w']}, c1={config['c1']}, c2={config['c2']}")

plt.title("Convergence of PSO with Different Parameters")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness Value")
plt.legend()
plt.grid()
plt.show()
