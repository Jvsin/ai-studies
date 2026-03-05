import numpy as np
import matplotlib.pyplot as plt

days = np.arange(1, 366)

angles = 2 * np.pi * days / 365 # w radianach

day_sin = np.sin(angles)
day_cos = np.cos(angles)

plt.figure(figsize=(6, 6))
plt.plot(day_cos, day_sin, 'o')
plt.title('Kodowanie cykliczne dni w roku')
plt.xlabel('cos(day)')
plt.ylabel('sin(day)')
plt.axis('equal')
plt.grid()
plt.show()
