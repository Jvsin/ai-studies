import numpy as np
import matplotlib.pyplot as plt

days = np.arange(1, 366)

# Kąt dla każdego dnia (w radianach)
angles = 2 * np.pi * days / 365

# Kodowanie cykliczne
day_sin = np.sin(angles)
day_cos = np.cos(angles)

# Dla przykładu - wrzućmy na wykres:
plt.figure(figsize=(6, 6))
plt.plot(day_cos, day_sin, 'o')
plt.title('Kodowanie cykliczne dni w roku')
plt.xlabel('cos(day)')
plt.ylabel('sin(day)')
plt.axis('equal')  # Równe osie -> będzie okrąg
plt.grid()
plt.show()


# Liczba dni w miesiącu
days = np.arange(1, 32)

# Obliczanie kątów
angles = 2 * np.pi * days / 31

# Obliczanie sin i cos
day_sin = np.sin(angles)
day_cos = np.cos(angles)

# Tworzenie wykresu
plt.figure(figsize=(6, 6))
plt.scatter(day_cos, day_sin, c=days, cmap='hsv', s=100)
for i, day in enumerate(days):
    plt.text(day_cos[i]*1.1, day_sin[i]*1.1, str(day), ha='center', va='center', fontsize=8)

plt.title("Kodowanie cykliczne dni miesiąca (sin/cos)")
plt.xlabel("cos komponenta")
plt.ylabel("sin komponenta")
plt.grid(True)
plt.axis("equal")
plt.show()

# Liczba miesięcy
months = np.arange(1, 13)

# Obliczanie kątów
month_angles = 2 * np.pi * months / 12

# Obliczanie sin i cos
month_sin = np.sin(month_angles)
month_cos = np.cos(month_angles)

# Tworzenie wykresu
plt.figure(figsize=(6, 6))
plt.scatter(month_cos, month_sin, c=months, cmap='hsv', s=150)
for i, month in enumerate(months):
    plt.text(month_cos[i]*1.15, month_sin[i]*1.15, str(month), ha='center', va='center', fontsize=10)

plt.title("Kodowanie cykliczne miesięcy (sin/cos)")
plt.xlabel("cos komponenta")
plt.ylabel("sin komponenta")
plt.grid(True)
plt.axis("equal")
plt.show()
