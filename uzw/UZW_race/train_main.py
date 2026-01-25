import pygame
import numpy as np
import random
from collections import deque
from myAgent import MyAgent
from training_env import TrainingCar, spawn_random_car, generate_grid_cars
from game import RED_CAR, WIDTH, HEIGHT, TRACK_BORDER_MASK

BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
MEMORY_SIZE = 50000
MAX_STEPS = 2000

def train():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Training")
    clock = pygame.time.Clock()

    agent = MyAgent()
    memory = deque(maxlen=MEMORY_SIZE)
    
    epsilon = EPSILON_START
    episodes = 1000
    n_cars = 4
    
    render = True

    for e in range(episodes):
        if e % 3 == 0:
            cars = generate_grid_cars(n_cars, TrainingCar, RED_CAR)
        else:
            cars = [spawn_random_car(TrainingCar, RED_CAR) for _ in range(n_cars)]
            
        states = [car.get_state([]) for car in cars]
        scores = [0] * n_cars
        active = [True] * n_cars
        
        step = 0
        run = True
        
        while run and any(active) and step < MAX_STEPS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                    render = not render
                    print(f"Render: {render}")

            actions = []
            for i, car in enumerate(cars):
                if active[i]:
                    if np.random.random() < epsilon:
                        actions.append(np.random.randint(0, 5))
                    else:
                        q_values = agent.predict(states[i])
                        actions.append(np.argmax(q_values))
                else:
                    actions.append(4)

            for i, car in enumerate(cars):
                if active[i]:
                    reward, done = car.step(actions[i])
                    next_state = car.get_state([])
                    scores[i] += reward
                    
                    # Zapisz do pamięci: (stan, akcja, nagroda, nowy_stan, czy_koniec)
                    memory.append((states[i], actions[i], reward, next_state, done))
                    states[i] = next_state
                    
                    if done:
                        active[i] = False

            if len(memory) > BATCH_SIZE:
                mini_batch = random.sample(memory, BATCH_SIZE)
                
                batch_states = [m[0] for m in mini_batch]
                batch_actions = [m[1] for m in mini_batch]
                batch_rewards = [m[2] for m in mini_batch]
                batch_next_states = [m[3] for m in mini_batch]
                batch_is_done = [m[4] for m in mini_batch]
                
                current_qs = agent.predict_batch(batch_states) # Obecne przewidywania
                next_qs = agent.predict_batch(batch_next_states) # Co przewidujemy po ruchu
                
                X = batch_states
                y = current_qs.copy()
                
                for i in range(BATCH_SIZE):
                    # wzor bellmana:
                    target = batch_rewards[i]
                    if not batch_is_done[i]:
                        target += GAMMA * np.max(next_qs[i])
                    
                    # aktualizujemy tylko te akcje, którą wykonaliśmy
                    y[i][batch_actions[i]] = target
                
                # Trenujemy sieć!
                agent.fit(X, y)

            if render:
                win.fill((0, 0, 0))
                win.blit(TRACK_BORDER_MASK.to_surface(setcolor=(255,255,255), unsetcolor=(0,0,0)), (0, 0))
                for i, car in enumerate(cars):
                    if active[i]: car.draw(win)
                pygame.display.update()
                clock.tick(60)
            
            step += 1

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        avg_score = np.mean(scores)
        print(f"Ep: {e} | Avg: {avg_score:.2f} | Best: {agent.best_reward:.2f} | Eps: {epsilon:.2f}")
        
        agent.save(avg_score)

    pygame.quit()

if __name__ == "__main__":
    train()