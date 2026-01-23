import pygame
import numpy as np
from myAgent import MyAgent
from training_env import TrainingCar, spawn_random_car
from game import RED_CAR, WIDTH, HEIGHT, TRACK_BORDER_MASK

def generate_grid_cars(n_cars):
    cars = []
    start_positions = [
        (180, 200), (150, 200), 
        (180, 160), (150, 160),
        (180, 120), (150, 120)
    ]
    
    for i in range(n_cars):
        pos = start_positions[i % len(start_positions)]
        
        car = TrainingCar(RED_CAR, pos, 0) 
        cars.append(car)
        
    return cars


def train():
    pygame.init()
    # Okno musi istnieć, aby Pygame mógł przetwarzać maski i obrazy, 
    # nawet jeśli nie będziemy na nim rysować w każdej klatce.
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Training Agent (Press 'V' to toggle visuals)")
    
    clock = pygame.time.Clock()
    
    # Input dims: 8 (walls) + 8 (cars) + 1 (velocity) = 17 
    agent = MyAgent(input_dims=17, n_actions=5) 
    
    # Przełącznik renderowania - domyślnie wyłączony dla szybkości
    render_visuals = False 
    
    episodes = 2000 # Możemy dać więcej, bo będzie szybciej
    best_score = -9999
    
    n_cars = 6 # Więcej aut = stabilniejszy gradient
    
    for e in range(episodes):
        if e % 3 == 0:
        # Scenariusz 1: Start ze startu (Tłok)
            cars = generate_grid_cars(n_cars)
        else:
            # Scenariusz 2: Losowe miejsca (Nauka toru)
            cars = [spawn_random_car(TrainingCar, RED_CAR) for _ in range(n_cars)]

        # Ważne: Inicjalizacja stanów musi być PO stworzeniu listy cars
        states = [car.get_state([]) for car in cars]

        # cars = [spawn_random_car(TrainingCar, RED_CAR) for _ in range(n_cars)] # wersja pierwotna, odkomentuj to.
        
        scores = [0] * n_cars
        active_cars = [True] * n_cars
        states = [car.get_state([]) for car in cars]
        
        done_count = 0
        step = 0
        
        # Pętla epizodu
        while done_count < n_cars and step < 2000:
            # Obsługa zdarzeń (kluczowe, aby okno nie zamarzło)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_v:
                        render_visuals = not render_visuals
                        print(f"Visuals: {'ON' if render_visuals else 'OFF'}")

            # --- LOGIKA GRY (zawsze działa) ---
            actions = []
            for i, car in enumerate(cars):
                if active_cars[i]:
                    action = agent.choose_action(states[i])
                    actions.append(action)
                else:
                    actions.append(4) 

            for i, car in enumerate(cars):
                if active_cars[i]:
                    reward, done = car.step(actions[i])
                    scores[i] += reward
                    new_state = car.get_state([]) 
                    agent.store_transition(states[i], actions[i], reward, new_state, done)
                    states[i] = new_state
                    if done:
                        active_cars[i] = False
                        done_count += 1
                else:
                    pass # Auto nieaktywne

            agent.learn()
            
            if step % 100 == 0:
                agent.update_target_network()
            
            step += 1

            # --- RENDEROWANIE (warunkowe) ---
            if render_visuals:
                win.fill((0, 0, 0))
                # Rysujemy maskę toru jako tło
                win.blit(TRACK_BORDER_MASK.to_surface(setcolor=(255, 255, 255), unsetcolor=(0, 0, 0)), (0, 0))
                
                for i, car in enumerate(cars):
                    if active_cars[i]:
                        car.draw(win)
                
                # Wyświetlanie info na ekranie
                font = pygame.font.SysFont("Arial", 18)
                text = font.render(f"Ep: {e} Step: {step}", True, (255, 0, 0))
                win.blit(text, (10, 10))
                
                pygame.display.update()
                clock.tick(60) # Limitujemy FPS tylko gdy oglądamy, żeby ludzkie oko nadążyło
            else:
                # Gdy brak wizualizacji, nie robimy clock.tick() -> MAX SPEED
                pass

        # Logowanie wyników po epizodzie
        avg_score = np.mean(scores)
        if e % 10 == 0: # Żeby nie spamować konsoli, logujemy co 10
             print(f"Ep: {e} | Avg: {avg_score:.2f} | Best: {best_score:.2f} | Eps: {agent.epsilon:.2f}")
        
        # Zapisywanie modelu
        if avg_score > best_score and e > 50:
            best_score = avg_score
            agent.save(best_score)

    pygame.quit()

if __name__ == "__main__":
    train()