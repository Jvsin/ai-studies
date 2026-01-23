import pygame
import math
import numpy as np
from abstract_car import AbstractCar
from game import TRACK_BORDER_MASK, CHECKPOINTS, TRACK_BORDER
from utils import blit_rotate_center

class TrainingCar(AbstractCar):
    def __init__(self, img, start_pos, start_angle):
        super().__init__("TrainCar")
        self.img = img
        self.set_position(start_pos)
        self.angle = start_angle
        self.mask = pygame.mask.from_surface(self.img)
        self.rect = self.img.get_rect(center=start_pos)
        self.vel = 0
        self.max_vel = 10 # Trochę szybciej dla treningu
        
        # Znajdź najbliższy checkpoint startowy
        self.next_checkpoint_id = 0
        self.update_closest_checkpoint()

    def update_closest_checkpoint(self):
        # Prosta logika do znalezienia najbliższego checkpointa na starcie
        dists = [math.hypot(self.x - cx, self.y - cy) for cx, cy in CHECKPOINTS]
        self.next_checkpoint_id = (np.argmin(dists) + 1) % len(CHECKPOINTS)

    def step(self, action_idx):
        # Mapowanie indeksu na akcje tekstowe
        actions = ["forward", "backward", "left", "right", "stop"]
        self.perform_action(actions[action_idx])
        
        # Zwracamy informację o kolizji i progresie
        reward = 0
        done = False
        
        # 1. Sprawdzenie kolizji ze ścianą
        if self.collide(TRACK_BORDER_MASK):
            reward = -15
            done = True
            return reward, done

        # 2. Nagroda za prędkość (chcemy żeby jechał szybko)
        reward += (self.vel / self.max_vel) * 0.5

        # 3. Sprawdzenie checkpointów
        target_ckpt = CHECKPOINTS[self.next_checkpoint_id]
        dist_to_ckpt = math.hypot(self.x - target_ckpt[0], self.y - target_ckpt[1])
        
        if dist_to_ckpt < 50: # Zaliczenie checkpointa
            reward += 15
            self.next_checkpoint_id = (self.next_checkpoint_id + 1) % len(CHECKPOINTS)
            # Bonus za czas (opcjonalne, by zapobiec kręceniu się)
        
        # Kara za stanie w miejscu
        if self.vel < 0.1:
            reward -= 0.1

        return reward, done

    def get_state(self, other_cars=[]):
        # Raycasting
        rays, wall_dists = self.get_rays_and_distances(TRACK_BORDER_MASK)
        car_dists = self.get_distances_to_cars(other_cars)
        
        # Zwracamy stan rozszerzony o prędkość (kluczowe!)
        # Struktura: [Walls, Cars, [Checkpoint_Idx, Progress], Velocity]
        return [wall_dists, car_dists, [self.next_checkpoint_id, 0], self.vel]

def spawn_random_car(car_class, car_img):
    # Wybierz losowy checkpoint (oprócz ostatniego, żeby nie było dziwnych bugów z pętlą)
    idx = np.random.randint(0, len(CHECKPOINTS) - 1)
    
    pos = CHECKPOINTS[idx]
    next_pos = CHECKPOINTS[idx + 1]
    
    # Oblicz kąt, żeby auto patrzyło w stronę następnego punktu
    dx = next_pos[0] - pos[0]
    dy = next_pos[1] - pos[1]
    angle = math.degrees(math.atan2(dy, dx)) + 90 # +90 bo w pygame 0 to góra/prawo zależnie od sprite'a
    # Korekta kąta zależna od orientacji Twojego sprite'a (często -90 lub +90)
    # W abstract_car: angle 0 to zazwyczaj góra. atan2 zwraca kąt matematyczny.
    # Warto eksperymentalnie dobrać offset, tutaj zakładam standard.
    
    car = car_class(car_img, pos, -angle) # Negacja kąta dla Pygame
    return car