import pygame
import math
import numpy as np
from abstract_car import AbstractCar
from game import TRACK_BORDER_MASK, CHECKPOINTS
from utils import blit_rotate_center

BORDER_REWARD = -20
NO_MOVE_REWARD = -0.1
CHECKPOINT_REWARD = 20

class TrainingCar(AbstractCar):
    def __init__(self, img, start_pos, start_angle):
        super().__init__("TrainCar")
        self.img = img
        self.set_position(start_pos)
        self.angle = start_angle
        self.mask = pygame.mask.from_surface(self.img)
        self.rect = self.img.get_rect(center=start_pos)
        self.vel = 0
        self.max_vel = 10 
        
        # next checkpoint
        dists = [math.hypot(self.x - cx, self.y - cy) for cx, cy in CHECKPOINTS]
        self.next_checkpoint_id = (np.argmin(dists) + 1) % len(CHECKPOINTS)

    def step(self, action_idx):
        if action_idx == 0:
            self.move_forward()
        elif action_idx == 1:
            self.move_backward()
        elif action_idx == 2:
            self.rotate(left=True)
        elif action_idx == 3:
            self.rotate(right=True)
        elif action_idx == 4:
            self.reduce_speed()
        
        reward = 0
        done = False
        
        if self.collide(TRACK_BORDER_MASK):
            reward = BORDER_REWARD
            done = True
            return reward, done

        reward += (self.vel / self.max_vel) * 0.5
        
        if self.vel < 0.1:
            reward -= NO_MOVE_REWARD

        target_ckpt = CHECKPOINTS[self.next_checkpoint_id]
        dist_to_checkpoint = math.hypot(self.x - target_ckpt[0], self.y - target_ckpt[1])
        
        if dist_to_checkpoint < 50:
            reward += CHECKPOINT_REWARD
            self.next_checkpoint_id = (self.next_checkpoint_id + 1) % len(CHECKPOINTS)
            
        return reward, done

    def get_state(self, other_cars=[]):
        _, wall_dists = self.get_rays_and_distances(TRACK_BORDER_MASK)
        car_dists = self.get_distances_to_cars(other_cars)
        
        return [wall_dists, car_dists, [self.next_checkpoint_id, 0], CHECKPOINTS, self.vel]

def spawn_random_car(car_class, car_img):
    idx = np.random.randint(0, len(CHECKPOINTS) - 1)
    pos = CHECKPOINTS[idx]
    next_pos = CHECKPOINTS[idx + 1]
    
    dx = next_pos[0] - pos[0]
    dy = next_pos[1] - pos[1]
    angle = math.degrees(math.atan2(dy, dx)) + 90 
    
    return car_class(car_img, pos, -angle)

def generate_grid_cars(n_cars, car_class, car_img):
    cars = []
    
    start_positions = [
        (180, 200), (150, 200), 
        (180, 160), (150, 160),
        # (180, 120), (150, 120)
    ]
    for i in range(n_cars):
        pos = start_positions[i % len(start_positions)]
        cars.append(car_class(car_img, pos, 0))
    return cars