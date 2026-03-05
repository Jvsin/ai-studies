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
        self.max_vel = 10
        
        self.next_checkpoint_id = 0
        self.update_closest_checkpoint()

    def update_closest_checkpoint(self):
        dists = [math.hypot(self.x - cx, self.y - cy) for cx, cy in CHECKPOINTS]
        self.next_checkpoint_id = (np.argmin(dists) + 1) % len(CHECKPOINTS)

    def step(self, action_idx):
        actions = ["forward", "backward", "left", "right", "stop"]
        self.perform_action(actions[action_idx])
        
        reward = 0
        done = False
        
        if self.collide(TRACK_BORDER_MASK):
            reward = -15
            done = True
            return reward, done

        reward += (self.vel / self.max_vel) * 0.5

        target_ckpt = CHECKPOINTS[self.next_checkpoint_id]
        dist_to_ckpt = math.hypot(self.x - target_ckpt[0], self.y - target_ckpt[1])
        
        if dist_to_ckpt < 50:
            reward += 15
            self.next_checkpoint_id = (self.next_checkpoint_id + 1) % len(CHECKPOINTS)
        
        if self.vel < 0.1:
            reward -= 0.1

        return reward, done

    def get_state(self, other_cars=[]):
        rays, wall_dists = self.get_rays_and_distances(TRACK_BORDER_MASK)
        car_dists = self.get_distances_to_cars(other_cars)
        
        return [wall_dists, car_dists, [self.next_checkpoint_id, 0], self.vel]

def spawn_random_car(car_class, car_img):
    idx = np.random.randint(0, len(CHECKPOINTS) - 1)
    
    pos = CHECKPOINTS[idx]
    next_pos = CHECKPOINTS[idx + 1]
    
    dx = next_pos[0] - pos[0]
    dy = next_pos[1] - pos[1]
    angle = math.degrees(math.atan2(dy, dx)) + 90
    
    car = car_class(car_img, pos, -angle)
    return car