import os
from collections import deque

import cv2
import numpy as np
import pygame
import torch

from game import Game, PlayerCar2, WIDTH, HEIGHT, FPS, CHECKPOINTS, TRACK_BORDER_MASK

OUT_ROOT = os.path.join(os.path.dirname(__file__), "expert_dataset")
os.makedirs(OUT_ROOT, exist_ok=True)

ACTION_MAP = {"forward": 0, "backward": 1, "left": 2, "right": 3, "stop": 4}


def get_local_camera_view(main_surface, car, camera_size=150):
    camera = pygame.Surface((camera_size, camera_size))
    camera.fill((0, 0, 0)) 
    car_center_x = car.x + (car.img.get_width() / 2)
    car_center_y = car.y + (car.img.get_height() / 2)
    
    offset_x = (camera_size / 2) - car_center_x
    offset_y = (camera_size / 2) - car_center_y
    
    camera.blit(main_surface, (offset_x, offset_y))
    return camera


def preprocess_frame(surface):
    rgb_frame = pygame.surfarray.array3d(surface).transpose(1, 0, 2)    
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    gray_expanded = np.expand_dims(gray, axis=-1)
    
    return gray_expanded.astype(np.float32) / 255.0


def collect_dataset():
    pygame.init()
    num_episodes = 5
    stack_size = 4
    
    for episode in range(1, num_episodes + 1):
        print(f"{episode}/{num_episodes}...")
        
        game = Game(WIDTH, HEIGHT, FPS, headless=False, show_rays=False)
        expert_car = PlayerCar2("Expert")
        game.add_car(expert_car)
        
        episode_dir = os.path.join(OUT_ROOT, f"epizod_{episode}")
        os.makedirs(episode_dir, exist_ok=True)
        
        frame_buffer = deque(maxlen=stack_size)
        
        game.draw()
        initial_cam = get_local_camera_view(game.win, expert_car, camera_size=150)
        initial_frame = preprocess_frame(initial_cam)
        for _ in range(stack_size):
            frame_buffer.append(initial_frame)
            
        step = 0
        running = True
        
        while running and len(game.cars) > 0:
            game.clock.tick(game.fps)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return
                    
            game.draw()
            
            cam_surface = get_local_camera_view(game.win, expert_car, camera_size=150)
            processed_frame = preprocess_frame(cam_surface)
            frame_buffer.append(processed_frame)
            
            expert_car.update_progress(CHECKPOINTS)
            _, distances = expert_car.get_rays_and_distances(TRACK_BORDER_MASK)
            car_distances = expert_car.get_distances_to_cars(game.cars)
            
            try:
                action = expert_car.choose_action([distances, car_distances, expert_car.get_progress(), CHECKPOINTS])
            except RuntimeError:
                action = "forward"
                
            action_idx = ACTION_MAP.get(action, 0)
            
            stacked_frames = np.stack(frame_buffer, axis=0)
            
            state_tensor = torch.from_numpy(stacked_frames)
            action_tensor = torch.tensor(action_idx, dtype=torch.long)
            
            sample_path = os.path.join(episode_dir, f"sample_{step:05d}.pt")
            torch.save({"state": state_tensor, "action": action_tensor}, sample_path)
            
            expert_car.perform_action(action)
            game.check_collisions()
            finish_lines = game.check_finish_line()
            
            if len(finish_lines) > 0:
                print(f"Koniec gry")
                running = False
                
            step += 1
            
        print(f"Nagranie {episode}: Zapisano {step} próbek")
        
    pygame.quit()

if __name__ == "__main__":
    collect_dataset()