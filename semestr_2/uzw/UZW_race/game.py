import os
import pygame
from abstract_car import AbstractCar
from utils import scale_image
from itertools import permutations
import numpy as np
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class ImitationCNN(nn.Module):
    def __init__(self, in_channels: int = 4, n_actions: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(64 * 17 * 17, 256) 
        self.q_out = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.q_out(x)

# Use imgs directory next to this file so loading works from any working dir
IMG_DIR = os.path.join(os.path.dirname(__file__), "imgs")

#Based on https://github.com/techwithtim/Pygame-Car-Racer

GRASS = scale_image(pygame.image.load(os.path.join(IMG_DIR, "grass.jpg")), 2.5)
TRACK = scale_image(pygame.image.load(os.path.join(IMG_DIR, "track.png")), 0.9)

TRACK_BORDER = scale_image(pygame.image.load(os.path.join(IMG_DIR, "track-border.png")), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = pygame.image.load(os.path.join(IMG_DIR, "finish.png"))
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)

RED_CAR = scale_image(pygame.image.load(os.path.join(IMG_DIR, "red-car.png")), 0.35)
GREEN_CAR = scale_image(pygame.image.load(os.path.join(IMG_DIR, "green-car.png")), 0.35)
PURPLE_CAR = scale_image(pygame.image.load(os.path.join(IMG_DIR, "purple-car.png")), 0.35)
GRAY_CAR = scale_image(pygame.image.load(os.path.join(IMG_DIR, "grey-car.png")), 0.35)


WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")

pygame.font.init()  # Initialize the font module
FONT = pygame.font.Font(None, 24)  # Use a default font with size 24


FPS = 60

track_path =  [(175, 119), (110, 70), (56, 133), (70, 481), (318, 731), (404, 680), (418, 521), (507, 475), (600, 551), (613, 715), (736, 713),
        (734, 399), (611, 357), (409, 343), (433, 257), (697, 258), (738, 123), (581, 71), (303, 78), (275, 377), (176, 388), (178, 260)]


# Interpolate evenly spaced checkpoints
def generate_checkpoints(track_path, num_checkpoints=100):
    checkpoints = []
    for i in range(len(track_path) - 1):
        x1, y1 = track_path[i]
        x2, y2 = track_path[i + 1]
        for t in np.linspace(0, 1, num_checkpoints // len(track_path)):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            checkpoints.append((int(x), int(y)))
    return checkpoints


CHECKPOINTS = generate_checkpoints(track_path)

def draw_checkpoints(win, checkpoints):
    for x, y in checkpoints:
        pygame.draw.circle(win, (0, 255, 0), (x, y), 5)

# In the game loop


class Game:
    def __init__(self, width, height, fps=60, headless=False, show_rays=False):
        """If headless=True, render to an off-screen Surface and do not open a window.
        If show_rays=False, do not draw sensor rays on cars."""
        self.headless = headless
        self.show_rays = show_rays
        if not self.headless:
            self.win = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Racing Game")
        else:
            self.win = pygame.Surface((width, height))

        self.clock = pygame.time.Clock()
        self.fps = fps
        self.cars = []  # List to hold car objects
        self.images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
            (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
        self.running = True

    def add_car(self, car):
        """Add a car to the game."""
        if not isinstance(car, AbstractCar):
            raise ValueError("Only instances of AbstractCar or its subclasses can be added.")

        if len(self.cars) == 0:
            car.set_image(RED_CAR)
            car.set_position((180, 200))
        elif len(self.cars) == 1:
            car.set_image(GREEN_CAR)
            car.set_position((150, 200))
        if len(self.cars) == 2:
            car.set_image(GRAY_CAR)
            car.set_position((180, 160))
        elif len(self.cars) == 3:
            car.set_image(PURPLE_CAR)
            car.set_position((150, 160))

        car.reset()
        if hasattr(car, "set_game_context"):
            car.set_game_context(self)
        self.cars.append(car)

    def draw(self):
        """Draw the background and all cars."""
        for img, pos in self.images:
            self.win.blit(img, pos)

        for car in self.cars:
            car.draw(self.win)
            if self.show_rays:
                car.draw_rays(self.win, TRACK_BORDER_MASK)

        # Update display only when not headless
        if not self.headless:
            pygame.display.update()

    def check_collisions(self):

        for car in self.cars:
            if car.collide(TRACK_BORDER_MASK):
                car.bounce()

        """Check for collisions between cars."""
        for i, car1 in enumerate(self.cars):
            for j, car2 in enumerate(self.cars):
                if i != j and car1.collide_car(car2):
                    car1.bounce()
                    car2.bounce()
                    # print(f"Collision between Car {i+1} and Car {j+1}!")

    def check_finish_line(self):

        finished = []

        for car in self.cars:
            finish_poi_collide = car.collide(FINISH_MASK, *FINISH_POSITION)
            if finish_poi_collide != None:
                if finish_poi_collide[1] == 0:
                    car.bounce()
                else:
                    finished.append(car.get_name())
                    self.cars.remove(car)

        return finished

    def move_cars(self):
        """Handle car movements."""

        for car in self.cars:
            car.update_progress(CHECKPOINTS)

        for car in self.cars:
            _, distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
            car_distances = car.get_distances_to_cars(self.cars)
            car.perform_action(car.choose_action([distances, car_distances, car.get_progress(), CHECKPOINTS]))
    def run(self):
        """Main game loop."""
        who_finished_first = []
        # Build the initial rendered frame so image-based agents can act on step 1.
        self.draw()
        while self.running and len(self.cars) != 0:
            self.clock.tick(self.fps)
            # draw_checkpoints(self.win, CHECKPOINTS)
            # pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False



            self.move_cars()
            self.check_collisions()
            finish_lines = self.check_finish_line()
            if len(finish_lines) != 0:
                who_finished_first.append(finish_lines)

            self.draw()


        pygame.quit()
        print("Game over!")
        print(who_finished_first)
        return who_finished_first


class PlayerCar(AbstractCar):

    def __init__(self, name):
        # Call the AbstractCar __init__ method
        super().__init__(name)

    def choose_action(self, state):
        """
        Perform an action based on the input.

        Actions:
        - "forward": Move the car forward.
        - "backward": Move the car backward.
        - "left": Turn the car left.
        - "right": Turn the car right.
        - "stop": Reduce the car's speed.
        """

        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            return "forward"
        elif keys[pygame.K_DOWN]:
            return "backward"
        elif keys[pygame.K_LEFT]:
            return "left"
        elif keys[pygame.K_RIGHT]:
            return "right"
        else:
            return "stop"


class PlayerCar2(AbstractCar):
    def __init__(self, name):
        super().__init__(name)
        from myAgent import MyAgent
        self.agent = MyAgent()
        self.agent.load()

    def choose_action(self, state):
        full_state = [
            state[0],
            state[1],
            state[2],
            CHECKPOINTS,
            self.vel
        ]
        
        q_values = self.agent.predict(full_state)
        action_idx = np.argmax(q_values)
        
        actions = ["forward", "backward", "left", "right", "stop"]
        return actions[action_idx]


class PlayerCarImageImitation(AbstractCar):
    def __init__(self, name, model_path="records/best_imitation_model.pth"):
        super().__init__(name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = ImitationCNN(in_channels=4, n_actions=5).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(f"Załadowano wyuczony model wizualny z: {model_path}")
        else:
            print(f"Brak znalezionego modelu w {model_path}")
            
        self.model.eval()
        
        self.actions = ["forward", "backward", "left", "right", "stop"]
        self.game = None
        
        self._frame_buffer = deque(maxlen=4)

    def set_game_context(self, game):
        self.game = game
        
        self.game.draw()
        initial_cam = self._get_local_view()
        initial_processed = self._preprocess(initial_cam)
        for _ in range(4):
            self._frame_buffer.append(initial_processed)

    def _get_local_view(self):
        camera_size = 150
        camera = pygame.Surface((camera_size, camera_size))
        camera.fill((0, 0, 0))
        
        car_center_x = self.x + (self.img.get_width() / 2)
        car_center_y = self.y + (self.img.get_height() / 2)
        
        offset_x = (camera_size / 2) - car_center_x
        offset_y = (camera_size / 2) - car_center_y
        
        camera.blit(self.game.win, (offset_x, offset_y))
        return camera

    def _preprocess(self, surface):
        rgb_frame = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        gray_expanded = np.expand_dims(gray, axis=-1)
        return gray_expanded.astype(np.float32) / 255.0

    def choose_action(self, state):
        if self.game is None:
            return "stop"

        current_cam = self._get_local_view()
        processed_frame = self._preprocess(current_cam)
        self._frame_buffer.append(processed_frame)

        stacked = np.stack(self._frame_buffer, axis=0)
        
        state_tensor = torch.from_numpy(stacked).squeeze(-1)
        state_tensor = state_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(state_tensor)
            action_idx = int(outputs.argmax(dim=1).item())

        return self.actions[action_idx]


def main():

    final_results = dict()

    #initializing players - it is possible to play up to 4 players together
    # players = [PlayerCar2("P1"), PlayerCar2("P2"), PlayerCar2("P3"), PlayerCar2("P4")]
    # players = [PlayerCar2("Gracz")]
    players = [PlayerCarImageImitation("ModelAI")]

    for p in players:
        final_results[p.get_name()] = 0

    perm = permutations(players)

    for p in perm:

        print(p)

        game = Game(WIDTH, HEIGHT, FPS)

        # Add cars
        for player in p:
            game.add_car(player)

        # Run the game
        temp_rank = game.run()

        points = len(players)

        for tr in temp_rank:
            for t in tr:
                final_results[t] += points
            points -= 1

    print(final_results)

if __name__ == "__main__":
    main()