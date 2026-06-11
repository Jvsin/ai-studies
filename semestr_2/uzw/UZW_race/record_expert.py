import os
import json
import time
from collections import deque

import pygame

from game import Game, PlayerCar2, WIDTH, HEIGHT, FPS, CHECKPOINTS, TRACK_BORDER_MASK


EXPERT_DIR = os.path.join(os.path.dirname(__file__), "expert")
os.makedirs(EXPERT_DIR, exist_ok=True)


def save_surface(surface, path):
    # surface is a pygame.Surface
    pygame.image.save(surface, path)


def record(max_samples=1000, max_steps=20000):
    pygame.init()

    # run headless and without sensor trackers to speed up recording
    game = Game(WIDTH, HEIGHT, FPS, headless=True, show_rays=False)

    # single expert agent
    expert = PlayerCar2("Expert")
    game.add_car(expert)

    frames = deque(maxlen=4)

    # render initial frame and fill deque
    game.draw()
    initial = game.win.copy()
    for _ in range(4):
        frames.append(initial.copy())

    sample_idx = 0
    records_path = os.path.join(EXPERT_DIR, "records.jsonl")

    running = True
    steps = 0

    # action mapping (string -> index)
    ACTIONS = ["forward", "backward", "left", "right", "stop"]

    with open(records_path, "a", encoding="utf-8") as records_file:
        while running and sample_idx < max_samples and steps < max_steps and len(game.cars) > 0:
            game.clock.tick(game.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # update progress for all cars (same as Game.move_cars does)
            for car in game.cars:
                car.update_progress(CHECKPOINTS)

            for car in list(game.cars):
                # compute sensor state used by PlayerCar2
                _, distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
                car_distances = car.get_distances_to_cars(game.cars)

                prev_progress = car.get_progress()[0]

                # choose action using agent
                action = car.choose_action([distances, car_distances, car.get_progress(), CHECKPOINTS])

                # ensure we have 4 frames for the "state"
                state_frames = list(frames)

                # perform action
                car.perform_action(action)

                # handle collisions and finish line as in main loop
                game.check_collisions()
                finished = game.check_finish_line()

                # draw new frame and append to frames
                game.draw()
                frames.append(game.win.copy())

                next_frames = list(frames)

                new_progress = car.get_progress()[0]

                # simple reward: +1 for progress, -1 for collision, small negative otherwise
                collision = car.collide(TRACK_BORDER_MASK)
                if collision:
                    reward = -1.0
                elif new_progress > prev_progress:
                    reward = 1.0
                else:
                    reward = -0.1

                # save sample
                sample_dir = os.path.join(EXPERT_DIR, f"sample_{sample_idx:06d}")
                os.makedirs(sample_dir, exist_ok=True)

                # save state frames
                for i, surf in enumerate(state_frames):
                    save_surface(surf, os.path.join(sample_dir, f"state_f{i}.png"))

                # save next frames
                for i, surf in enumerate(next_frames):
                    save_surface(surf, os.path.join(sample_dir, f"next_f{i}.png"))

                # metadata
                meta = {
                    "sample": sample_idx,
                    "action": action,
                    "action_idx": ACTIONS.index(action) if action in ACTIONS else None,
                    "reward": reward,
                    "collision": bool(collision),
                    "prev_progress": int(prev_progress),
                    "new_progress": int(new_progress),
                    "car_pos": [float(car.x), float(car.y)],
                    "timestamp": time.time()
                }

                with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                # append to records.jsonl for easy batching
                records_file.write(json.dumps(meta, ensure_ascii=False) + "\n")
                records_file.flush()

                sample_idx += 1
                steps += 1

                # stop early if finished
                if finished:
                    running = False
                    break

    pygame.quit()
    print(f"Recorded {sample_idx} samples to {EXPERT_DIR}")


if __name__ == "__main__":
    # adjust max_samples as needed
    record(max_samples=2000, max_steps=50000)
