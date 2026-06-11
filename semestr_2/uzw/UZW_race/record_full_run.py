import os
import json
import time
from collections import deque

import pygame

from game import Game, PlayerCar2, WIDTH, HEIGHT, FPS, CHECKPOINTS, TRACK_BORDER_MASK


OUT_ROOT = os.path.join(os.path.dirname(__file__), "expert", "full_runs")
os.makedirs(OUT_ROOT, exist_ok=True)


def save_surface(surface, path):
    pygame.image.save(surface, path)


def record_full_run(max_steps=100000):
    pygame.init()

    # run headless and without sensor trackers to speed up recording
    game = Game(WIDTH, HEIGHT, FPS, headless=True, show_rays=False)

    # add single expert car (PlayerCar2 loads MyAgent)
    expert_car = PlayerCar2("Expert")
    game.add_car(expert_car)

    frames = deque(maxlen=4)

    # draw initial frame multiple times to fill history
    game.draw()
    initial = game.win.copy()
    for _ in range(4):
        frames.append(initial.copy())

    # prepare run folder
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(OUT_ROOT, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    meta_path = os.path.join(run_dir, "meta.jsonl")
    action_map = ["forward", "backward", "left", "right", "stop"]

    step = 0
    running = True

    with open(meta_path, "w", encoding="utf-8") as meta_file:
        while running and step < max_steps and len(game.cars) > 0:
            game.clock.tick(game.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # update progress
            for car in game.cars:
                car.update_progress(CHECKPOINTS)

            # capture current frame
            curr_frame = game.win.copy()
            frame_filename = f"frame_{step:06d}.png"
            frame_path = os.path.join(frames_dir, frame_filename)
            save_surface(curr_frame, frame_path)

            # append to history
            frames.append(curr_frame.copy())

            # for each car (we only use the expert)
            for car in list(game.cars):
                _, distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
                car_distances = car.get_distances_to_cars(game.cars)
                prev_progress = car.get_progress()[0]

                # agent chooses action
                action = car.choose_action([distances, car_distances, car.get_progress(), CHECKPOINTS])

                # perform action
                car.perform_action(action)

                # collisions/finish
                game.check_collisions()
                finished = game.check_finish_line()

                # draw after action
                game.draw()

                # compute reward
                new_progress = car.get_progress()[0]
                collision = car.collide(TRACK_BORDER_MASK)
                if collision:
                    reward = -1.0
                elif new_progress > prev_progress:
                    reward = 1.0
                else:
                    reward = -0.1

                # store metadata: indices of frames used for state (last 4) and next (next 4)
                # current frames deque has last up to 4 frames including curr_frame
                state_frame_indices = list(range(max(0, step - len(frames) + 1), step + 1))

                # take a snapshot for next frame (game.win already updated)
                next_frame = game.win.copy()
                next_frame_filename = f"frame_{step+1:06d}.png"
                next_frame_path = os.path.join(frames_dir, next_frame_filename)
                save_surface(next_frame, next_frame_path)

                # append next frame to deque for consistency
                frames.append(next_frame.copy())

                next_frame_indices = list(range(step+1 - len(frames) + 1 + 1, step+2))

                meta = {
                    "step": step,
                    "state_frames": [f"frames/frame_{i:06d}.png" for i in state_frame_indices[-4:]],
                    "next_frames": [f"frames/frame_{i:06d}.png" for i in (state_frame_indices + [step+1])[-4:]],
                    "action": action,
                    "action_idx": action_map.index(action) if action in action_map else None,
                    "reward": reward,
                    "collision": bool(collision),
                    "car_pos": [float(car.x), float(car.y)],
                    "timestamp": time.time()
                }

                meta_file.write(json.dumps(meta, ensure_ascii=False) + "\n")
                meta_file.flush()

                step += 1

                if finished:
                    running = False
                    break

    pygame.quit()
    print(f"Saved full run to: {run_dir}, frames: {step+1}")


if __name__ == "__main__":
    record_full_run()
