import numpy as np
import sys
import os
from Visualizer import plot_transmitters

def read_transmitters(filepath):
    radius = None
    transmitters = []

    with open(filepath, "r") as file:
        radius = float(file.readline())
        for t in file:
            x, y = map(lambda p: float(p), t.split(","))
            transmitters.append((x, y))

    return np.array(transmitters), radius

if __name__ == "__main__":
    num_transmitters = 100
    radius = 0.3
    new_seed = 42
    filename = "default"
    directory = "transmitters"

    if len(sys.argv) < 4:
        print("Not enough arguments. You can invoke this script like this:")
        print("python gen_transmitters.py <num_transmitters> <radius> <random_seed> <filename>")
        mode = input("However, do you want to create transmitters using [d]efault values or using [i]nteractive questions? d/i/n ")
        if mode == "n":
            sys.exit(1)
        if mode == "i":
            print("Transmitters will be generated inside of 2x2 square centered at (0, 0)")
            print("If you want to leave default, just press enter")

            user_input = input(f"Provide number of transmitters: (default {num_transmitters}) ")
            num_transmitters = int(user_input) if len(user_input) > 0 else num_transmitters

            user_input = input(f"Provide radius of transmitters: (default {radius}) ")
            radius = float(user_input) if len(user_input) > 0 else radius

            user_input = input(f"Provide seed for random number generator: (default {new_seed}) ")
            new_seed = int(user_input) if len(user_input) > 0 else new_seed

            user_input = input(f"Provide filename to save in the transmitters directory: (default is '{filename}') ")
            filename = user_input if len(user_input) > 0 else filename

    np.random.seed(new_seed)
    transmitters = np.random.rand(num_transmitters, 2) * 2 - 1

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f"{directory}/{filename}.txt", "w") as file:
        file.write(f"{radius}\n")
        for t in transmitters: file.write(f"{t[0]},{t[1]}\n")
    plot_transmitters(transmitters, np.ones(num_transmitters, dtype=bool), radius, title="Transmitters", save_path=f"{directory}/{filename}.png")
    print("Generated successfully")
