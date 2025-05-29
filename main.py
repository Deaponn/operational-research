import numpy as np
import sys
from Visualizer import Visualizer
from Bee import BeeAlgorithm
from Crossing import Crossing
from gen_transmitters import read_transmitters

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(len(sys.argv), sys.argv)
        print("Please provide algorithm type and file with transmitters")
        print("for example: python ./main.py bee transmitters/my_transmitters.txt")
        print("to create file with transmitters, use python gen_transmitters.py")
        sys.exit(1)

    transmitters, radius = read_transmitters(sys.argv[2])
    vis = Visualizer(transmitters, radius, "bee" if sys.argv[1] == "bee" else "crossing")
    best_member, best_score = None, None

    if sys.argv[1] == "bee":
        n_population = 50
        user_input = input(f"Provide population size: (default {n_population}) ")
        n_population = int(user_input) if len(user_input) != 0 else n_population

        num_generations = 100
        user_input = input(f"Provide number of generations: (default {num_generations}) ")
        num_generations = int(user_input) if len(user_input) != 0 else num_generations

        bee_algo = BeeAlgorithm(transmitters=transmitters,
                                radius=radius, num_bees=n_population)

        best_member, best_score = bee_algo.run_iteration(vis, num_generations)
    else:
        n_population = 50 # Population size, should be even number
        user_input = input(f"Provide population size: (default {n_population}) ")
        n_population = int(user_input) if len(user_input) != 0 else n_population

        n_generations = 100 # Number of generations
        user_input = input(f"Provide number of generations: (default {n_generations}) ")
        n_generations = int(user_input) if len(user_input) != 0 else n_generations

        n_crossover = 1.0 # A chance for crossover
        user_input = input(f"Provide chance of crossover: (default {n_crossover}) ")
        n_crossover = float(user_input) if len(user_input) != 0 else n_crossover

        n_mutation = 0.05 # A chance for mutation
        user_input = input(f"Provide chance of mutation: (default {n_mutation}) ")
        n_mutation = float(user_input) if len(user_input) != 0 else n_mutation

        crossing = Crossing(transmitters, radius, n_population, n_generations, n_crossover, n_mutation)
        mask = np.ones(len(transmitters), dtype=bool)
        best_member, best_score = crossing.run_iteration(vis)
    
    vis.add_frame(best_member, best_score, last_frame=True)
    vis.save_animation()
