import sys
from timeit import default_timer as timer
from main import run_bee, run_crossing
from Visualizer import MockVisualizer

if __name__ == "__main__":
    transmitters_filenames = ["transmitters/smatt.txt", "transmitters/default.txt"]

    # bee algo, total 30 combinations
    num_bees_list = [20, 35, 50, 65, 80, 100]
    num_generations_list = [25, 50, 75, 100, 150]

    # crossing algo, total 360 combinations
    n_population_list = [20, 35, 50, 65, 80, 100]
    n_generations_list = [25, 50, 75, 100, 150]
    n_crossover_list = [0.7, 0.9, 1.0]
    n_mutation_list = [0.01, 0.05, 0.1, 0.2]

    for transmitters_filename in transmitters_filenames:
        transmitters, radius = read_transmitters(sys.argv[2])
        for num_bees in num_bees_list:
            for num_generations in num_generations_list:
                vis = MockVisualizer() # faster testing, if you want to visualize, you need to run main.py explicitly

                start = timer()
                bees = run_bee(transmitters, radius, num_bees, num_generations, vis)
                elapsed = timer() - start
        for n_population in n_population_list:
            for n_generations in n_generations_list:
                for n_crossover in n_crossover_list:
                    for n_mutation in n_mutation_list:
                        vis = MockVisualizer() # faster testing, if you want to visualize, you need to run main.py explicitly

                        start = timer()
                        crossover = run_crossing(transmitters, radius, n_population, n_generations, n_crossover, n_mutation)
                        elapsed = timer() - start
