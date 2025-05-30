import sys
import os
from timeit import default_timer as timer
from main import run_bee, run_crossing
from Visualizer import MockVisualizer, Visualizer
from gen_transmitters import read_transmitters
import numpy as np


def calculate_scores_difference(scores):
    diffs = np.diff(scores)
    bigger_count = np.sum(diffs > 0)
    smaller_count = np.sum(diffs < 0)
    total_comparisons = diffs.size
    bigger_percentage = (bigger_count / total_comparisons) * 100 if total_comparisons > 0 else 0
    smaller_percentage = (smaller_count / total_comparisons) * 100 if total_comparisons > 0 else 0

    return f"{bigger_count},{bigger_percentage},{smaller_count},{smaller_percentage}"


if __name__ == "__main__":
    transmitters_filenames = ["transmitters/small.txt", "transmitters/default.txt"]

    repetitions = 3

    # bee algo, total 30 combinations
    num_bees_list = [20, 35]
    num_generations_list = [25, 50]
    # num_bees_list = [20, 35, 50, 65, 80, 100]
    # num_generations_list = [25, 50, 75, 100, 150]

    # crossing algo, total 360 combinations
    n_population_list = [20, 35, 50, 65, 80, 100]
    n_generations_list = [25, 50, 75, 100, 150]
    n_crossover_list = [0.7, 0.9, 1.0]
    n_mutation_list = [0.01, 0.05, 0.1, 0.2]

    if not os.path.exists("benchmark"):
        os.makedirs("benchmark")

    for transmitters_filename in transmitters_filenames:
        transmitters, radius = read_transmitters(transmitters_filename)
        with open(f"benchmark/bee.csv", "w") as f:
            f.write("num_bees,num_generations,best_score,score_func_calls,num_improvements,percent_improvements,num_declines,percent_declines,time\n")
            for num_bees in num_bees_list:
                for num_generations in num_generations_list:
                    filename = f"benchmark/bee_{num_bees}bees_{num_generations}gens"
                    for rep in range(repetitions):
                        vis = MockVisualizer() # faster testing, if you want to visualize, you need to run main.py explicitly

                        if rep == 0: # visualize once, but dont count stats to not skew the results
                            vis = Visualizer(transmitters, radius, filename)
                            best_member, best_score, best_generation, best_score_list = run_bee(
                                transmitters,
                                radius,
                                num_bees,
                                num_generations,
                                vis).get_results()
                            vis.add_frame(best_member, best_score, best_score_list, last_frame=True, best_iteration_idx=best_generation)
                            vis.save_animation()
                            continue

                        start = timer()
                        bees = run_bee(transmitters, radius, num_bees, num_generations, vis)
                        elapsed = timer() - start

                        score_diffs = calculate_scores_difference(bees.best_score_list)

                        f.write(f"{num_bees},{num_generations},{bees.best_score},{bees.score_calculations},{score_diffs},{elapsed}\n")
        
        with open(f"benchmark/crossing.csv", "w") as f:
            f.write("n_population,n_generations,n_crossover,n_mutation,best_score,score_func_calls,num_improvements,percent_improvements,num_declines,percent_declines,time\n")
            for n_population in n_population_list:
                for n_generations in n_generations_list:
                    for n_crossover in n_crossover_list:
                        for n_mutation in n_mutation_list:
                            filename = f"benchmark/crossing_{n_population}pop_{n_generations}gens_{n_crossover}cross_{n_mutation}mut"
                            for rep in range(repetitions):
                                vis = MockVisualizer() # faster testing, if you want to visualize, you need to run main.py explicitly

                                if rep == 0: # visualize once, but dont count stats to not skew the results
                                    vis = Visualizer(transmitters, radius, filename)
                                    best_member, best_score, best_generation, best_score_list = run_crossing(
                                        transmitters,
                                        radius,
                                        n_population,
                                        n_generations,
                                        n_crossover,
                                        n_mutation,
                                        vis).get_results()
                                    vis.add_frame(best_member, best_score, best_score_list, last_frame=True, best_iteration_idx=best_generation)
                                    vis.save_animation()
                                    continue

                                start = timer()
                                crossover = run_crossing(transmitters, radius, n_population, n_generations, n_crossover, n_mutation, vis)
                                elapsed = timer() - start

                                score_diffs = calculate_scores_difference(crossover.best_score_list)

                                f.write(f"{n_population},{n_generations},{n_crossover},{n_mutation},{crossover.best_score},{crossover.score_calculations},{score_diffs},{elapsed}\n")
