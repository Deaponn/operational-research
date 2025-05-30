import sys
import os
from timeit import default_timer as timer
from main import run_bee, run_crossing
from Visualizer import MockVisualizer, Visualizer
from gen_transmitters import read_transmitters
import numpy as np
import concurrent.futures


def calculate_scores_difference(scores):
    diffs = np.diff(scores)
    bigger_count = np.sum(diffs > 0)
    smaller_count = np.sum(diffs < 0)
    total_comparisons = diffs.size
    bigger_percentage = (bigger_count / total_comparisons) * 100 if total_comparisons > 0 else 0
    smaller_percentage = (smaller_count / total_comparisons) * 100 if total_comparisons > 0 else 0

    return f"{bigger_count},{bigger_percentage},{smaller_count},{smaller_percentage}"


def benchmark_bee_worker(transmitters_data, radius_data, num_bees_val, num_generations_val):
    np.random.seed(os.getpid())

    current_vis = MockVisualizer()

    start_time = timer()
    bees = run_bee(transmitters_data, radius_data, num_bees_val, num_generations_val, current_vis)
    elapsed_time = timer() - start_time

    score_diff_str = calculate_scores_difference(bees.best_score_list)

    return (
        f"{num_bees_val},"
        f"{bees.iterations_ran},"
        f"{bees.best_score},"
        f"{bees.score_calculations},"
        f"{score_diff_str},"
        f"{elapsed_time}"
    )


def benchmark_crossing_worker(transmitters_data, radius_data, n_population, n_generations, n_crossover, n_mutation):
    np.random.seed(os.getpid())

    current_vis = MockVisualizer()

    start_time = timer()
    crossover = run_crossing(transmitters_data, radius_data, n_population, n_generations, n_crossover, n_mutation, current_vis)
    elapsed_time = timer() - start_time

    score_diff_str = calculate_scores_difference(crossover.best_score_list)

    return (
        f"{n_population},"
        f"{n_crossover},"
        f"{n_mutation},"
        f"{crossover.iterations_ran},"
        f"{crossover.best_score},"
        f"{crossover.score_calculations},"
        f"{score_diff_str},"
        f"{elapsed_time}"
    )


if __name__ == "__main__":
    transmitters_names = ["small", "default", "big"]

    repetitions = 16

    # bee algo, total 6 * 16 combinations
    num_bees_list = [20, 35, 50, 65, 80, 100]
    num_generations_list = [150] # with early stopping this is essentially just the max_iterations setting 

    # crossing algo, total 72 * 16 combinations
    n_population_list = [20, 35, 50, 65, 80, 100]
    n_generations_list = [150] # with early stopping this is essentially just the max_iterations setting 
    n_crossover_list = [0.7, 0.9, 1.0]
    n_mutation_list = [0.01, 0.05, 0.1, 0.2]

    if not os.path.exists("benchmark"):
        os.makedirs("benchmark")

    for transmitters_name in transmitters_names:
        transmitters, radius = read_transmitters(f"transmitters/{transmitters_name}.txt")
        with open(f"benchmark/bee_{transmitters_name}.csv", "w") as f:
            f.write("num_bees,num_generations,best_score,score_func_calls,num_improvements,percent_improvements,num_declines,percent_declines,time\n")
            for num_bees in num_bees_list:
                for num_generations in num_generations_list:
                    filename = f"benchmark/bee_{transmitters_name}_{num_bees}bees"
                    # visualize once, but dont count stats to not skew the results
                    vis = Visualizer(transmitters, radius, filename)
                    best_member, best_score, best_generation, best_score_list = run_bee(
                        transmitters,
                        radius,
                        num_bees,
                        num_generations,
                        vis).get_results()
                    vis.add_frame(best_member, best_score, best_score_list, last_frame=True, best_iteration_idx=best_generation)
                    vis.save_animation()

                    # run in parallel
                    print(f"Starting {repetitions} parallel benchmark runs for {num_bees} bees...")
                    benchmark_results_lines = []

                    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
                        futures = []
                        for i in range(repetitions):
                            future = executor.submit(
                                benchmark_bee_worker,
                                transmitters,
                                radius,
                                num_bees,
                                num_generations
                            )
                            futures.append(future)

                        for i, future in enumerate(concurrent.futures.as_completed(futures)):
                            try:
                                result_line = future.result()
                                benchmark_results_lines.append(result_line)
                                print(f"Completed benchmark run {i+1}/{repetitions} for {num_bees} bees, {num_generations} generations.")
                            except Exception as e:
                                print(f"Benchmark run {i+1}/{repetitions} for {num_bees} bees, {num_generations} generations failed: {e}")

                    for line_data in benchmark_results_lines:
                        f.write(line_data + "\n")
                    if benchmark_results_lines:
                        f.flush()
                    print(f"Finished all benchmark runs for {num_bees} bees, {num_generations} generations.\n\n\n")

        with open(f"benchmark/crossing_{transmitters_name}.csv", "w") as f:
            f.write("n_population,n_crossover,n_mutation,n_generations,best_score,score_func_calls,num_improvements,percent_improvements,num_declines,percent_declines,time\n")
            for n_population in n_population_list:
                for n_generations in n_generations_list:
                    for n_crossover in n_crossover_list:
                        for n_mutation in n_mutation_list:
                            filename = f"benchmark/crossing_{transmitters_name}_{n_population}pop_{n_generations}gens_{n_crossover}cross_{n_mutation}mut"
                            # visualize once, but dont count stats to not skew the results
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

                            # run in parallel
                            print(f"Starting {repetitions} parallel benchmark runs for crossing {n_population} specimen...")
                            benchmark_results_lines = []

                            with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
                                futures = []
                                for i in range(repetitions):
                                    future = executor.submit(
                                        benchmark_crossing_worker,
                                        transmitters,
                                        radius,
                                        n_population,
                                        n_generations,
                                        n_crossover,
                                        n_mutation
                                    )
                                    futures.append(future)

                                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                                    try:
                                        result_line = future.result()
                                        benchmark_results_lines.append(result_line)
                                        print(f"Completed benchmark run {i+1}/{repetitions} for crossing {n_population} specimen, {num_generations} generations.")
                                    except Exception as e:
                                        print(f"Benchmark run {i+1}/{repetitions} for crossing {n_population} specimen, {num_generations} generations failed: {e}")

                            for line_data in benchmark_results_lines:
                                f.write(line_data + "\n")
                            if benchmark_results_lines:
                                f.flush()
                            print(f"Finished all benchmark runs for crossing {n_population} specimen, {num_generations} generations.\n\n\n")
