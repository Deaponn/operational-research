import numpy as np
from utils import normalize
from Visualizer import Visualizer
from GeneticAlgorithm import GeneticAlgorithm

if __name__ == "__main__":
    np.random.seed(420)

    transmitters = np.array([
        [0, 1], [2, 3], [-1, 3], [-2, 1.5], [3, 1], [0.3, 0.2], [1, 1], [2, 2], [1.5, 3.5], [3, 0.5]
    ])
    radius = 2

    num_generations = 10
    num_populations = 5

    normalized_transmitters, normalized_radius = normalize(transmitters, radius)

    print("Normalized transmitters:\n", normalized_transmitters, f"\nNormalized radius: {normalized_radius}")

    alg = GeneticAlgorithm(normalized_transmitters, normalized_radius)
    vis = Visualizer(normalized_transmitters, normalized_radius, alg.get_max_score())

    vis.add_frame(alg.get_all(), alg.get_max_score())

    # use binary mask [True, True, True, ...] to include all transmitters
    max_score = alg.calculate_score(alg.get_all())

    # main loop of optimisation
    # maybe it should be moved to GenericAlgorithm.py? but access to Visualizer is handy aswell
    populations = alg.generate_population(num_populations)
    scores = alg.calculate_scores(populations)
    max_idx = np.argmax(scores)
    for _ in range(num_generations):
        print("Current scores:", scores)
        vis.add_frame(populations[max_idx], scores[max_idx])
        populations = alg.breed_population(populations[max_idx], num_populations)
        scores = alg.calculate_scores(populations)
        max_idx = np.argmax(scores)
    vis.save_animation()
