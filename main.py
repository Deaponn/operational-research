import numpy as np
from utils import normalize
from Visualizer import Visualizer
from GeneticAlgorithm import GeneticAlgorithm

if __name__ == "__main__":
    transmitters = np.array([
        [0, 1], [2, 3], [-1, 3], [-2, 1.5]
    ])
    radius = 2

    normalized_transmitters, normalized_radius = normalize(transmitters, radius)

    alg = GeneticAlgorithm(normalized_transmitters, normalized_radius)

    # binary mask [1, 1, 1, ...] to include all transmitters
    max_score = alg.calculate_score([1] * len(normalized_transmitters))

    print(transmitters, radius)
    print(normalized_transmitters, f"radius: {normalized_radius}, max score: {max_score}")

    vis = Visualizer(normalized_transmitters, normalized_radius, max_score)
    
    vis.add_frame([1, 0, 0, 1], 70)
    vis.add_frame([1, 0, 1, 1], 60)
    vis.add_frame([1, 1, 0, 0], 90)
    vis.save_animation()
