import numpy as np
from GeneticAlgorithm import GeneticAlgorithm
from Visualizer import Visualizer, plot_transmitters

class Crossing(GeneticAlgorithm):

    def __init__(self, transmitters, radius, n_population, n_generations, n_crossover, n_mutation):
        self.n_population = n_population
        self.n_generations = n_generations
        self.n_crossover = n_crossover
        self.n_mutation = n_mutation
        super().__init__(transmitters, radius,c1=1, c2=1)


    def select(self, population, scores, n_tournament_size=5):
        # Random parent
        best_parent_idx = np.random.choice(range(self.n_population), 1)[0]
        best_score = scores[best_parent_idx]

        # Tournament selection
        for i in np.random.choice(range(self.n_population), n_tournament_size, replace=False):
            if scores[i] < best_score:
                best_parent_idx = i
                best_score = scores[i]
        
        return population[best_parent_idx]


    def crossover(self, parent1, parent2):
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        if np.random.rand() < self.n_crossover:
            point = np.random.randint(1, len(parent1)-2)

            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
        
        return [child1, child2]
    

    def mutation(self, child):
        for i, bit in enumerate(child):
            if np.random.rand() < self.n_mutation:
                child[i] = not bit


    def calculate_score(self, bitmask):
        active_indices = np.where(bitmask)[0]
        active_transmitters = self.transmitters[active_indices]

        return self.c1 * len(active_transmitters) - self.c2 * self.approximate_coverage_area(active_transmitters)


    def run_iteration(self, vis):
        population = self.generate_population(n_population)
        
        # Random score
        random_member = np.random.randint(0, self.n_population)
        best_score = self.calculate_scores(population)[random_member]
        best_member = population[random_member]

        for generation in range(self.n_generations):
            print(f"Running generation #{generation}")

            # Evaluation
            scores = np.array([self.calculate_score(member) for member in population])

            # Finding best solution
            for i, score in enumerate(scores):
                if score < best_score:
                    best_score = score
                    best_member = population[i]
                    print(f"New best score {score:.2f}")
                    vis.add_frame(best_member, best_score)

            # Parents selection
            parents = [self.select(population, scores) for _ in range(self.n_population)]

            # Creating next generation
            children = []
            for i in range(0, self.n_population, 2):
                for child in self.crossover(parents[i], parents[i+1]):
                    self.mutation(child)
                    children.append(child)

            # Replacing population
            population = children
        
        print(f"Best score {best_score:.2f}")
        return best_member, best_score

np.random.seed(123)

n_population = 20 # Population size, should be even number
n_transmitters = 50 # Number of transmitters
n_generations = 20 # Number of generations
n_crossover = 1.0 # A chance for crossover
n_mutation = 0.05 # A chance for mutation

transmitters = np.random.rand(n_transmitters, 2) * 100
radius = 10 #np.random.randint(1, 10)

crossing = Crossing(transmitters, radius, n_population, n_generations, n_crossover, n_mutation)

mask = np.ones(n_transmitters, dtype=bool)

plot_transmitters(transmitters, mask, radius, title="Initial problem", save_path="out/crossing_initial.png")

vis = Visualizer(transmitters, radius, 0)

best_member, best_score = crossing.run_iteration(vis)

vis.save_animation()

plot_transmitters(transmitters, best_member, radius, title="Solution", save_path="out/crossing_solution.png")


