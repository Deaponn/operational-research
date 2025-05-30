import numpy as np
from GeneticAlgorithm import GeneticAlgorithm
from Visualizer import Visualizer


STAGNATION_THRESHOLD = 10


class Crossing(GeneticAlgorithm):
    def __init__(self, transmitters, radius, n_population, n_generations, p_crossover, p_mutation, c1=-1.0, c2=5.0, c3=1000.0):
        self.n_population = n_population
        self.n_generations = n_generations
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.best_score = -np.inf
        self.best_generation = None
        self.best_generation_idx = 0
        self.best_score_list = []
        self.iterations_since_no_improvement = 0
        self.last_best_score = 0
        self.iterations_ran = 0
        super().__init__(transmitters, radius, c1=c1, c2=c2, c3=c3)

    def select(self, population, scores, n_tournament_size=5):
        candidates = np.random.choice(range(self.n_population), n_tournament_size, replace=False)
        best = candidates[np.argmax(scores[candidates])]
        return population[best]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.p_crossover:
            point = np.random.randint(1, len(parent1))
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
        else:
            child1 = np.copy(parent1)
            child2 = np.copy(parent2)
        return [child1, child2]

    def mutation(self, child):
        for i in range(len(child)):
            if np.random.rand() < self.p_mutation:
                child[i] = not child[i]

    def run_iteration(self, vis):
        population = self.generate_population(self.n_population)
        scores = np.array([self.calculate_score(ind) for ind in population])
        best_idx = np.argmax(scores)
        self.best_score = scores[best_idx]
        self.best_score_list.append(self.best_score)
        self.best_generation = population[best_idx].copy()
        vis.add_frame(self.best_generation, self.best_score, self.best_score_list)
        print(f"Initial best score: {self.best_score:.4f}")

        for generation in range(self.n_generations):
            scores = np.array([self.calculate_score(ind) for ind in population])
            best_idx = np.argmax(scores)
            gen_best_score = scores[best_idx]
            self.best_score_list.append(gen_best_score)
            vis.add_frame(population[best_idx], gen_best_score, self.best_score_list)

            if gen_best_score > self.best_score:
                self.last_best_score = self.best_score
                self.best_score = gen_best_score
                self.best_generation = population[best_idx].copy()
                self.best_generation_idx = generation
            #     print(f"[Gen {generation}] NEW GLOBAL BEST: {self.best_score:.4f}")
            # else:
            #     print(f"[Gen {generation}] Best score in generation: {gen_best_score:.4f}")

            parents = [self.select(population, scores) for _ in range(self.n_population)]
            children = []

            for i in range(0, self.n_population, 2):
                p1 = parents[i]
                p2 = parents[i+1 if i+1 < self.n_population else 0]
                offspring = self.crossover(p1, p2)
                for child in offspring:
                    self.mutation(child)
                    children.append(child)

            children[np.random.randint(len(children))] = self.best_generation.copy()
            population = children

            if self.last_best_score == self.best_score:
                self.iterations_since_no_improvement += 1
            else:
                self.last_best_score = self.best_score
                self.iterations_since_no_improvement = 0

            if self.iterations_since_no_improvement > STAGNATION_THRESHOLD:
                self.iterations_ran = generation
                print(f"No improvement for {STAGNATION_THRESHOLD} generations detected. Stopping early.")
                print(f"Final best score: {self.best_score:.4f}\n")
                # print(f"Best generation bitmask: {self.best_generation}")
                return

        self.iterations_ran = self.n_generations
        print(f"Final best score: {self.best_score:.4f}\n")
        # print(f"Best generation bitmask: {self.best_generation}")

    def get_results(self):
        return self.best_generation, self.best_score, self.best_generation_idx, self.best_score_list
