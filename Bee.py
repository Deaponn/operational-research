from GeneticAlgorithm import GeneticAlgorithm
import numpy as np
from Visualizer import Visualizer


class BeeAlgorithm(GeneticAlgorithm):

    def __init__(self, transmitters, radius, num_bees):
        super().__init__(transmitters, radius)
        self.num_bees = num_bees
        self.population = self.generate_population(num_bees)
        self.scores = self.calculate_scores(self.population)
        self.trials = np.zeros(num_bees)
        self.limit = 10
        self.best_score = np.max(self.scores)
        self.best_score_list = [self.best_score]
        self.best_population = self.population[np.argmax(self.scores)].copy()
        self.best_population_idx = 0
        print(f"Initial best score: {self.best_score:.4f}")

    def local_search(self, solution):
        new_solution = solution.copy()
        idx = np.random.randint(len(solution))
        new_solution[idx] = not new_solution[idx]
        return new_solution

    def select_onlooker_source(self):
        clipped_scores = np.clip(self.scores, a_min=0, a_max=None)
        fitness_sum = np.sum(clipped_scores)

        if fitness_sum == 0:
            return np.random.randint(self.num_bees)

        probs = clipped_scores / fitness_sum
        return np.random.choice(range(self.num_bees), p=probs)

    def run_iteration(self, vis, num_generations=100):
        for gen in range(num_generations):
            for i in range(self.num_bees):
                new_sol = self.local_search(self.population[i])
                new_score = self.calculate_score(new_sol)
                if new_score > self.scores[i]:
                    self.population[i] = new_sol
                    self.scores[i] = new_score
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1

            for _ in range(self.num_bees):
                i = self.select_onlooker_source()
                new_sol = self.local_search(self.population[i])
                new_score = self.calculate_score(new_sol)
                if new_score > self.scores[i]:
                    self.population[i] = new_sol
                    self.scores[i] = new_score
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1

            for i in range(self.num_bees):
                if self.trials[i] > self.limit:
                    self.population[i] = self.generate_population(1)[0]
                    self.scores[i] = self.calculate_score(self.population[i])
                    self.trials[i] = 0

            best_idx = np.argmax(self.scores)
            self.best_score_list.append(self.scores[best_idx])
            vis.add_frame(self.population[best_idx], self.scores[best_idx], self.best_score_list)
            print(f"[Iter {gen}] Best score in iteration: {self.scores[best_idx]:.4f}")

            if self.scores[best_idx] > self.best_score:
                self.best_score = self.scores[best_idx]
                self.best_population = np.array(self.population[best_idx])
                self.best_population_idx = gen
                print(f"[Iter {gen}] NEW Global Best Score: {self.best_score:.4f}")

        print(f"\nFinal Global Best Score: {self.best_score:.4f}")
        return self.population[best_idx], self.scores[best_idx], self.best_population_idx, self.best_score_list
