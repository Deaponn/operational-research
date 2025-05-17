from GeneticAlgorithm import GeneticAlgorithm
import numpy as np
from Visualizer import Visualizer, plot_transmitters


class BeeAlgorithm(GeneticAlgorithm):

    def __init__(self, transmitters, radius, num_bees):
        super().__init__(transmitters, radius)
        self.num_bees = num_bees
        self.population = self.generate_population(num_bees)
        self.scores = self.calculate_scores(self.population)
        self.trials = np.zeros(num_bees)
        self.limit = 10
        self.best_score = 0
        self.best_population = []

    def local_search(self, solution):
        new_solution = solution.copy()
        idx = np.random.randint(len(solution))
        new_solution[idx] = not new_solution[idx]
        return new_solution

    def select_onlooker_source(self):
        fitness_sum = np.sum(self.scores)
        if fitness_sum == 0:
            return np.random.randint(self.num_bees)
        probs = self.scores / fitness_sum
        return np.random.choice(range(self.num_bees), p=probs)

    def run_iteration(self, vis):

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

        vis.add_frame(self.population[best_idx], self.scores[best_idx])

        print(f"Current Best Score: {self.scores[best_idx]:.2f}")

        if self.scores[best_idx] > self.best_score:
            self.best_score = self.scores[best_idx]
            self.best_population = np.array(self.population[best_idx])


np.random.seed(123)
transmitters = np.random.rand(50, 2) * 100
radius = 10
num_bees = 1000
print(transmitters)

vis = Visualizer(transmitters, radius, 0, "bee")

bee_algo = BeeAlgorithm(transmitters=transmitters,
                        radius=radius, num_bees=num_bees)
initial_mask = np.ones(len(transmitters), dtype=bool)
plot_transmitters(transmitters, initial_mask, radius,
                  title="Initial Configuration", save_path="starting.png")

num_generations = 30
for i in range(num_generations):
    if i % (num_generations // 10) == 0: print(f"{i / num_generations * 100}%")
    bee_algo.run_iteration(vis)

vis.add_frame(bee_algo.best_population, bee_algo.best_score)
vis.save_animation()

print("Best:", bee_algo.best_population, "Score:", bee_algo.best_score)
