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

    def get_best_solution(self):
        best_idx = np.argmax(self.scores)
        return self.population[best_idx], self.scores[best_idx]


np.random.seed(123)
transmitters = np.random.rand(50, 2) * 100
radius = 10
num_bees = 30
print(transmitters)

vis = Visualizer(transmitters, radius, 0)

bee_algo = BeeAlgorithm(transmitters=transmitters,
                        radius=radius, num_bees=num_bees)
initial_mask = np.ones(len(transmitters), dtype=bool)
plot_transmitters(transmitters, initial_mask, radius,
                  title="Initial Configuration", save_path="starting.png")


for _ in range(1000):
    bee_algo.run_iteration(vis)

best_mask, best_score = bee_algo.get_best_solution()
plot_transmitters(transmitters, best_mask, radius,
                  title=f"Final Solution (Score: {best_score:.2f})",
                  save_path="final.png")

vis.save_animation()

print("Best:", best_mask, "Score:", best_score)
