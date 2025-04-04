import numpy as np

new_transmitters = 3

class GeneticAlgorithm:
    def __init__(self, transmitters, radius):
        self.transmitters = transmitters
        self.radius = radius
        self.max_score = self.calculate_score(self.get_all())

    # returns <num_members> randomly generated bitmasks which are describing which transmitters are currently active
    # TODO: implement random generation
    def generate_population(self, num_members):
        return np.array([
            [False] * len(self.transmitters) for _ in range(num_members)
        ])

    # returns <num_members> variations based on <source_member> bitmask
    # TODO: implement something smarter probably?
    def breed_population(self, source_member, num_members):
        new_population = [source_member.copy() for _ in range(num_members)]

        # randomly turn some transmitters on
        for x in new_population:
            x[np.random.choice(len(self.transmitters), new_transmitters, replace=False)] = True

        return new_population

    def calculate_scores(self, bitmasks):
        return np.array([self.calculate_score(bitmask) for bitmask in bitmasks])

    # calculates score of a given bitmask (covered area, number of connected transmitters etc)
    # TODO: implement, currently counts the count of transmitters which are turned on
    def calculate_score(self, bitmask):
        return np.sum(bitmask) * 10

    def get_all(self):
        return [True] * len(self.transmitters)

    def get_max_score(self):
        return self.max_score