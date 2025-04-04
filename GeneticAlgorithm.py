class GeneticAlgorithm:
    def __init__(self, transmitters, radius):
        self.transmitters = transmitters
        self.radius = radius

    # returns <num_members> randomly generated bitmasks which are describing which transmitters are currently active
    # TODO: implement random generation
    def generate_population(num_members):
        return np.array([
            [1] * len(self.transmitters) for _ in range(num_members)
        ])

    # returns <num_members> variations based on <source_member> bitmask
    # TODO: implement
    def breed_population(source_member, num_members):
        return np.array([
            [1] * len(self.transmitters) for _ in range(num_members)
        ])

    # calculates score of a given bitmask (covered area, number of connected transmitters etc)
    # TODO: implement
    def calculate_score(self, bitmask):
        return 100