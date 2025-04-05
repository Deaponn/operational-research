import numpy as np
from scipy.spatial import KDTree


class GeneticAlgorithm:
    def __init__(self, transmitters, radius, c1=1.0, c2=5.0):
        self.transmitters = np.array(transmitters)  # Nx2 array of coordinates
        self.radius = radius
        self.c1 = c1
        self.c2 = c2
        self.max_score = self.calculate_score(self.get_all())

    def generate_population(self, num_members):
        return np.random.rand(num_members, len(self.transmitters)) < 0.5

    def calculate_score(self, bitmask):
        active_indices = np.where(bitmask)[0]
        if len(active_indices) == 0:
            return 0

        active_transmitters = self.transmitters[active_indices]

        if not self.is_connected(active_transmitters):
            return 0

        area = self.approximate_coverage_area(active_transmitters)
        num_active = len(active_indices)
        return self.c1 * num_active + self.c2 * area

    def calculate_scores(self, population):
        return np.array([self.calculate_score(ind) for ind in population])

    def is_connected(self, points):
        if len(points) == 1:
            return True

        tree = KDTree(points)
        adjacency = tree.query_ball_tree(tree, r=2 * self.radius)
        visited = set()

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    dfs(neighbor)

        dfs(0)
        return len(visited) == len(points)

    def approximate_coverage_area(self, points, resolution=100):
        if len(points) == 0:
            return 0

        x_min, y_min = np.min(points, axis=0) - self.radius
        x_max, y_max = np.max(points, axis=0) + self.radius

        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T

        coverage = np.zeros(len(grid_points), dtype=bool)
        for pt in points:
            coverage |= np.sum((grid_points - pt)**2, axis=1) <= self.radius**2

        cell_area = ((x_max - x_min) / resolution) * \
            ((y_max - y_min) / resolution)
        return np.sum(coverage) * cell_area

    def get_all(self):
        return np.array([True] * len(self.transmitters))

    def get_max_score(self):
        return self.max_score
