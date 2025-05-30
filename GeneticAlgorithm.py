import numpy as np
from scipy.spatial import KDTree

class GeneticAlgorithm:
    def __init__(self, transmitters, radius, c1=-1.0, c2=5.0, c3=1000.0, resolution=100):
        self.transmitters = np.array(transmitters)
        self.radius = radius
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.score_calculations = 0
        self.grid_points = None
        self.resolution = resolution
        self.max_score = self.calculate_score(self.get_all())

    def generate_population(self, num_members):
        return np.random.rand(num_members, len(self.transmitters)) < 0.5

    def calculate_score(self, bitmask):
        self.score_calculations += 1
        if self.grid_points is None: self._generate_grid_points()
        active_indices = np.where(bitmask)[0]
        active_transmitters = self.transmitters[active_indices]
        if len(active_transmitters) == 0:
            return -np.inf
        base_score = self.c1 * (len(active_transmitters) / len(self.transmitters))
        coverage_score = self.c2 * (self.approximate_coverage_area(active_transmitters) / self.approximate_coverage_area(self.transmitters))
        connectivity_score = 0 if self.is_connected(active_transmitters) else -self.c3
        return base_score + coverage_score + connectivity_score

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
        if self.grid_points is None: self._generate_grid_points()
        if len(points) == 0:
            return 0
        coverage = np.zeros(len(self.grid_points), dtype=bool)
        for pt in points:
            coverage |= np.sum((self.grid_points - pt)**2, axis=1) <= self.radius**2
        return np.sum(coverage) * self.cell_area

    def _generate_grid_points(self):
        x_min, y_min = np.min(self.transmitters, axis=0) - self.radius
        x_max, y_max = np.max(self.transmitters, axis=0) + self.radius
        xs = np.linspace(x_min, x_max, self.resolution)
        ys = np.linspace(y_min, y_max, self.resolution)
        grid_x, grid_y = np.meshgrid(xs, ys)
        self.grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
        
        self.cell_area = ((x_max - x_min) / self.resolution) * ((y_max - y_min) / self.resolution)

    def get_all(self):
        return np.array([True] * len(self.transmitters))

    def get_max_score(self):
        return self.max_score