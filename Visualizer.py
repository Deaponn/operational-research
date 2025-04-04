import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import HTMLWriter

class Visualizer:
    def __init__(self, transmitters, radius, max_score):
        self.transmitters = transmitters
        self.radius = radius
        self.max_score = max_score

        self.fig, self.ax = plt.subplots()

        self.writer = HTMLWriter(fps=0.5)
        self.writer.setup(self.fig, 'visualization.html', dpi=100)

        self.fig.suptitle(f"Current/Max score: N/A / {self.max_score}")
        dots, circles = self._draw_transmitters(self.transmitters[:, 0], self.transmitters[:, 1], "grey")

    def add_frame(self, active_transmitters, score):
        self.fig.suptitle(f"Current/Max score: {score} / {self.max_score}")
        dots, circles = self._draw_transmitters(self.transmitters[:, 0][active_transmitters], self.transmitters[:, 1][active_transmitters], "b")
        self.writer.grab_frame()
        self._erase(dots, circles)

    def save_animation(self):
        self.writer.finish()

    def _draw_transmitters(self, xs, ys, color):
        dots = self.ax.plot(xs, ys, "o", color=color, markersize=10)

        circles = []
        for x, y in zip(xs, ys):
            circle = patches.Circle((x, y), self.radius, color=color, fill=False, linewidth=4)
            self.ax.add_patch(circle)
            circles.append(circle)

        return dots, circles

    def _erase(self, dots, circles):
        for x in dots: x.remove()
        for x in circles: x.remove()
