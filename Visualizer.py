import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import HTMLWriter
from matplotlib.patches import Circle


class Visualizer:
    def __init__(self, transmitters, radius, max_score):
        self.transmitters = transmitters
        self.radius = radius
        self.max_score = max_score

        self.fig, self.ax = plt.subplots()

        self.writer = HTMLWriter(fps=0.5)
        self.writer.setup(self.fig, 'visualization.html', dpi=100)

        self.fig.suptitle(f"Current/Max score: N/A / {self.max_score}")
        dots, circles = self._draw_transmitters(
            self.transmitters[:, 0], self.transmitters[:, 1], "grey")

    def add_frame(self, active_transmitters, score):
        self.fig.suptitle(f"Current/Max score: {score} / {self.max_score}")
        dots, circles = self._draw_transmitters(
            self.transmitters[:, 0][active_transmitters], self.transmitters[:, 1][active_transmitters], "b")
        self.writer.grab_frame()
        self._erase(dots, circles)

    def save_animation(self):
        self.writer.finish()

    def _draw_transmitters(self, xs, ys, color):
        dots = self.ax.plot(xs, ys, "o", color=color, markersize=10)

        circles = []
        for x, y in zip(xs, ys):
            circle = patches.Circle(
                (x, y), self.radius, color=color, fill=False, linewidth=4)
            self.ax.add_patch(circle)
            circles.append(circle)

        return dots, circles

    def _erase(self, dots, circles):
        for x in dots:
            x.remove()
        for x in circles:
            x.remove()


def plot_transmitters(transmitters, bitmask, radius, title="Transmitters", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    active_indices = np.where(bitmask)[0]
    inactive_indices = np.where(~bitmask)[0]
    for i in inactive_indices:
        x, y = transmitters[i]
        ax.plot(x, y, 'o', color='gray')
        circle = Circle((x, y), radius, color='red', alpha=0.05)
        ax.add_patch(circle)

    for i in active_indices:
        x, y = transmitters[i]
        ax.plot(x, y, 'o', color='black')
        circle = Circle((x, y), radius, color='green', alpha=0.2)
        ax.add_patch(circle)

    for i in active_indices:
        for j in active_indices:
            if i < j:
                dist = np.linalg.norm(transmitters[i] - transmitters[j])
                if dist <= 2 * radius:
                    xi, yi = transmitters[i]
                    xj, yj = transmitters[j]
                    ax.plot([xi, xj], [yi, yj], color='blue',
                            linewidth=1, alpha=0.4)

    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    x_vals, y_vals = transmitters[:, 0], transmitters[:, 1]
    padding = radius + 5
    ax.set_xlim(x_vals.min() - padding, x_vals.max() + padding)
    ax.set_ylim(y_vals.min() - padding, y_vals.max() + padding)

    num_active = len(active_indices)
    ax.text(0.7, 0.8, f"Active: {num_active}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
