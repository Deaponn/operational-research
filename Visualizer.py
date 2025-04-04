import matplotlib.pyplot as plt
from matplotlib.animation import HTMLWriter

class Visualizer:
    def __init__(self, transmitters, radius, max_score):
        self.transmitters = transmitters
        self.max_score = max_score

        self.fig, self.ax = plt.subplots()

        self.writer = HTMLWriter()
        self.writer.setup(self.fig, 'visualization.html', dpi=100)

    def add_frame(self, active_transmitters, score):
        self.fig.suptitle(f"Current score: {score}")
        self.ax.plot(self.transmitters[:][0][active_transmitters], self.transmitters[:][1][active_transmitters])
        self.writer.grab_frame()

    def save_animation(self):
        self.writer.finish()
