import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import itertools
from directed_percolation import dp_step
from pc_percolation import pc_step

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, L, N, initial, p, r, mode):
        self.L = L
        self.N = N
        self.initial = initial
        self.p = p
        self.r = r
        self.mode = mode
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                          init_func=self.setup_plot, blit=True)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlabel("Space", size=14)
        self.ax.set_ylabel("Time", size=14)
        if self.mode == "dp":
            self.ax.set_title(f"Directed Percolation for system size {L} and time {N}", size=18)
        elif self.mode == "pc":
            self.ax.set_title(f"Parity Conserving Percolation for system size {L} and time {N}", size=18)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, c=c, s=s)
        # self.ax.axis([-10, 10, -10, 10])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Animate directed percolation"""
        xy = np.array(list(itertools.product(np.arange(self.L), self.N - np.arange(self.N))))
        s = np.ones(self.L * self.N) * 0.08
        true_c = np.zeros((self.N, self.L))
        true_c[0] = self.initial
        for i in range(self.N-1):
            if self.mode == "dp":
                true_c[i+1] = dp_step(true_c[i], self.p)
            elif self.mode == "pc":
                true_c[i+1] = pc_step(true_c[i], self.p, self.r)
        c = true_c.T.reshape(self.N * self.L)
        while True:
            if self.mode == "dp":
                next_row = dp_step(true_c[-1], self.p)
            elif self.mode == "pc":
                next_row = pc_step(true_c[-1], self.p, self.r)
            for i in range(self.N-1):
                true_c[i] = true_c[i+1]
            true_c[-1] = next_row
            c = true_c.T.reshape(self.N * self.L)
            yield np.c_[xy[:,0], xy[:,1], s, c]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_sizes(300 * abs(data[:, 2])**1.5)
        # Set colors..
        self.scat.set_array(data[:, 3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,


if __name__ == '__main__':
    L = 100
    N = 100
    initial = np.zeros(L)
    initial[int(L/2)] = 1
    p = 0.6447001
    r = 0.5
    mode = "pc"  # or "pc"
    a = AnimatedScatter(L, N, initial, p, r, mode)
    plt.show()