import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np
from dp_model import dp_step
from pc_model import pc_step


class AnimatedMatrix(object):
    """An animated matrix plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, L, N, initial, p, r, mode):
        self.L = L
        self.N = N
        self.initial = initial
        self.p = p
        self.r = r
        self.mode = mode
        self.stream = self.data_stream()
        self.counter = 0

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                          init_func=self.setup_plot, blit=True)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlabel(f"Space     (p = {p}, r = {r})", size=14)
        self.ax.set_ylabel("Time", size=14)
        if self.mode == "dp":
            self.ax.set_title(f"DP Model for system size {L} and time {N}", size=18)
        elif self.mode == "pc":
            self.ax.set_title(f"PC Model for system size {L} and time {N}", size=18)

    def setup_plot(self):
        """Initial drawing of the matrix plot."""
        data = next(self.stream)
        self.img = self.ax.imshow(data)
        values = [0, 1]
        colors = [self.img.cmap(self.img.norm(value)) for value in values]
        label_dict = {
            0: "Inactive",
            1: "Active",
        }
        patches = [mpatches.Patch(color=colors[i], label=label_dict[i].format(l=values[i]) ) for i in range(len(values))]
        plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., fontsize=12)
        return self.img,

    def data_stream(self):
        """Animate directed percolation"""
        data = np.zeros((self.N, self.L))
        data[0] = self.initial
        while True:
            self.counter += 1
            if self.counter < self.N:
                if self.mode == "dp":
                    next_row = dp_step(data[self.counter-1], self.p)
                elif self.mode == "pc":
                    next_row = pc_step(data[self.counter-1], self.p, self.r)
                data[self.counter] = next_row
            else:
                if self.mode == "dp":
                    next_row = dp_step(data[-1], self.p)
                elif self.mode == "pc":
                    next_row = pc_step(data[-1], self.p, self.r)
                data = np.roll(data, shift=-1, axis=0)
                data[-1] = next_row
            # plt.pause(1)
            yield data

    def update(self, i):
        """Update the matrix plot."""
        data = next(self.stream)

        self.img.set_data(data)

        return self.img,


class DoubleAnimatedMatrix(object):
    """An animated matrix plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, L, N, initial, p, r):
        self.L = L
        self.N = N
        self.initial = initial
        self.p = p
        self.r = r
        self.stream = self.data_stream()
        self.counter = 0

        self.fig, (self.ax1, self.ax2) = plt.subplots(ncols=2, figsize=(17, 8))
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                          init_func=self.setup_plot, blit=True)
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        self.ax1.set_xlabel("Space", size=14)
        self.ax1.set_ylabel("Time", size=14)
        self.ax1.set_title(f"DP Model for system size {L} and time {N}", size=18)
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        self.ax2.set_xlabel("Space", size=14)
        self.ax2.set_ylabel("Time", size=14)
        self.ax2.set_title(f"PC Model for system size {L} and time {N}", size=18)

    def setup_plot(self):
        """Initial drawing of the matrix plot."""
        data = next(self.stream)
        self.img = [self.ax1.imshow(data[0]), self.ax2.imshow(data[1])]
        values = [0, 1]
        colors = [self.img[0].cmap(self.img[0].norm(value)) for value in values]
        label_dict = {
            0: "Inactive",
            1: "Active",
        }
        patches = [mpatches.Patch(color=colors[i], label=label_dict[i].format(l=values[i]) ) for i in range(len(values))]
        plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., fontsize=12)
        return self.img

    def data_stream(self):
        """Animate directed percolation"""
        data = np.zeros((2, self.N, self.L))
        data[0][0] = self.initial
        data[1][0] = self.initial
        while True:
            self.counter += 1
            if self.counter < self.N:
                next_dp_row = dp_step(data[0][self.counter-1], self.p)
                next_pc_row = pc_step(data[1][self.counter-1], self.p, self.r)
                data[0][self.counter] = next_dp_row
                data[1][self.counter] = next_pc_row
            else:
                next_dp_row = dp_step(data[0][-1], self.p)
                next_pc_row = pc_step(data[1][-1], self.p, self.r)
                data[0] = np.roll(data[0], shift=-1, axis=0)
                data[1] = np.roll(data[1], shift=-1, axis=0)
                data[0][-1] = next_dp_row
                data[1][-1] = next_pc_row
            # plt.pause(1)
            yield data

    def update(self, i):
        """Update the matrix plot."""
        data = next(self.stream)

        self.img[0].set_data(data[0])
        self.img[1].set_data(data[1])

        return self.img


class GridAnimatedMatrix(object):
    """An animated matrix plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, L, N, initial, mode):
        self.L = L
        self.N = N
        self.K = 4
        self.initial = initial
        self.mode = mode
        self.ps = np.round(np.linspace(0, 1, self.K), 2)
        self.rs = np.round(np.linspace(0, 1, self.K), 2)
        self.stream = self.data_stream()
        self.counter = 0

        self.fig, self.axes = plt.subplots(ncols=self.K, nrows=self.K, figsize=(8, 8))
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                          init_func=self.setup_plot, blit=True)
        for i in range(self.K):
            for j in range(self.K):
                ax = self.axes[i, j]
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.set_title(f"p={self.ps[i]:2}, r={self.rs[j]:2}")
                if i == 0:
                    ax.set_title(f"r={self.rs[j]:2}", size=14)
                if j == 0:
                    ax.set_ylabel(f"p={self.ps[i]:2}", size=14)

    def setup_plot(self):
        """Initial drawing of the matrix plot."""
        data = next(self.stream)
        self.img = []
        for i in range(self.K):
            for j in range(self.K):
                self.img.append(self.axes[i, j].imshow(data[i, j]))
        values = [0, 1]
        colors = [self.img[0].cmap(self.img[0].norm(value)) for value in values]
        label_dict = {
            0: "0",
            1: "1",
        }
        patches = [mpatches.Patch(color=colors[i], label=label_dict[i].format(l=values[i]) ) for i in range(len(values))]
        plt.legend(handles=patches, bbox_to_anchor=(1.02, self.K + (self.K-1) * 0.2), loc=2, borderaxespad=0., fontsize=12)
        return self.img

    def data_stream(self):
        """Animate directed percolation"""
        data = np.zeros((self.K, self.K, self.N, self.L))
        for i in range(self.K):
            for j in range(self.K):
                data[i, j, 0] = self.initial
        while True:
            self.counter += 1
            if self.counter < self.N:
                for i in range(self.K):
                    for j in range(self.K):
                        if self.mode == "dp":
                            next_dp_row = dp_step(data[i, j, self.counter-1], self.ps[i])
                            data[i, j, self.counter] = next_dp_row
                        elif self.mode == "pc":
                            next_pc_row = pc_step(data[i, j, self.counter-1], self.ps[i], self.rs[j])
                            data[i, j, self.counter] = next_pc_row
            else:
                for i in range(self.K):
                    for j in range(self.K):
                        if self.mode == "dp":
                            next_dp_row = dp_step(data[i, j, -1], self.ps[i])
                            data[i, j] = np.roll(data[i, j], shift=-1, axis=0)
                            data[i, j, -1] = next_dp_row
                        elif self.mode == "pc":
                            next_pc_row = pc_step(data[i, j, -1], self.ps[i], self.rs[j])
                            data[i, j] = np.roll(data[i, j], shift=-1, axis=0)
                            data[i, j, -1] = next_pc_row
            # plt.pause(1)
            yield data

    def update(self, i):
        """Update the matrix plot."""
        data = next(self.stream)
        for i in range(self.K):
            for j in range(self.K):
                self.img[i + j * self.K].set_data(data[i, j])

        return self.img


if __name__ == '__main__':
    # for p in [0.0, 1.0]:
    #     for r in [0.0, 1.0]:
    L = 50
    N = L
    # initial = np.zeros(L)
    # initial[int(L/2)] = 1
    initial = np.random.randint(0, 2, L)
    # p = 0.6447001
    # r = 0.3
    p = 1.0
    r = 1.0
    mode = "pc"
    # mode = "dp"
    a = AnimatedMatrix(L, N, initial, p, r, mode)
    # a = DoubleAnimatedMatrix(L, N, initial, p, r)
    # a = GridAnimatedMatrix(L, N, initial, mode)
    plt.show()
