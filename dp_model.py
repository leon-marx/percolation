import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as mpatches
import numba


@numba.jit()
def dp_step(old_state, p):
    """
    Performs a single step of directed percolation.

    old_state: np.array of shape(L)
        L: system size in the spatial dimension
    p: float in the range [0, 1], hyperparameter governing the transmission rate
    """
    L = len(old_state)
    new_state = np.zeros(L, dtype=np.uint8)
    for i in range(L):
        left = old_state[i-1]
        right = old_state[i]
        v = np.random.uniform(0, 1)
        if (left == 1) & (right == 1):
            if v < 1 - (1-p)**2:
                new_state[i] = 1
        elif (left == 1) | (right == 1):
            if v < p:
                new_state[i] = 1

    return new_state


def run_dp(initial_state, p, N):
    """
    Performs the directed percolation for N number of steps, starting from the given intial state.

    initial_state: np.array of shape(L)
        L: system size in the spatial dimension
    p: float in the range [0, 1], hyperparameter governing the transmission rate
    N: int, number of steps for which the simulation should run
    """
    print("Running the system...")
    all_states = np.zeros((N+1, L), dtype=np.uint8)
    all_states[0] = initial_state
    for i in tqdm(range(N)):
        state = dp_step(all_states[i], p)
        all_states[i+1] = state
    return all_states


def plot_dp(all_states):
    """
    Plots the whole directed percolation process given the corresponding data.

    all_states: list of N np.arrays, each of size L or size L-1.
    """
    data = np.array(all_states)
    L = data.shape[1]
    N = data.shape[0] - 1
    plt.figure(figsize=(10, 8))
    im = plt.imshow(np.array(all_states))
    values = np.unique(data.ravel())
    colors = [im.cmap(im.norm(value)) for value in values]
    label_dict = {
        0: "Inactive",
        1: "Active",
    }
    patches = [mpatches.Patch(color=colors[i], label=label_dict[i].format(l=values[i]) ) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Space", size=16)
    plt.ylabel("Time", size=16)
    plt.title(f"DP Model for system size {L} and time {N}", size=18)
    plt.show()


if __name__ == "__main__":
    L = 100000
    p = 0.6447001 + 0.1
    initial_state = np.random.randint(0, 2, L, dtype=np.uint8)  # (random)
    # initial_state = np.ones(L)  # all active
    # initial_state = np.zeros(L)  # (single point)
    # initial_state[int(L/2)] = 1
    N = L
    C = 5

    data = run_dp(initial_state, p, N)
    # plot_dp(data)
    print(data[-1, -1])
