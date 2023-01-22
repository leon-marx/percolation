import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as mpatches
import numba


@numba.jit()
def pc_step(old_state, p, r):  # taken from Livi & Politi
    """
    Performs a single step of parity conserving percolation.

    old_state: np.array of shape(L)
        L: system size in the spatial dimension
    p: float in the range [0, 1], hyperparameter governing the transmission rate
    r: float in the range [0, 1], hyperparameter governing the annihilation rate
    """
    L = len(old_state)
    C = np.count_nonzero(old_state)
    if C == 0:
        return old_state
    new_state = old_state.copy()
    ind = np.random.choice(np.array([j for j, site in enumerate(old_state) if site == 1]))  # choose random occupied site
    v = np.random.uniform(0, 1)
    if v < p:  # diffuse
        vv = np.random.uniform(0, 1)
        if vv < 0.5:  # left
            if old_state[ind-1] == 1:  # occupied
                vvv = np.random.uniform(0, 1)
                if vvv < r:  # annihilate
                    new_state[ind-1] = 0
                    new_state[ind] = 0
            else:  # actually diffuse
                new_state[ind-1] = 1
                new_state[ind] = 0
        else:  # right
            if old_state[(ind+1) % L] == 1:  # occupied
                vvv = np.random.uniform(0, 1)
                if vvv < r:  # annihilate
                    new_state[ind] = 0
                    new_state[(ind+1) % L] = 0
            else:  # actually diffuse
                new_state[ind] = 0
                new_state[(ind+1) % L] = 1
    else:  # generate
        if (old_state[ind-1] == 0) & (old_state[(ind+1) % L] == 0):  # actually generate
            new_state[ind-1] = 1
            new_state[ind] = 1
            new_state[(ind+1) % L] = 1
        elif (old_state[ind-1] == 1) & (old_state[(ind+1) % L] == 0):  # left occupied
            vv = np.random.uniform(0, 1)
            if vv < r:  # annihilate
                new_state[ind-1] = 0
                new_state[ind] = 1
                new_state[(ind+1) % L] = 1
        elif (old_state[ind-1] == 0) & (old_state[(ind+1) % L] == 1):  # right occupied
            vv = np.random.uniform(0, 1)
            if vv < r:  # annihilate
                new_state[ind-1] = 1
                new_state[ind] = 1
                new_state[(ind+1) % L] = 0
        else:  # both occupied
            vv = np.random.uniform(0, 1)
            if vv < r ** 2:  # annihilate
                new_state[ind-1] = 0
                new_state[ind] = 1
                new_state[(ind+1) % L] = 0

    return new_state


def run_pc(initial_state, p, r, N):
    """
    Performs the parity conserving percolation for N number of steps, starting from the given intial state.

    initial_state: np.array of shape(L)
        L: system size in the spatial dimension
    p: float in the range [0, 1], hyperparameter governing the transmission rate
    r: float in the range [0, 1], hyperparameter governing the annihilation rate
    N: int, number of steps for which the simulation should run
    """
    print("Running the system...")
    all_states = np.zeros((N+1, L), dtype=np.uint8)
    all_states[0] = initial_state
    for i in tqdm(range(N)):
        state = pc_step(all_states[i], p, r)
        all_states[i+1] = state
    return all_states


def plot_pc(all_states):
    """
    Plots the whole parity conserving percolation process given the corresponding data.

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
    plt.title(f"PC Model for system size {L} and time {N}", size=18)
    plt.show()


if __name__ == "__main__":
    L = 100000
    p = 0.6447001
    r = 0.5
    initial_state = np.random.randint(0, 2, L, dtype=np.uint8)  # (random)
    # initial_state = np.zeros(L)  # (single point)
    # initial_state[int(L/2)] = 1
    # initial_state = np.ones(L)  # all active
    N = L

    data = run_pc(initial_state, p, r, N)
    # plot_pc(data)
    # print(data)
    print(data[-1, -1])
