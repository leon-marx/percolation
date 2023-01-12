import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

def pc_step(old_state, p, r):
    """
    Performs a single step of parity conserving percolation.

    old_state: np.array of shape(L)
        L: system size in the spatial dimension
    p: float in the range [0, 1], hyperparameter governing the transmission rate
    r: float in the range [0, 1], hyperparameter governing the annihilation rate
    """
    for i in range(np.count_nonzero(old_state)):
        new_state = copy.deepcopy(old_state)
        ind = np.random.choice([i for i, site in enumerate(old_state) if site == 1])  # choose random occupied site
        v = np.random.uniform(0, 1)
        if v < 1 - p:  # generate
            new_state[ind-1] = 1
            new_state[(ind+1) % len(old_state)] = 1
            vv = np.random.uniform(0, 1)
            if vv < r:  # (possible) annihilation
                if old_state[ind-1] == 1:
                    new_state[ind-1] = 0
                if old_state[(ind+1) % len(old_state)] == 1:
                    new_state[(ind+1) % len(old_state)] = 0
        else:  # diffuse
            vv = np.random.uniform(0, 1)
            if vv < 0.5:  # left
                if old_state[ind-1] == 1:
                    vvv = np.random.uniform(0, 1)
                    if vvv < r:  # annihilation
                        new_state[ind-1] = 0
                        new_state[ind] = 0
            else:  # right
                if old_state[(ind+1) % len(old_state)] == 1:
                    vvv = np.random.uniform(0, 1)
                    if vvv < r:  # annihilation
                        new_state[(ind+1) % len(old_state)] = 0
                        new_state[ind] = 0

        old_state = new_state
    return new_state


def run_pc(initial_state, p, r, N):
    """
    Performs the parity conserving percolation for N number of steps, starting from the given intial state.

    initial_state: np.array of shape(L)
        L: system size in the spatial dimension
    p: float in the range [0, 1], hyperparameter governing the transmission rate
    N: number of steps for which the simulation should run
    """
    print("Running the system...")
    all_states = [initial_state]
    for i in tqdm(range(N)):
        state = pc_step(all_states[i], p, r)
        all_states.append(state)
    return all_states


def plot_pc(all_states):
    """
    Plots the whole parity conserving percolation process given the corresponding data.

    all_states: list of N np.arrays, each of size L or size L-1.
    """
    L = len(all_states[0])
    # N = len(all_states)
    x = []
    y = []
    v = []
    for i, state in enumerate(all_states):
        y += [L-i for k in range(L)]
        x += [j for j in range(L)]
        v += [*state]
        # for j, site in enumerate(state):
        #     if len(state) < L:
        #         offset = 0.5
        #     else:
        #         offset = 0
        #     x.append(j + offset)
        #     y.append(L-i)
        #     v.append(site)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(x=x, y=y, c=v, marker="o", s=800/L, label=v)
    plt.legend(handles=scatter.legend_elements()[0], labels=["Inactive", "Active"], bbox_to_anchor=[1.1, 0], fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Space", size=14)
    plt.ylabel("Time", size=14)
    plt.title(f"Parity Conserving Percolation for system size {L} and time {N}", size=18)
    plt.show()


L = 100
p = 0.6447001
r = 0.5
# initial_state = np.random.randint(0, 2, L)  # (random)
# initial_state = np.zeros(L)  # (single point)
initial_state = np.ones(L)  # all active
initial_state[int(L/2)] = 1
N = L

data = run_pc(initial_state, p, r, N)
plot_pc(data)
