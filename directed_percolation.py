import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def dp_step(old_state, p):
    """
    Performs a single step of directed percolation.

    old_state: np.array of shape(L)
        L: system size in the spatial dimension
    p: float in the range [0, 1], hyperparameter governing the transmission rate
    """
    new_state = np.zeros(len(old_state))
    for i in range(len(new_state)):
        left = old_state[i]
        right = old_state[(i+1) % len(old_state)]  # PERIODIC BOUNDARY CONDITIONS ARE ASSUMED
        if (left == 0) & (right == 0):
            new_state[i] = 0
        elif (left == 1) & (right == 0):
            r = np.random.uniform(0, 1)
            if r < p:
                new_state[i] = 1
            else:
                new_state[i] = 0
        elif (left == 0) & (right == 1):
            r = np.random.uniform(0, 1)
            if r < 1-(1-p)**2:
                new_state[i] = 1
            else:
                new_state[i] = 0
        else:
            r = np.random.uniform(0, 1)
            if r < p:
                new_state[i] = 1
            else:
                new_state[i] = 0
    return new_state


def run_dp(initial_state, p, N):
    """
    Performs directed percolation for N number of steps, starting from the given intial state.

    initial_state: np.array of shape(L)
        L: system size in the spatial dimension
    p: float in the range [0, 1], hyperparameter governing the transmission rate
    N: number of steps for which the simulation should run
    """
    print("Running the system...")
    all_states = [initial_state]
    for i in tqdm(range(N)):
        state = dp_step(all_states[i], p)
        all_states.append(state)
    return all_states


def plot_dp(all_states):
    """
    Plots the whole directed percolation process given the corresponding data.

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
    plt.title(f"Directed Percolation for system size {L} and time {N}", size=18)
    plt.show()


if __name__ == "__main__":
    L = 100
    p = 0.6447001
    # initial_state = np.random.randint(0, 2, L)  # (random)
    initial_state = np.zeros(L)  # (single point)
    initial_state[int(L/2)] = 1
    N = L

    data = run_dp(initial_state, p, N)
    plot_dp(data)
