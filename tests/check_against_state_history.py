import numpy as np
import re
import pickle
import lzma
from pathlib import Path
import dstar_lite
import matplotlib.pyplot as plt
from dstar_python import DStarLite, State

def save_compressed_pickle(object, path: Path, overwrite=False):
    if type(path) == str:
        path = Path(path)
    if path.exists() and not overwrite:
        raise ValueError(f'{path} already exists')
    with lzma.open(path, 'wb') as f:
        pickle.dump(object, f)
def load_compressed_pickle(path):
    with lzma.open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj




def check_state(dstar:DStarLite, state_file:Path, costmap)->bool:
    state = load_compressed_pickle(state_file)
    g_values, rhs_values = np.asarray(state["g_values"]).reshape(dstar.g.shape), np.asarray(state["rhs_values"]).reshape(dstar.rhs.shape)
    cmap='viridis'
    if not np.allclose(dstar.g, g_values, atol=1e-6):
        diff = np.sum(np.abs(dstar.g - g_values) > 1e-6)
        print(g_values[np.abs(dstar.g - g_values) > 1e-6], dstar.g[np.abs(dstar.g - g_values) > 1e-6], np.where(np.abs(dstar.g - g_values) > 1e-6))
        fig, ax = plt.subplots(1,5, sharex=True, sharey=True)
        ax[0].imshow(dstar.g, cmap=cmap, interpolation='none')
        ax[1].imshow(g_values, cmap=cmap, interpolation='none')
        ax[2].imshow(np.abs(dstar.g - g_values) > 1e-6, cmap=cmap, interpolation='none')
        ax[3].imshow(dstar.g - g_values, cmap=cmap, interpolation='none')
        ax[4].imshow(costmap, cmap=cmap, interpolation='none')
        plt.show()
        print("g values do not match! total number that differ ", diff) 
        return False
    if not np.allclose(dstar.rhs, rhs_values):
        diff = np.sum(np.abs(dstar.rhs - rhs_values) > 1e-6)
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(dstar.rhs, cmap='gray', interpolation='none')
        ax[1].imshow(rhs_values, cmap='gray', interpolation='none')
        ax[2].imshow(dstar.rhs - rhs_values, cmap='gray', interpolation='none')
        plt.show()
        print("rhs values do not match! total number that differ ", diff)
        return False
    return True



def check_against_log(root_dir:Path):

    init_state = load_compressed_pickle(root_dir / "init_state.pkl")

    costmap, start, goal, keys = init_state["costmap"], init_state["start"], init_state["goal"], init_state['keys']
    print("start", start, "goal", goal)
    python_dstar = DStarLite(costmap, State(*start), State(*goal))
    python_dstar.Initialize()
    python_dstar.ComputeShortestPath()
    print(python_dstar.pq)
    print("C++ keys")
    for key in keys:
        print(f"\t{key[2]},{key[3]}: {key[0]},{key[1]}")

    initial_state_matches = check_state(python_dstar, root_dir / "init_state.pkl", costmap)
    if not initial_state_matches:
        raise ValueError(f"Initial state does not match")

    with open(root_dir / "log.txt", "r") as f:
        lines = f.readlines()

    for line in lines:
        print("processing line ", line, end="")
        if "Update Start idx" in line:
            match = re.search(r"Update Start idx \d+: (\d+), (\d+)", line)
            i, j = int(match.group(1)), int(match.group(2))
            python_dstar.UpdateStart(State(i, j))
        elif "Update Costmap idx" in line:
            match = re.search(r"Update Costmap idx (\d+)", line)
            idx = int(match.group(1))
            state_file = root_dir / f"costmap_update_{idx}.pkl"
            costmap_changes = load_compressed_pickle(state_file)
            indexes, values = np.asarray(costmap_changes["indexes"]), np.asarray(costmap_changes["values"])
            python_dstar.IncorperateEdgeChanges(indexes, values)
        elif "Replan" in line:
            match = re.search(r"Replan (\d+) called with", line)
            idx = int(match.group(1))
            pre_file = root_dir / f"pre_state_{idx}.pkl"
            if not check_state(python_dstar, pre_file, costmap):
                raise ValueError(f"Pre state {idx} does not match")
            print("passed zero")
            python_dstar.ComputeShortestPath()
            print("passed one")
            post_file = root_dir / f"post_state_{idx}.pkl"
            if not check_state(python_dstar, post_file, costmap):
                raise ValueError(f"Post state {idx} does not match")

        elif "Dstar Lite Log" in line:
            pass
        else:
            raise ValueError(f"Unknown log line: {line}")


if __name__ == "__main__":
    root_dir = Path("/tmp/dstar_lite_logs")
    check_against_log(root_dir)
