import dstar_lite
import pyastar2d
import numpy as np 
import matplotlib.pyplot as plt

def plot_paths(mapArg, paths, path_names, start,goal):
    fig, ax = plt.subplots()
    ax.imshow(mapArg, cmap='gray', interpolation='nearest')
    for path,name in zip(paths, path_names):
        ax.plot([x[1] for x in path], [x[0] for x in path], label=name)
    ax.plot(start[1], start[0], 'ro', label='start')
    ax.plot(goal[1], goal[0], 'bo', label='goal')
    ax.legend()
    plt.show()


def get_cost_from_path(path, mapArg):
    cost = 0
    for i in range(len(path)-1):
        scale = 1
        if abs(path[i-1][0] - path[i][0]) + abs(path[i-1][1] - path[i][1]) > 1:
            scale = np.sqrt(2)
        cost += mapArg[path[i][0], path[i][1]] * scale
    return cost

def generate_map(shape):
    valid_map = np.random.rand(shape[0],shape[1]) * 5 + 1

    # make random untraversable patches
    for _ in range(20):
        ii, jj = np.random.randint(0,shape[0]), np.random.randint(0,shape[1])
        width = np.random.randint(0,shape[0]//10)
        height = np.random.randint(0,shape[1]//10)
        valid_map[ii:min(ii+width,5000),jj:min(jj+height,5000)] = np.inf
    return valid_map

start = (0,0)
goal = (9,9)
valid_map = generate_map((10,10))
dstar = dstar_lite.Dstar(valid_map)
dstar.init(*start, *goal)
dstar_success = dstar.replan()
# dstar_plan = dstar.getPath()
# print("Dstar cost: ", get_cost_from_path(dstar_plan, valid_map), " length", len(dstar_plan), "success", dstar_success)


# astar_success, path, cost = pyastar2d.astar_path(valid_map.astype(np.float32), start, goal, allow_diagonal=True)
# print("Astar cost: ", cost, " length", len(path), "success", astar_success)

# plot_paths(valid_map, [dstar_plan, path], ["Dstar", "Astar"], start, goal)


