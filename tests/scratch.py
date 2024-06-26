import dstar_lite
import pyastar2d
import time 
import faulthandler
import numpy as np 
import matplotlib.pyplot as plt
# from radial_learning.planners.planner_general import get_acc_costs

faulthandler.enable()

def assemble_values(value_list, shape = None):
    if shape is None:
        max_i, max_j = 0,0
        for item in value_list:
            i,j = item[:2] 
            max_i = max(max_i, i)
            max_j = max(max_j, j)
        shape = (max_i+1, max_j+1)
    output = np.ones(shape) * np.inf
    if len(value_list[0]) == 4:
        # add a new axis for the second value
        output = np.stack([output, np.ones(shape) * np.inf], axis=-1)
    for item in value_list:
        if len(item) == 3:
            i,j,value = item
            output[i,j] = value
        elif len(item) == 4:
            i,j,value1,value2 = item
            output[i,j,0] = value1
            output[i,j,1] = value2
    return output

def plot_values(valueArg):
    if len(valueArg) == 0:
        return
    if len(valueArg[0]) == 3:
        fig, ax = plt.subplots()
    elif len(valueArg[0]) == 4:
        fig, ax = plt.subplots(1,2)
    max_i, max_j = 0,0
    for item in valueArg:
        i,j = item[:2] 
        max_i = max(max_i, i)
        max_j = max(max_j, j)
    output1 = np.ones((max_i+1, max_j+1)) * np.inf
    output2 = np.ones((max_i+1, max_j+1)) * np.inf
    if len(valueArg) > np.prod(output1.shape):
        print("Error: valueArg has more elements than the shape", len(valueArg), output1.shape)
    for item in valueArg:
        value2 = np.inf
        if len(item) == 3:
            i,j,value1 = item
        elif len(item) == 4:
            i,j,value1,value2 = item
        output1[i,j] = value1
        output2[i,j] = value2
    if len(valueArg[0]) == 3: 
        ax.imshow(output1, cmap='gray', interpolation='none')
    elif len(valueArg[0]) == 4:
        ax[0].imshow(output1, cmap='gray', interpolation='none')
        ax[1].imshow(output2, cmap='gray', interpolation='none')


def plot_paths(mapArg, paths, path_names, start,goal):
    fig, ax = plt.subplots()
    ax.imshow(mapArg, cmap='gray', interpolation='nearest')
    for path,name in zip(paths, path_names):
        ax.plot([x[1] for x in path], [x[0] for x in path],'.-', label=name)
    ax.plot(start[1], start[0], 'ro', label='start')
    ax.plot(goal[1], goal[0], 'bo', label='goal')
    ax.legend()
    plt.show()


def get_acc_costs(path, mapArg, _):
    cost = 0
    path = list(reversed(path))
    for i in range(len(path)-1):
        scale = 1
        if abs(path[i+1][0] - path[i][0]) + abs(path[i+1][1] - path[i][1]) > 1:
            scale = np.sqrt(2)
        cost += mapArg[path[i][0], path[i][1]] * scale
    return [cost]

def generate_map(shape):
    valid_map = np.random.rand(shape[0],shape[1]) * 5 + 1

    # make random untraversable patches
    # for _ in range(20):
    #     ii, jj = np.random.randint(0,shape[0]), np.random.randint(0,shape[1])
    #     width = np.random.randint(0,max(1,shape[0]//10))
    #     height = np.random.randint(0,max(shape[1]//10, 1))
    #     valid_map[ii:min(ii+width,shape[0]),jj:min(jj+height,shape[1])] = np.inf
    return valid_map


def perform_planning_iteration(dstar, current_state, goal, current_map, indexes, values, plot=False):
    # before_g = dstar.getGValues()
    # before_rhs = dstar.getRHSValues()
    # before_keys = dstar.getKeys()
    # for i in range(len(indexes)):
        # print("index", indexes[i], "value", values[i], "current value", current_map[indexes[i][0], indexes[i][1]])
    t1 = time.time()
    dstar.updateStart(*current_state)
    if len(indexes) > 0:
        indexes = np.asarray(indexes, dtype=np.int32)
        values = np.asarray(values)
        # print("updating cells")
        dstar.updateCells(indexes, values)

        # print("AFTER FIRST CALL TO update map cells")
        # print("G\n", assemble_values(dstar.getGValues(), valid_map.shape))
        # print("RHS\n", assemble_values(dstar.getRHSValues(), valid_map.shape))
        # print("First keys\n", assemble_values(dstar.getKeys(), valid_map.shape)[:,:,0])
        # print("Second keys\n", assemble_values(dstar.getKeys(), valid_map.shape)[:,:,1])
        # print("replanning")
        # assert False
    dstar_success = dstar.replan()
    t2 = time.time()
    if not dstar_success:
        # plot_values(before_g)
        # plot_values(before_rhs)
        # plot_values(before_keys) 
        # print(before_keys)
        # plot_values(dstar.getGValues())
        # plot_values(dstar.getRHSValues())
        # plot_values(dstar.getKeys())
        # fig, ax = plt.subplots()
        # ax.imshow(current_map, cmap='gray', interpolation='none')
        print("Dstar replan failed")
        return -1
        # plt.show()
    # print("getting path")
    dstar_plan = dstar.getPath()
    print("done with dstar")
    if plot:
        plot_values(dstar.getGValues())
        plot_values(dstar.getRHSValues())
        plot_values(dstar.getKeys())

    print("Starting from ", current_state)
    dstar_costs = get_acc_costs(dstar_plan, current_map, 1.0)
    print("Dstar cost: ", dstar_costs[-1], " length", len(dstar_plan), "success", dstar_success, "time", t2-t1)
    t1 = time.time()
    astar_success, path, cost = pyastar2d.astar_path(current_map.astype(np.float32), current_state, goal, allow_diagonal=True)
    t2 = time.time()
    astar_costs = get_acc_costs(path, current_map, 1.0)
    print("Astar cost: ", astar_costs[-1], " length", len(path), "success", astar_success, "time", t2-t1)
    if not np.isclose(dstar_costs[-1], astar_costs[-1]):
        print("COSTS ARE DIFFERENT!!!! ########")
    # if plot:
    #     plot_paths(current_map, [dstar_plan, path], ["Dstar", "Astar"], current_state, goal)

    # if not np.isclose(get_cost_from_path(dstar_plan, current_map), get_cost_from_path(path, current_map)):
    #     return -1

def change_costs_around_state(state, goal, valid_map, radius=100):
    indexes = []
    values = []
    for i in range(state[0]-radius, state[0]+radius):
        for j in range(state[1]-radius, state[1]+radius):
            if i < 0 or j < 0 or i >= valid_map.shape[0] or j >= valid_map.shape[1]:
                continue
            if state[0] == i and state[1] == j:
                continue
            if goal[0] == i and goal[1] == j:
                continue
            if ((i-state[0])**2 + (j-state[1])**2) < radius**2:
                indexes.append((i,j))
                new_value = np.random.rand() * 5 + 1
                new_value = round(new_value, 0)
                if np.random.rand() > 0.99:
                    new_value = np.inf
                # print("changing value at", i,j, "from", valid_map[i,j], "to", new_value)
                values.append(new_value)
                valid_map[i,j] = new_value

    return valid_map, indexes, values

def test_against_paper_example():
    paper_map = np.ones((5,3))
    paper_map[1,1] = np.inf
    paper_map[2,1] = np.inf
    start = (1,0)
    goal = (4,2)
    dstar = dstar_lite.Dstar(paper_map, 10000, scale_diag_cost=False)
    dstar.init(*start, *goal)
    perform_planning_iteration(dstar, start, goal, paper_map, [], [], True)
    dstar.updateStart(2,0)
    dstar.updateCells(np.asarray((3,1), dtype=np.int32).reshape(1,2), np.asarray([np.inf]))
    plot_values(dstar.getGValues())
    plot_values(dstar.getRHSValues())
    plot_values(dstar.getKeys())
    perform_planning_iteration(dstar, (2,0), goal, paper_map, [], [], True)


# test_against_paper_example()

best_seed = None
best_iter_seed = np.inf
# for seed_option in [126]:
for seed_option in range(1000):
    size=4000
    start = (0,0)
    plot_things = False
    print("Seed is", seed_option)
    np.random.seed(seed_option)

    goal = (size-1, size-1)
    valid_map = generate_map((size,size)).round(0)
    print("initial map is", valid_map)
    print("start is ", start)
    print("goal is ", goal)
    # print("Generated map, starting test")
    t1 = time.time()
    dstar = dstar_lite.Dstar(valid_map, size*size*10, scale_diag_cost=True)
    dstar.init(*start, *goal)
    t2 = time.time()
    # print("Dstar init time", t2-t1)
    perform_planning_iteration(dstar, start, goal, valid_map, [], [], plot_things)
    # print("AFTER FIRST CALL TO CALCULATESHORTESTPATH")
    # print("G\n", assemble_values(dstar.getGValues(), valid_map.shape))
    # print("RHS\n", assemble_values(dstar.getRHSValues(), valid_map.shape))
    # print("First keys\n", assemble_values(dstar.getKeys(), valid_map.shape)[:,:,0])
    # print("Second keys\n", assemble_values(dstar.getKeys(), valid_map.shape)[:,:,1])
    # print("Path\n", dstar.getPath())
    for i in range(100000):
        print(i)
        if len(dstar.getPath()) <= 2:
            break
        next_state = dstar.getPath()[1]
        # next_state = [np.random.randint(0,size), np.random.randint(0,size)]
        # while True:
        #     if valid_map[next_state[0], next_state[1]] < np.inf:
        #         break
        #     next_state = [np.random.randint(0,size), np.random.randint(0,size)]
        valid_map, indexes, values = valid_map, [], []
        print("NEXT STATE", next_state)
        # print(valid_map)
        valid_map, indexes, values = change_costs_around_state(next_state, goal, valid_map, 100)
        # print("CHANGES ARE ", indexes, values)
        # print("map is now \n", valid_map)


        result = perform_planning_iteration(dstar, next_state, goal, valid_map, indexes, values, plot_things)
        if result == -1:
            if i < best_iter_seed:
                best_iter_seed = i
                best_seed = seed_option
            print(best_seed)
            assert False

print("Best seed", best_seed, "best iter seed", best_iter_seed)

        
        # perform_planning_iteration(dstar, next_state, goal, valid_map, [], [], True)
