from dstar_python import DStarLite, State
from dstar_lite import Dstar
import numpy as np

map_size = 4
start_map = np.asarray([
 [2, 5, np.inf,2.],
 [5, 4, np.inf,1.],
 [3, np.inf, np.inf, 5.],
 [3, 6, 1, 2.]
], dtype=float)
start = (0,0)
goal = (3,3)

def edge_cost_changes(i, s, g):
    # if i == 0:
    #     idxs = [(0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2)]
    #     values =  [3.0, 3.0, 5.0, 5.0, 4.0, 4.0, 5.0, 4.0, 3.0, 6.0, 2.0, 5.0, 3.0, 2.0]
    #     return [(i, j, v) for (i, j), v in zip(idxs, values)]
    if i == 1:
        idxs = [(0, 0)]
        values = [np.inf]
        return [(i, j, v) for (i, j), v in zip(idxs, values)]
    # elif i == 2:
    #     idxs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2)] 
    #     values = [6.0, 3.0, 2.0, 4.0, 3.0, 3.0, 6.0, 5.0, 4.0, 4.0, 4.0, 1.0, 4.0, 5.0]
    #     return [(i, j, v) for (i, j), v in zip(idxs, values)]
    else:
        return []
def edge_costs_to_updateCells(changes):
    n = len(changes)
    if n == 0:
        return [], []
    indexes = np.zeros((n,2), dtype=np.int32)
    values = np.zeros((n,), dtype=float)
    for i, change in enumerate(changes):
        indexes[i,0] = change[0]
        indexes[i,1] = change[1]
        values[i] = change[2]
    return indexes, values
dstar_cpp = Dstar(start_map, 100, scale_diag_cost=True)
dstar_cpp.init(*start, *goal)
start_s = State(*start)
goal_s = State(*goal)
dstar_py = DStarLite(start_map, start_s, goal_s)


dstar_py.Main(edge_cost_changes)

print ("############################ DONE WITH PYTHON##############################")

dstar_cpp.replan()
print("G", np.asarray(dstar_cpp.getGValues()).reshape(map_size,map_size))
print("RHS", np.asarray(dstar_cpp.getRHSValues()).reshape(map_size,map_size))
print("Keys", dstar_cpp.getKeys())
print("Path", dstar_cpp.getPath())
new_start = dstar_cpp.getPath()[1]
dstar_cpp.updateStart(*new_start)
idx, val = edge_costs_to_updateCells(edge_cost_changes(1,None,None))
dstar_cpp.updateCells(idx,val)
print ("####### STATE AFTER UPDATING CELLS ")
print("G", dstar_cpp.getGValues().reshape(map_size,map_size))
print("RHS", dstar_cpp.getRHSValues().reshape(map_size,map_size))
print("Keys", dstar_cpp.getKeys())
dstar_cpp.replan()
print("G", dstar_cpp.getGValues().reshape(map_size,map_size))
print("RHS", dstar_cpp.getRHSValues().reshape(map_size,map_size))
print("Keys", dstar_cpp.getKeys())

