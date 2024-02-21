from dstar_python import DStarLite, State
from dstar_lite import Dstar
import numpy as np

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
map_size = 4
start_map = np.asarray([
    [2,1,1,3],
    [2,4,6,2],
    [1,4,6,5],
    [5,1,3,2]
], dtype=float)
start = (0,0)
goal = (3,3)

def edge_cost_changes(i, s, g):
    if i == 0:
        return [
            (0,0,1),
            (0,1,3),
            (1,1,2), 
            (2,0,4),
            (2,1,3)
        ]
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
print("G", assemble_values(dstar_cpp.getGValues(), (map_size, map_size)))
print("RHS", assemble_values(dstar_cpp.getRHSValues(), (map_size,map_size)))
print("Keys", dstar_cpp.getKeys())
print("Path", dstar_cpp.getPath())
new_start = dstar_cpp.getPath()[1]
dstar_cpp.updateStart(*new_start)
idx, val = edge_costs_to_updateCells(edge_cost_changes(0,None,None))
dstar_cpp.updateCells(idx,val)
print ("####### STATE AFTER UPDATING CELLS ")
print("G", assemble_values(dstar_cpp.getGValues(), (map_size,map_size)))
print("RHS", assemble_values(dstar_cpp.getRHSValues(), (map_size,map_size)))
print("Keys", dstar_cpp.getKeys())
dstar_cpp.replan()
print("G", assemble_values(dstar_cpp.getGValues(), (map_size,map_size)))
print("RHS", assemble_values(dstar_cpp.getRHSValues(), (map_size,map_size)))
print("Keys", dstar_cpp.getKeys())

