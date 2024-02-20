from dataclasses import dataclass
from typing import Any
import numpy as np 

@dataclass
class State:
    i: int
    j: int

    def __eq__(self, __value: object) -> bool:
        return (self.i,self.j) == (__value.i, __value.j)
    def __hash__(self) -> int:
        return hash((self.i,self.j))
    def __repr__(self) -> str:
        return f"State({self.i},{self.j})"

class TwoKeyQueue:
    def __init__(self):
        self.items = {} # map key pair (state->[float, float])
    def Insert(self,u:State, key:tuple[float,float])->None:
        # check that it's not in the list
        if u in self.items:
            raise RuntimeError(f"Inserting state {u} that is already in the queue with value {self.items[u]}")
        self.items[u] = key

    def Update(self, u:State, key:tuple[float,float])->None:
        if u not in self.items:
            raise RuntimeError(f"Updating state {u} that is not in queue")
        
        # if key == self.items[u]:
        #     raise RuntimeWarning(f"Called update and set the key to be the same as it already was")
        self.items[u] = key

    def Remove(self, u:State):
        if u not in self.items:
            raise RuntimeError(f"Called remove on a state that is not in the queue: {u}")
        del self.items[u]

    def In(self, u:State):
        return u in self.items 
    def Pop(self):
        _, best_state = self._find_top_of_queue()
        del self.items[best_state]
    def TopKey(self):
        best_key, _ = self._find_top_of_queue()
        return best_key
    def Top(self):
        _, best_state = self._find_top_of_queue()
        return best_state

    def _find_top_of_queue(self):
        """Return the best key, value pair"""
        if len(self.items) == 0:
            raise RuntimeError("Called queue sort when the queue is empty")
        best_key = (np.inf, np.inf)
        best_state = None
        for state, key in self.items.items():
            if self._lessthan(key, best_key):
                best_key = key
                best_state = state
        return best_key, best_state

    @staticmethod
    def _lessthan(a, b):
        """Is a less than b?"""
        if a[0] < b[0]:
            return True
        elif a[0] > b[0]:
            return False
        else:
            return a[1] < b[1]
    def __repr__(self) -> str:
        output = ""
        if len(self.items) == 0:
            return "PQ is empty"
        k, s = self._find_top_of_queue()
        output += f"Top: {s} -> {k}\n"
        output += "List:\n"
        for k, v in self.items.items():
            output += f"\t{k}->{v}\n"
        return output
        
def test_priority_queue():
    pq = TwoKeyQueue()
    pq.Insert(State(2,2), (4.55, 8.2))
    pq.Insert(State(1,1), (3.55, 8.2))
    pq.Insert(State(3,3), (4.55, 8.1))
    assert pq.Top() == State(1,1)
    assert pq.TopKey() == (3.55, 8.2)
    pq.Pop()
    assert pq.Top() == State(3,3)
    assert pq.TopKey() == (4.55, 8.1)
    pq.Pop()
    assert pq.Top() == State(2,2)
    assert pq.TopKey() == (4.55, 8.2)
    pq.Pop()
    try:
        pq.Top()
    except RuntimeError:
        print("handled exception correctly")
test_priority_queue()

class DStarLite:
    def __init__(self, mapArg:np.ndarray, startArg:State, goalArg:State) -> None:
        self.costmap = mapArg
        self.s_start = startArg
        self.s_goal = goalArg
        self.g = None
        self.s_last = None
        self.rhs = None
        self.pq = None
        self.k_m = None

    def Initialize(self):
        # line 02
        self.pq = TwoKeyQueue()
        #line 03
        self.k_m = 0.0
        # line 04
        self.g = np.ones(self.costmap.shape) * np.inf
        self.rhs = np.ones(self.costmap.shape) * np.inf
        # line 05
        self.rhs[self.s_goal.i, self.s_goal.j] = 0
        # line 06
        self.pq.Insert(self.s_goal, self.CalcKey(self.s_goal))

    def UpdateVertex(self, u:State)->None:
        g_is_rhs = self.g[u.i, u.j] == self.rhs[u.i, u.j]
        u_in_pq = self.pq.In(u)
        if not g_is_rhs and u_in_pq:
            self.pq.Update(u, self.CalcKey(u))
        elif not g_is_rhs and not u_in_pq:
            self.pq.Insert(u, self.CalcKey(u))
        elif g_is_rhs and u_in_pq:
            self.pq.Remove(u)

    def Pred(self, u:State)->list[State]:
        output = []
        i,j = u.i, u.j
        for idelta in [-1, 0, 1]:
            for jdelta in [-1, 0, 1]:
                new_i = i+idelta
                new_j = j+jdelta
                if new_i < 0 or new_i >= self.costmap.shape[0] or new_j < 0 or new_j >= self.costmap.shape[1]:
                    continue
                if idelta == 0 and jdelta == 0:
                    continue
                output.append(State(new_i, new_j))
        return output
    
    def c(self, u:State, v:State)->float:
        if (abs(u.i-v.i) + abs(u.j - v.j)) > 2:
            raise RuntimeError(f"c was called for two states not next to each other: {u}->{v}")
        scale = 1
        if (abs(u.i-v.i) + abs(u.j - v.j)) == 2:
            scale = np.sqrt(2)
        return scale * self.costmap[v.i, v.j]
                

    
    def ComputeShortestPath(self, printout=True)->None:
        # line 10
        while (TwoKeyQueue._lessthan(self.pq.TopKey(), self.CalcKey(self.s_start)) or \
               self.rhs[self.s_start.i, self.s_start.j] > self.g[self.s_start.i, self.s_start.j]):
            # line 11, 12, 13 
            u = self.pq.Top()
            k_old = self.pq.TopKey()
            k_new = self.CalcKey(u)
            if printout: print(f"Working with state {u}, k_old {k_old} and k_new {k_new}")
            # line 14
            if TwoKeyQueue._lessthan(k_old, k_new):
                # line 15
                if printout: print(f"\tupdated it's key: {u} -> {k_new}")
                self.pq.Update(u, k_new)
            # line 16
            elif self.g[u.i, u.j] > self.rhs[u.i, u.j]:
                # line 17
                self.g[u.i, u.j] =  self.rhs[u.i, u.j]
                # line 18
                self.pq.Remove(u)
                if printout: print(f"\tExpanding state {u} and adding:")
                # line 19
                for s in self.Pred(u):
                    # line 20
                    self.rhs[s.i, s.j] = min(self.rhs[s.i, s.j], self.c(s,u) + self.g[u.i, u.j])
                    if printout: print(f"\t\t{s} with rhs {self.rhs[s.i, s.j]}")
                    # line 21
                    self.UpdateVertex(s)
            # line 22
            else:
                # line 23
                g_old = self.g[u.i, u.j]
                # line 24
                self.g[u.i, u.j] = np.inf
                # line 25
                pred_plus_u = self.Pred(u)
                pred_plus_u.append(u)
                if printout: print(f"\tChanging g to inf and updating predecessors and self:")
                for s in pred_plus_u:
                    # line 26
                    if self.rhs[s.i, s.j] == self.c(s, u) + g_old:
                        # line 27
                        if s != self.s_goal:
                            min_value = np.inf
                            for sp in self.Pred(s):
                                min_value = min(min_value, self.c(s, sp) + self.g[sp.i, sp.j])
                            self.rhs[s.i, s.j] = min_value
                    if printout: print(f"\t\t{s}, setting rhs to {self.rhs[s.i, s.j]}") 
                    self.UpdateVertex(s)

    def Main(self, edge_cost_changes:callable)->None:
        """
        edge_cost_changes[int iteration, State current_state, State goal]->[(i,j,newValue)]
        """
        # line 29
        self.s_last = self.s_start
        # line 30
        self.Initialize()
        # line 31
        self.ComputeShortestPath()
        # line 32
        iteration = 0
        while self.s_start != self.s_goal:
            print(f"Iteration: {iteration}. State: {self}")
            # line 33
            if self.rhs[self.s_start.i, self.s_start.j] == np.inf:
                raise RuntimeError("No path to goal")
            # line 34
            min_state = None
            min_value = np.inf
            for sp in self.Pred(self.s_start):
                val = self.c(self.s_start, sp) + self.g[sp.i, sp.j]
                if val < min_value:
                    min_value = val
                    min_state = sp
            print("moving to ", min_state)
            # line 35
            self.s_start = min_state
            edge_changes = edge_cost_changes(iteration, self.s_start, self.s_goal)
            print("Incorporating changes", edge_changes)
            # line 37
            if len(edge_changes) > 0:
                # line 38
                self.k_m += self.h(self.s_last, self.s_start)
                # line 39
                self.s_last = self.s_start
                # line 40
                edge_changed = False
                for proposed_change in edge_changes:
                    # if value doesn't change, don't update edges
                    if self.costmap[proposed_change[0], proposed_change[1]] == proposed_change[2]:
                        continue
                    print("### incorperating updates for proposed change", proposed_change)
                    edge_changed = True
                    v = State(proposed_change[0], proposed_change[1])
                    old_costs = []
                    new_costs = []
                    pred = self.Pred(v)
                    for u in pred:
                        old_costs.append(self.c(u,v))
                    # line 42
                    self.costmap[v.i, v.j] = proposed_change[2]
                    for u in pred:
                        new_costs.append(self.c(u,v))
                    # line 41, 40
                    for u, c_new, c_old in zip(pred, new_costs, old_costs):
                        print(f"Looking at edge {u}->{v} with c_old {c_old} and c_new {c_new}")
                        # line 43
                        if c_old > c_new:
                            # line 44
                            print(f"\t updating rhs from {self.rhs[u.i, u.j]} to {min(self.rhs[u.i, u.j], c_new + self.g[v.i, v.j])}")
                            self.rhs[u.i, u.j] = min(self.rhs[u.i, u.j], c_new + self.g[v.i, v.j])
                        elif self.rhs[u.i, u.j] == c_old + self.g[v.i, v.j]:
                            print("\trhs is close to c_old + g")
                            if u != self.s_goal:
                                min_val = np.inf
                                succ = self.Pred(u)
                                print("\t successor scan:")
                                for sp in succ:
                                    print(f"\t\t{sp} cost {self.c(u,sp)} g value {self.g[sp.i, sp.j]}")
                                    min_val = min(min_val, self.c(u,sp) + self.g[sp.i, sp.j])
                                print(f"\tsetting rhs from {self.rhs[u.i, u.j]} to {min_val}")
                                self.rhs[u.i, u.j] = min_val
                            else:
                                print("u is goal, so exiting")
                        else:
                            print("\tNo action taken")
                        self.UpdateVertex(u)

                if edge_changed:
                    print("State after changes before computeShortestPath is ", self)
                    self.ComputeShortestPath() 







            iteration += 1

    
    def CalcKey(self, s:State)->tuple[float,float]:
        min_g_rhs = min(self.g[s.i, s.j], self.rhs[s.i, s.j])
        return (min_g_rhs + self.h(self.s_start, s) + self.k_m, min_g_rhs)

    def h(self, s1:State, s2:State)->float:
        min_delta = abs(s1.i - s2.i)
        max_delta = abs(s1.j - s2.j)
        if min_delta > max_delta:
            tmp = min_delta
            min_delta = max_delta
            max_delta = tmp
        return round((np.sqrt(2) - 1) * min_delta + max_delta, 3)

    def __repr__(self) -> str:
        out = f"k_m: {self.k_m} start: {self.s_start} goal: {self.s_goal} s_last: {self.s_last}\n"
        out += f"G: \n{self.g}\n"
        out += f"RHS: \n{self.rhs}\n"
        out += f"Pq: \n{self.pq}\n"
        return out

if __name__ == "__main__":
    costmap = np.asarray([
        [3,1,2],
        [3,2,1],
        [1,2,1]
    ], dtype=float)

    print("costmap is \n", costmap)
    start = State(0,0)
    goal = State(2,2)
    dstar =DStarLite(costmap, start, goal)

    def edge_cost_changes(i:int, start:State, goal:State)->list[tuple[int,int,float]]:
        return []

    dstar.Main(edge_cost_changes)