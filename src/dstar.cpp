/* Dstar.cpp
 * FROM https://github.com/ArekSredzki/dstar-lite/tree/master
 * James Neufeld (neufeld@cs.ualberta.ca)
 * Compilation fixed by Arek Sredzki (arek@sredzki.com)
 */

#include "dstar.h"


std::ostream& operator<<(std::ostream &os, const state& s)
{
    os << "i: " << s.i << " j: " << s.j << " k: " << s.k.first << " " << s.k.second;
    return os;
}

/* void Dstar::Dstar()
 * --------------------------
 * mapArg must be a double numpy 2d array
 * Each cell contains the cost of moving through it. So the heuristic is valid all cells must have a cost >= 1
 * Untraversable cells have a cost of infinity
 */
Dstar::Dstar(const py::array_t<double, py::array::c_style>& mapArg, int maxStepsArg, bool scale_diag_cost)
{
    maxSteps = maxStepsArg; // node expansions before we give up
    init_called = false;

    if (scale_diag_cost) {
        diag_cost_scale = sqrt(2);
    } else {
        diag_cost_scale = 1;
    }

    // create a copy of the numpy array
    auto buffer = mapArg.request();
    this->map = py::array_t<double>(buffer.shape, buffer.strides);
    auto r = this->map.mutable_unchecked<2>();
    auto r2 = mapArg.unchecked<2>();
    if (r.shape(0) > MAX_MAP_DIM || r.shape(1) > MAX_MAP_DIM)
    {
        std::throw_with_nested(std::runtime_error("Map is too large"));
    }
    for (py::ssize_t i = 0; i < r.shape(0); i++)
    {
        for (py::ssize_t j = 0; j < r.shape(1); j++)
        {
            if (r2(i, j) < 1) {
                throw std::runtime_error("All cells must have a cost >= 1");
            }
            r(i, j) = r2(i, j);
        }
    }

}
double Dstar::getMapCell(const state& s){
    auto r = map.unchecked<2>();
    return r(s.i, s.j);
}
void Dstar::setMapCell(const state& s, double val){
    auto r = map.mutable_unchecked<2>();
    r(s.i, s.j) = val;
}
void Dstar::printMap() {
    auto r = map.unchecked<2>();
    for (py::ssize_t i = 0; i < r.shape(0); i++)
    {
        for (py::ssize_t j = 0; j < r.shape(1); j++)
        {
            std::cout << r(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

/* float Dstar::keyHashCode(state u)
 * --------------------------
 * Returns the key hash code for the state u, this is used to compare
 * a state that have been updated
 */
float Dstar::keyHashCode(const state& u)
{
    if (u.k.first == std::numeric_limits<double>::infinity() || u.k.second == std::numeric_limits<double>::infinity()) {
        throw std::runtime_error("keyHashCode called with infinity");
    }
    return static_cast<float>(u.k.first + MAX_FIRST_KEY_VALUE * u.k.second);
}

bool Dstar::isStateInOpenList(const state& u)
{
    ds_oh::iterator cur = openHash.find(u);
    if (cur == openHash.end()) {
        return false;
    }
    return true;
}
bool Dstar::isStateWithKeyInOpenList(const state& u) {
    ds_oh::iterator cur = openHash.find(u);
    if (cur == openHash.end()) {
        return false;
    }
    return close(cur->second, keyHashCode(u));
}

bool Dstar::isStateInMap(const state& u) {
    auto r = map.unchecked<2>();
    return u.i >= 0 && u.j >= 0 && u.i < r.shape(0) && u.j < r.shape(1);
}

/* void Dstar::getPath()
 * --------------------------
 * Returns the path created by replan()
 */
py::list Dstar::getPath()
{
    py::list output_path;
    for (auto &s : path)
    {
        output_path.append(py::make_tuple(s.i, s.j));
    }
    return output_path;
}

/* bool Dstar::occupied(state u)
 * --------------------------
 * returns true if the cell is occupied (non-traversable), false
 * otherwise. non-traversable are marked with a cost of infinity.
 */
bool Dstar::occupied(const state& u)
{
    if (!isStateInMap(u))
    {
        return true;
    }
    return getMapCell(u) == std::numeric_limits<double>::infinity();
}

/* void Dstar::init(int sX, int sY, int gX, int gY)
 * --------------------------
 * Init dstar with start and goal coordinates, rest is as per
 * [S. Koenig, 2002]
 */
void Dstar::init(int sI, int sJ, int gI, int gJ)
{
    cellHash.clear();
    path.clear();
    openHash.clear();
    while (!openList.empty()) {
        openList.pop(); // line 02
    }

    k_m = 0; // line 03

    s_start.i = sI;
    s_start.j = sJ;
    s_goal.i = gI;
    s_goal.j = gJ;
    s_last = s_start; // line 29
    if (!isStateInMap(s_start) || !isStateInMap(s_goal))
    {
        throw std::runtime_error("Start or goal is outside of the map");
    }

    cellInfo tmp;
    tmp.rhs = 0; // line 05
    cellHash[s_goal] = tmp;

    insertOpen(calculateKey(s_goal)); // line 06

    init_called = true;
    edge_cost_changed = true;
}
/* void Dstar::makeNewCell(state u)
 * --------------------------
 * Checks if a cell is in the hash table, if not it adds it in.
 */
void Dstar::makeNewCell(const state& u)
{

    if (cellHash.find(u) != cellHash.end())
        return;

    cellInfo tmp;
    tmp.g = tmp.rhs = std::numeric_limits<double>::infinity(); // line 06
    cellHash[u] = tmp;
}

/* double Dstar::getG(state u)
 * --------------------------
 * Returns the G value for state u.
 */
double Dstar::getG(const state& u)
{

    if (cellHash.find(u) == cellHash.end()) {
        // return default value of cellInfo, line 04
        cellInfo tmp;
        return tmp.g;
    }
    return cellHash[u].g;
}

/* double Dstar::getRHS(state u)
 * --------------------------
 * Returns the rhs value for state u.
 */
double Dstar::getRHS(const state& u)
{

    if (u == s_goal)
        return 0;

    if (cellHash.find(u) == cellHash.end()) {
        cellInfo tmp; // return default value of cellInfo, line 04
        return tmp.rhs;
    }
    return cellHash[u].rhs;
}

/* void Dstar::setG(state u, double g)
 * --------------------------
 * Sets the G value for state u
 */
void Dstar::setG(state u, double g)
{

    makeNewCell(u);
    cellHash[u].g = g;
}

/* void Dstar::setRHS(state u, double rhs)
 * --------------------------
 * Sets the rhs value for state u
 */
void Dstar::setRHS(state u, double rhs)
{
    makeNewCell(u);
    cellHash[u].rhs = rhs;
}

/* double Dstar::eightCondist(state a, state b)
 * --------------------------
 * Returns the 8-way distance between state a and state b.
 */
double Dstar::eightCondist(const state& a, const state& b)
{
    double min = fabs(a.i - b.i);
    double max = fabs(a.j - b.j);
    if (min > max)
    {
        double temp = min;
        min = max;
        max = temp;
    }
    return ((M_SQRT2 - 1.0) * min + max);
}

double Dstar::radiusFromStart(const state& start, const state& b)
{
    return std::max(fabs(start.i - b.i), fabs(start.j - b.j));
}

/* int Dstar::computeShortestPath()
 * --------------------------
 * As per [S. Koenig, 2002] except for 2 main modifications:
 * 1. We stop planning after a number of steps, 'maxsteps' we do this
 *    because this algorithm can plan forever if the start is
 *    surrounded by obstacles.
 * 2. We lazily remove states from the open list so we never have to
 *    iterate through it.
 */
int Dstar::computeShortestPath()
{
    list<state> s;
    list<state> s2;
    list<state>::iterator i;
    double g_old;
    double min_over_succ;
    // std::cout << "openList empty check" << std::endl;
    if (openList.empty())
        return 1;
    // std::cout << "openList not empty" << std::endl;
    int k = 0;
    // line 10
    while ((!openList.empty()) &&
        ((openList.top() < (s_start = calculateKey(s_start))) ||
           (getRHS(s_start) > getG(s_start))))
    {
        // std::cout << "starting while loop" << std::endl;

        if (k++ > maxSteps)
        {
            fprintf(stderr, "At maxsteps\n");
            return -1;
        }
        // line 11
        state u;
        // lazy remove
        while (1)
        {
            if (openList.empty())
                return 1;
            u = openList.top();
            // std::cout << " in lazy remove with state: " << u << " g " << getG(u) << " and rhs" << getRHS(u) << std::endl;
            double cur_key = keyHashCode(u);
            // std::cout << "cur_key: " << cur_key << " openHash " << openHash[u] << std::endl;
            if (!isStateWithKeyInOpenList(u)) {
                openList.pop();
                // std::cout << "decided state is invalid, popped" << std::endl;
                continue;
            }
            // bool test = (getRHS(s_start) != getG(s_start));
            // std::cout << "test: " << test << std::endl;
            // if (!(u < s_start) && (!test))
            //     return 2;
            break;
        }
        // std::cout << "finished lazy remove, starting with state: " << u << std::endl;

        // line 12
        state k_old = u;
        // line 13
        state k_new = calculateKey(u);
        // line 14
        if (k_old < k_new)
        { // u is out of date
            // std::cout << "u is out of date" << std::endl;
            // line 15
            removeOpen(u);
            insertOpen(k_new);
        }
        // line 16
        else if (getG(u) > getRHS(u))
        { // needs update (got better)
            // std::cout << "needs update (got better)" << std::endl;
            // line 17
            setG(u, getRHS(u));
            // line 18
            removeOpen(u);
            // line 19
            getPred(u, s);
            for (i = s.begin(); i != s.end(); i++)
            {
                // line 20
                setRHS(*i, std::min(getRHS(*i), cost(*i, u) + getG(u)));
                // line 21
                updateVertex(*i);
            }
        }
        // line 22
        else
        { // g <= rhs, state has got worse
            // std::cout << "g <= rhs, state has got worse" << std::endl;
            // line 23
            g_old = getG(u);
            // line 24
            setG(u, std::numeric_limits<double>::infinity());
            //line 25
            getPred(u, s);
            s.push_back(u);
            for (i = s.begin(); i != s.end(); i++)
            {
                // line 26
                if (close(getRHS(*i), cost(*i, u) + g_old)) {
                    // line 27
                    if (*i != s_goal) {
                        getPred(*i, s2); // getPred == getSucc for undirected graphs
                        min_over_succ = std::numeric_limits<double>::infinity();
                        for (auto j = s2.begin(); j != s2.end(); j++) {
                            min_over_succ = std::min(min_over_succ, cost(*i, *j) + getG(*j));
                        }
                        setRHS(*i, min_over_succ);
                    }
                }
                // line 28
                updateVertex(*i);
            }
        }
    }
    return 0;
}

/* bool Dstar::close(double x, double y)
 * --------------------------
 * Returns true if x and y are within 10E-5, false otherwise
 */
bool Dstar::close(double x, double y)
{

    if (isinf(x) && isinf(y))
        return true;
    return (fabs(x - y) < 0.00001);
}

/* void Dstar::updateVertex(state u)
 * --------------------------
 * As per [S. Koenig, 2002]
 */
void Dstar::updateVertex(state u)
{
    // line 07
    bool g_is_rhs = close(getG(u), getRHS(u));
    bool is_in_open = isStateInOpenList(u);
    // std::cout << " in updateVertex with state: " << u << " g_is_rhs: " << g_is_rhs << " is_in_open: " << is_in_open << " rhs/g are " << getRHS(u) << "/" << getG(u) << std::endl;
    if (!g_is_rhs && is_in_open) {
        removeOpen(u);
        insertOpen(calculateKey(u));
    } else if (!g_is_rhs && !is_in_open) {
        // line 08
        insertOpen(calculateKey(u));
    } else if (g_is_rhs && is_in_open) {
        // line 09
        removeOpen(u);
    }
}

/* void Dstar::insert(state u)
 * --------------------------
 * Inserts state u into openList and openHash.
 */
void Dstar::insertOpen(state u)
{
    ds_oh::iterator cur;
    float csum;

    cur = openHash.find(u);
    csum = keyHashCode(u);

    // std::cout << "inserting state: " << u << " with csum: " << csum << std::endl;
    // std::cout << "openHash size: " << openHash.size() << std::endl;

    // if the state is in the open hash table with the same key already
    if ((cur != openHash.end()) && (close(csum,cur->second))) {
        throw std::runtime_error("Inserting state already on the open list. insert shouldn't have been called.");
    }

    openHash[u] = csum;
    // std::cout << "for state " << u << " csum is " << openHash[u] << std::endl;
    openList.push(u);
}

/* void Dstar::remove(state u)
 * --------------------------
 * Removes state u from openHash. The state is removed from the
 * openList lazilily (in replan) to save computation.
 */
void Dstar::removeOpen(const state& u)
{
    ds_oh::iterator cur = openHash.find(u);
    if (cur == openHash.end())
        throw std::runtime_error("Removing state not on the open list");
    openHash.erase(cur);
}

/* double Dstar::trueDist(state a, state b)
 * --------------------------
 * Euclidean cost between state a and state b.
 */
double Dstar::trueDist(const state& a, const state& b)
{

    float i = a.i - b.i;
    float j = a.j - b.j;
    return sqrt(i * i + j * j);
}

/* double Dstar::heuristic(state a, state b)
 * --------------------------
 * Pretty self explanitory, the heristic we use is the 8-way distance
 */
double Dstar::heuristic(const state& a, const state& b)
{   
    // return 0.0;
    if (diag_cost_scale > 1) {
        return eightCondist(a, b);
    } else {
        return radiusFromStart(a, b);
    }    
}

/* state Dstar::calculateKey(state u)
 * --------------------------
 * As per [S. Koenig, 2002]
 */
state Dstar::calculateKey(state u)
{
    // line 01
    double val = fmin(getRHS(u), getG(u));
    u.k.first = val + heuristic(u, s_start) + k_m;
    if (u.k.first > MAX_FIRST_KEY_VALUE) {
        // to my knowledge, this should only ever happen to the start state
        if (u != s_start) { 
            throw std::runtime_error("k.first > MAX_FIRST_KEY_VALUE");
        }
    }
    u.k.second = val;

    return u;
}

/* double Dstar::cost(state a, state b)
 * --------------------------
 * Returns the cost of moving from state a to state b. This could be
 * either the cost of moving off state a or onto state b, we went with
 * the former. This is also the 8-way cost.
 */
double Dstar::cost(const state& a, const state& b)
{
    if (a == b)
        return 0;
    int id = fabs(a.i - b.i);
    int jd = fabs(a.j - b.j);
    double scale = 1;

    if (id + jd > 1)
        scale = diag_cost_scale;
    // NOTE: if this changes, make sure to update line 43-45
    return scale * getMapCell(b);
}

void Dstar::updateCells(const py::array_t<int>& indexes, const py::array_t<double>& values) {
    // std::cout << "############ updateCells called" << std::endl;
    auto indexes_buffer = indexes.request();
    auto values_buffer = values.request();
    auto indexes_r = indexes.unchecked<2>();
    auto values_r = values.unchecked<1>();
    if (indexes_buffer.shape[0] != values_buffer.shape[0]) {
        throw std::runtime_error("indexes and values must have the same length");
    }
    // std::cout << "starting to update cells" << std::endl;
    for (py::ssize_t i = 0; i < indexes_buffer.shape[0]; i++)
    {
        int index_i = indexes_r(i, 0);
        int index_j = indexes_r(i, 1);
        double value = values_r(i);
        // std::cout << "updating cell: " << index_i << " " << index_j << " with value: " << value << std::endl;
        updateCell(index_i, index_j, value);
    }
    // std::cout << "finished updating cells" << std::endl;
}
/* void Dstar::updateCell(int i, int j, double val)
 * --------------------------
 * As per [S. Koenig, 2002]
 */
void Dstar::updateCell(int i, int j, double val)
{

    state v;

    v.i = i;
    v.j = j;

    double old_cell_value = getMapCell(v);
    if (close(old_cell_value, val)) {
        return;
    }

    if ((v == s_start) || (v == s_goal)) {
        return;
    }

    // line 38
    // std::cout << "increasint k_m from " << k_m << " by " << heuristic(s_last, s_start) << std::endl;
    k_m = k_m + heuristic(s_last, s_start); // no update if s_last hasn't changed
    // line 39
    s_last = s_start;

    // line 42
    setMapCell(v, val);
    // line 40
    // since cost of a->b is the cost of cell b, only the directed edges x -> v need to be updated
    list<state> pred;
    getPred(v, pred);
    list<state>::iterator u;
    double c_old, c_new;
    for (u = pred.begin(); u != pred.end(); u++)
    {
        // std::cout << "processing pred: " << *u << std::endl;
        // line 41
        c_new = cost(*u, v);
        c_old = old_cell_value;
        if (abs(u->i - v.i) + abs(u->j - v.j) > 1) {
            c_new *= diag_cost_scale;
        }
        // std::cout << "c_old: " << c_old << " c_new: " << c_new << std::endl;
        // line 43
        if (c_old > c_new) {
            // line 44
            // std::cout << "c_old is greater than c_new" << std::endl;
            setRHS(*u, std::min(getRHS(*u), c_new + getG(v)));
        // line 45
        } else if (close(getRHS(*u), c_old + getG(v))) {
            // std::cout << "rhs is close to c_old + g" << std::endl;
            // line 46
            if (*u != s_goal) {
                double min_over_succ = std::numeric_limits<double>::infinity();
                list<state> succ;
                getPred(*u, succ);
                for (auto sp = succ.begin(); sp != succ.end(); sp++) {
                    min_over_succ = std::min(min_over_succ, cost(*u, *sp) + getG(*sp));
                }
                setRHS(*u, min_over_succ);
            }
        }
        updateVertex(*u);
    }

    edge_cost_changed = true;
}
/* void Dstar::getPred(state u,list<state> &s)
 * --------------------------
 * Returns a list of all the predecessor states for state u. Since
 * this is for an 8-way connected graph the list contains all the
 * neighbours for state u. Occupied neighbours are not added to the
 * list.
 * NOTE: changes the key for the state u
 */
void Dstar::getPred(state u, list<state> &s)
{

    s.clear();
    u.k.first = -1;
    u.k.second = -1;

    u.i += 1;
    if (!occupied(u))
        s.push_front(u);
    u.j += 1;
    if (!occupied(u))
        s.push_front(u);
    u.i -= 1;
    if (!occupied(u))
        s.push_front(u);
    u.i -= 1;
    if (!occupied(u))
        s.push_front(u);
    u.j -= 1;
    if (!occupied(u))
        s.push_front(u);
    u.j -= 1;
    if (!occupied(u))
        s.push_front(u);
    u.i += 1;
    if (!occupied(u))
        s.push_front(u);
    u.i += 1;
    if (!occupied(u))
        s.push_front(u);
}

/* void Dstar::updateStart(int i, int j)
 * --------------------------
 * Update the position of the robot, this does not force a replan.
 */
void Dstar::updateStart(int i, int j)
{
    if (!init_called)
    {
        throw std::runtime_error("init must be called before updateStart");
    }

    // line 35
    s_start.i = i;
    s_start.j = j;

    if (!isStateInMap(s_start))
    {
        throw std::runtime_error("Start is outside of the map");
    }
}

/* void Dstar::updateGoal(int x, int y)
 * --------------------------
 * This is somewhat of a hack, to change the position of the goal we
 * first save all of the non-empty on the map, clear the map, move the
 * goal, and re-add all of non-empty cells. Since most of these cells
 * are not between the start and goal this does not seem to hurt
 * performance too much. Also it free's up a good deal of memory we
 * likely no longer use.
 */
// void Dstar::updateGoal(int iArg, int jArg)
// {

//     list<pair<ipoint2, double>> toAdd;
//     pair<ipoint2, double> tp;

//     ds_ch::iterator iter;
//     list<pair<ipoint2, double>>::iterator kk;

//     for (iter = cellHash.begin(); iter != cellHash.end(); iter++)
//     {
//         if (!close(iter->second.cost, C1))
//         {
//             tp.first.i = iter->first.i;
//             tp.first.j = iter->first.j;
//             tp.second = iter->second.cost;
//             toAdd.push_back(tp);
//         }
//     }

//     cellHash.clear();
//     openHash.clear();

//     while (!openList.empty())
//         openList.pop();

//     k_m = 0;

//     s_goal.i = iArg;
//     s_goal.j = jArg;

//     cellInfo tmp;
//     tmp.g = tmp.rhs = 0;
//     tmp.cost = C1;

//     cellHash[s_goal] = tmp;

//     tmp.g = tmp.rhs = heuristic(s_start, s_goal);
//     tmp.cost = C1;
//     cellHash[s_start] = tmp;
//     s_start = calculateKey(s_start);

//     s_last = s_start;

//     for (kk = toAdd.begin(); kk != toAdd.end(); kk++)
//     {
//         updateCell(kk->first.i, kk->first.j, kk->second);
//     }
// }

py::list Dstar::getGValues()
{
    py::list output;
    for (auto &s : cellHash)
    {
        output.append(py::make_tuple(s.first.i, s.first.j, s.second.g));
    }
    return output;
}
py::list Dstar::getRHSValues()
{
    py::list output;
    for (auto &s : cellHash)
    {
        output.append(py::make_tuple(s.first.i, s.first.j, s.second.rhs));
    }
    return output;
}
py::list Dstar::getKeys()
{
    std::vector<state> states;
    while (!openList.empty())
    {
        state tmp = openList.top();
        if (!isStateWithKeyInOpenList(tmp)) {
            openList.pop();
            continue;
        }

        states.push_back(tmp);
        removeOpen(tmp);
        openList.pop();
    }
    py::list output;
    for (auto &s : states)
    {
        output.append(py::make_tuple(s.i, s.j, s.k.first, s.k.second));
    }
    for (auto &s : states)
    {
        insertOpen(s);
    }
    return output;
}

/* bool Dstar::replan()
 * --------------------------
 * Updates the costs for all cells and computes the shortest path to
 * goal. Returns true if a path is found, false otherwise. The path is
 * computed by doing a greedy search over the cost+g values in each
 * cells. In order to get around the problem of the robot taking a
 * path that is near a 45 degree angle to goal we break ties based on
 *  the metric euclidean(state, goal) + euclidean(state,start).
 */
bool Dstar::replan()
{
    if (!init_called)
    {
        throw std::runtime_error("init must be called before replan");
    }
    path.clear();
    // std::cout << "starting replan" << std::endl;
    // line 31 / 48
    if (edge_cost_changed) {

        int res = computeShortestPath();
        printf("res: %d ols: %zu ohs: %zu tk: [%f %f] sk: [%f %f] sgr: (%f,%f)\n",res,openList.size(),openHash.size(),openList.top().k.first,openList.top().k.second, s_start.k.first, s_start.k.second,getRHS(s_start),getG(s_start));
        edge_cost_changed = false;
        if (res < 0)
        {
            fprintf(stderr, "NO PATH TO GOAL\n");
            return false;
        }
    }


    // line 33
    if (isinf(getRHS(s_start)))
    {
        fprintf(stderr, "NO PATH TO GOAL\n");
        return false;
    }

    // build a path by greedily following successors
    // sort of line 34
    list<state> n;
    list<state>::iterator i;
    state cur = s_start;
    int max_iters = 10000000000;
    // std::cout << "goal: " << s_goal << " has g value " << getG(s_goal) << " and rhs " << getRHS(s_goal) << std::endl;
    while (cur != s_goal)
    {
        path.push_back(cur);
        // std::cout << " added cur: " << cur << " with g value " << getG(cur) <<  std::endl;
        getPred(cur, n); // getPred == getSucc for undirected graphs
        if (n.empty())
        {
            fprintf(stderr, "NO PATH TO GOAL\n");
            return false;
        }
        double cmin = std::numeric_limits<double>::infinity();
        double tmin = std::numeric_limits<double>::infinity();
        state smin;
        for (i = n.begin(); i != n.end(); i++)
        {
            double val = cost(cur, *i);
            // for breaking ties
            double val2 = trueDist(*i, s_goal) + trueDist(s_start, *i); // (Euclidean) cost to goal + cost to pred
            val += getG(*i);

            if (close(val, cmin))
            {
                if (tmin > val2)
                {
                    tmin = val2;
                    cmin = val;
                    smin = *i;
                }
            }
            else if (val < cmin)
            {
                tmin = val2;
                cmin = val;
                smin = *i;
            }
        }
        n.clear();
        cur = smin;
        // std::cout << "cur: " << cur << " with g value " << getG(cur) <<  std::endl;
        if (max_iters-- < 0)
        {
            fprintf(stderr, "NO PATH TO GOAL\n");
            return false;
        }
    }
    path.push_back(s_goal);
    return true;
}
