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
Dstar::Dstar(const py::array_t<double, py::array::c_style>& mapArg, unsigned long maxStepsArg, bool scale_diag_cost)
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
    map_height = buffer.shape[0];
    map_width = buffer.shape[1];
    this->map = std::vector<double>(map_width * map_height);
    auto r2 = mapArg.unchecked<2>();
    if (r2.shape(0) > MAX_MAP_DIM || r2.shape(1) > MAX_MAP_DIM)
    {
        std::throw_with_nested(std::runtime_error("Map is too large"));
    }
    for (py::ssize_t i = 0; i < r2.shape(0); i++)
    {
        for (py::ssize_t j = 0; j < r2.shape(1); j++)
        {
            if (r2(i, j) < 1) {
                throw std::runtime_error("All cells must have a cost >= 1");
            }
            this->map[i * map_width + j] = r2(i, j);
        }
    }

    this->g_values = std::vector<double>(map_width * map_height, std::numeric_limits<double>::infinity());
    this->rhs_values = std::vector<double>(map_width * map_height, std::numeric_limits<double>::infinity());
    this->firstKey = std::vector<double>(map_width * map_height, std::numeric_limits<double>::infinity());
    this->secondKey = std::vector<double>(map_width * map_height, std::numeric_limits<double>::infinity());

}
double Dstar::getMapCell(const state& s){
    return this->map[s.i * map_width + s.j];
}
void Dstar::setMapCell(const state& s, double val){
    this->map[s.i * map_width + s.j] = val;
}

void Dstar::printMap() {
    for (auto i = 0; i < map_height; i++)
    {
        for (auto j = 0; j < map_width; j++)
        {
            std::cout << this->map[i * map_width + j] << " ";
        }
        std::cout << std::endl;
    }
}


bool Dstar::isStateInOpenList(const state& u)
{
    // if inf, this state hasn't been added to the open list yet

    if (firstKey[u.i * map_width + u.j] == std::numeric_limits<double>::infinity()) {
        return false;
    }
    return true;
}
bool Dstar::isStateWithKeyInOpenList(const state& u) {
    if (!isStateInOpenList(u)) {
        return false;
    }
    return close(firstKey[u.i * map_width + u.j], u.k.first) && close(secondKey[u.i * map_width + u.j], u.k.second);
}

bool Dstar::isStateInMap(const state& u) {
    return u.i >= 0 && u.j >= 0 && u.i < map_height && u.j < map_width;
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
    return false; //getMapCell(u) == std::numeric_limits<double>::infinity();
}

/* void Dstar::init(int sX, int sY, int gX, int gY)
 * --------------------------
 * Init dstar with start and goal coordinates, rest is as per
 * [S. Koenig, 2002]
 */
void Dstar::init(int sI, int sJ, int gI, int gJ)
{
    // reset g and rhs values
    // line 06
    std::fill(g_values.begin(), g_values.end(), std::numeric_limits<double>::infinity());
    std::fill(rhs_values.begin(), rhs_values.end(), std::numeric_limits<double>::infinity());
    path.clear();
    std::fill(firstKey.begin(), firstKey.end(), std::numeric_limits<double>::infinity());
    std::fill(secondKey.begin(), secondKey.end(), std::numeric_limits<double>::infinity());
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

    setRHS(s_goal, 0); // line 05

    insertOpen(calculateKey(s_goal)); // line 06

    init_called = true;
    edge_cost_changed = true;
}

/* double Dstar::getG(state u)
 * --------------------------
 * Returns the G value for state u.
 */
double Dstar::getG(const state& u)
{
    return g_values[u.i * map_width + u.j];
}

/* double Dstar::getRHS(state u)
 * --------------------------
 * Returns the rhs value for state u.
 */
double Dstar::getRHS(const state& u)
{
    // TODO: remove. Why is this here? 
    if (u == s_goal)
        return 0;
    return rhs_values[u.i * map_width + u.j];
}

/* void Dstar::setG(state u, double g)
 * --------------------------
 * Sets the G value for state u
 */
void Dstar::setG(const state& u, double g)
{
    g_values[u.i * map_width + u.j] = g;
}

/* void Dstar::setRHS(state u, double rhs)
 * --------------------------
 * Sets the rhs value for state u
 */
void Dstar::setRHS(const state& u, double rhs)
{
    rhs_values[u.i * map_width + u.j] = rhs;
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
    std::vector<state> s;
    std::vector<state> s2;
    double g_old;
    double min_over_succ;
    // std::cout << "openList empty check" << std::endl;
    if (openList.empty())
        return 1;
    // std::cout << "openList not empty" << std::endl;
    unsigned long k = 0;

    while (!openList.empty()) 
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
            if (!isStateWithKeyInOpenList(u)) {
                openList.pop();
                // std::cout << "decided state is invalid, popped" << std::endl;
                continue;
            }
            break;
        }
        // std::cout << "finished lazy remove, starting with state: " << u << std::endl;

        // recheck while loop conditions now that junk has been removed
        // line 10
        if (!((openList.top() < calculateKey(s_start)) || (getRHS(s_start) > getG(s_start)))) {
            break;
        }

        // line 12
        state k_old = u;
        // line 13
        state k_new = calculateKey(u);
        // line 14
        if (k_old < k_new)
        { 
            // std::cout << "\tu is out of date" << std::endl;
            // std::cout << "\tupdating key from " << u << " to " << k_new << " which has comp result " << (k_old < k_new) << std::endl;
            // line 15
            removeOpen(u);
            insertOpen(k_new);
        }
        // line 16
        else if (getG(u) > getRHS(u))
        { 
            // std::cout << "\texpanding state " << u << std::endl;
            // line 17
            setG(u, getRHS(u));
            // line 18
            removeOpen(u);
            // line 19
            getPred(u, s);
            for (auto i = s.begin(); i != s.end(); i++)
            {
                // line 20
                setRHS(*i, std::min(getRHS(*i), cost(*i, u) + getG(u)));
                // line 21
                updateVertex(*i);
                // std::cout << "\t\t" << *i << " with rhs " << std::min(getRHS(*i), cost(*i, u) + getG(u)) << std::endl;
            }
        }
        // line 22
        else
        { // g <= rhs, state has got worse
            // std::cout << "\tg <= rhs, state has got worse. Setting g to inf and updating self and pred" << std::endl;
            // line 23
            g_old = getG(u);
            // line 24
            setG(u, std::numeric_limits<double>::infinity());
            //line 25
            getPred(u, s);
            s.push_back(u);
            for (auto i = s.begin(); i != s.end(); i++)
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
                // std::cout << "\t\t" << *i << " with rhs " << getRHS(*i) << std::endl;
                updateVertex(*i);
            }
        }
    }
    return 0;
}

bool Dstar::close(double x, double y, double eps, double abs_th)
{
    assert(std::numeric_limits<double>::epsilon() < eps);
    assert(eps < 1.f);
    if (x == y) return true;
    auto diff = std::abs(x-y);
    auto norm = std::min(std::abs(x+y), std::numeric_limits<double>::max());
    return diff < std::max(abs_th, eps * norm);
}

/* void Dstar::updateVertex(state u)
 * --------------------------
 * As per [S. Koenig, 2002]
 */
void Dstar::updateVertex(const state& u)
{
    // line 07
    bool g_is_rhs = close(getG(u), getRHS(u));
    bool is_in_open = isStateInOpenList(u);
    // std::cout << "\t in updateVertex with state: " << calculateKey(u) << " g_is_rhs: " << g_is_rhs << " is_in_open: " << is_in_open << " rhs/g are " << getRHS(u) << "/" << getG(u) << std::endl;
    if (!g_is_rhs && is_in_open) {
        // don't re-insert if the key hasn't changed
        if (!isStateWithKeyInOpenList(calculateKey(u))) {
            removeOpen(u);
            insertOpen(calculateKey(u));
            // std::cout << "\t\t inserted state " << calculateKey(u) << std::endl;
        } else {
            // std::cout << "\t\tDidn't re-insert state" << calculateKey(u) << std::endl;
        }
    } else if (!g_is_rhs && !is_in_open) {
        // line 08
        insertOpen(calculateKey(u));
    } else if (g_is_rhs && is_in_open) {
        // line 09
        removeOpen(u);
    }
}

void Dstar::insertOpen(const state& u)
{

    if (isStateWithKeyInOpenList(u)) {
        throw std::runtime_error("Inserting state already on the open list. insert shouldn't have been called.");
    }
    if (u.k.first == std::numeric_limits<double>::infinity() || u.k.second == std::numeric_limits<double>::infinity() ) {
        throw std::runtime_error("Inserting state with key of infinity!");
    }
    firstKey[u.i * map_width + u.j] = u.k.first;
    secondKey[u.i * map_width + u.j] = u.k.second;

    openList.push(u);
}

void Dstar::removeOpen(const state& u)
{
    if (!isStateInOpenList(u)) {
        throw std::runtime_error("Removing state not on the open list");
    }
    firstKey[u.i * map_width + u.j] = std::numeric_limits<double>::infinity();
    secondKey[u.i * map_width + u.j] = std::numeric_limits<double>::infinity();

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
state Dstar::calculateKey(const state& u)
{
    // line 01
    state s(u);
    double val = fmin(getRHS(s), getG(s));
    s.k.first = val + heuristic(s, s_start) + k_m;
    s.k.second = val;

    return s;
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
        // std::cout << "########## updating cell: " << index_i << " " << index_j << " with value: " << value << std::endl;
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
    if (old_cell_value == val) {
        return;
    }
    edge_cost_changed = true;

    std::vector<state> pred;
    getPred(v, pred);

    std::vector<double> old_edge_costs;
    for (auto u = pred.begin(); u != pred.end(); u++) {
        old_edge_costs.push_back(cost(*u, v));
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
    for (unsigned int index = 0; index != pred.size(); index++) 
    {
        // std::cout << "PROCESSING EDGE: " << u->i << "," << u->j << "->" << v.i << "," << v.j << std::endl;
        // line 41
        const double c_old = old_edge_costs[index];
        const state u = pred[index];
        const double c_new = cost(u, v);
        // std::cout << "\tc_old: " << c_old << " c_new: " << c_new << std::endl;
        // line 43
        if (c_old > c_new) {
            // line 44
            // std::cout << "\tc_old is greater than c_new. RHS is updating to " << std::min(getRHS(u), c_new + getG(v)) << " from " << getRHS(u) << std::endl;
            setRHS(u, std::min(getRHS(u), c_new + getG(v)));
        // line 45
        } else if (getRHS(u) == c_old + getG(v)) {
            // std::cout << "\trhs is close to c_old + g" << std::endl;
            // line 46
            if (u != s_goal) {
                double min_over_succ = std::numeric_limits<double>::infinity();
                std::vector<state> succ;
                getPred(u, succ);
                for (auto sp = succ.begin(); sp != succ.end(); sp++) {
                    min_over_succ = std::min(min_over_succ, cost(u, *sp) + getG(*sp));
                    // std::cout << "\t\t succ " << *sp << " with cost " << cost(*u, *sp) << " and g " << getG(*sp) << std::endl;
                }
                // std::cout << "\tsetting rhs to min_over_succ: " << min_over_succ << " from " << getRHS(*u) << std::endl;
                setRHS(u, min_over_succ);
            }
        } else {
            // std::cout << "\t no action taken" << std::endl;
        }
        updateVertex(u);
    }

}
/* void Dstar::getPred(state u,list<state> &s)
 * --------------------------
 * Returns a list of all the predecessor states for state u. Since
 * this is for an 8-way connected graph the list contains all the
 * neighbours for state u. Occupied neighbours are not added to the
 * list.
 */
void Dstar::getPred(const state& uArg, std::vector<state> &s)
{
    state u(uArg);
    s.clear();
    u.k.first = -1;
    u.k.second = -1;

    u.i += 1;
    // std::cout << "checking state " << u << "..";
    if (!occupied(u)) {
        s.push_back(u);
        // std::cout << "valid" << std::endl;
    } else {
        // std::cout << "INVALID" << std::endl;
    }
    u.j += 1;
    // std::cout << "checking state " << u << "..";
    if (!occupied(u)) {
        s.push_back(u);
        // std::cout << "valid" << std::endl;
    } else {
        // std::cout << "INVALID" << std::endl;
    }
    u.i -= 1;
    // std::cout << "checking state " << u << "..";
    if (!occupied(u)) {
        s.push_back(u);
        // std::cout << "valid" << std::endl;
    } else {
        // std::cout << "INVALID" << std::endl;
    }
    u.i -= 1;
    // std::cout << "checking state " << u << "..";
    if (!occupied(u)) {
        s.push_back(u);
        // std::cout << "valid" << std::endl;
    } else {
        // std::cout << "INVALID" << std::endl;
    }
    u.j -= 1;
    // std::cout << "checking state " << u << "..";
    if (!occupied(u)) {
        s.push_back(u);
        // std::cout << "valid" << std::endl;
    } else {
        // std::cout << "INVALID" << std::endl;
    }
    u.j -= 1;
    // std::cout << "checking state " << u << "..";
    if (!occupied(u)) { 
        s.push_back(u);
        // std::cout << "valid" << std::endl;
    } else {
        // std::cout << "INVALID" << std::endl;
    }
    u.i += 1;
    // std::cout << "checking state " << u << "..";
    if (!occupied(u)) { 
        s.push_back(u);
        // std::cout << "valid" << std::endl;
    } else {
        // std::cout << "INVALID" << std::endl;
    }
    u.i += 1;
    // std::cout << "checking state " << u << "..";
    if (!occupied(u)) {
        s.push_back(u);
        // std::cout << "valid" << std::endl;
    } else {
        // std::cout << "INVALID" << std::endl;
    }
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

py::list Dstar::getMap() {
    py::list output;
    for (auto i=map.begin(); i!=map.end(); i++){
        output.append(*i);
    }
    return output;
}
py::list Dstar::getGValues()
{
    py::list output;
    for (auto i = g_values.begin(); i != g_values.end(); i++)
    {
        output.append(*i);
    }
    return output;
}
py::list Dstar::getRHSValues()
{
    py::list output;
    for (auto i = rhs_values.begin(); i != rhs_values.end(); i++)
    {
        output.append(*i);
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
        printf("res: %d ols: %zu tk: [%f %f] sk: [%f %f] sgr: (%f,%f)\n",res,openList.size(),openList.top().k.first,openList.top().k.second, s_start.k.first, s_start.k.second,getRHS(s_start),getG(s_start));
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
        fprintf(stderr, "NO PATH TO GOAL, rhs of s_start is inf!\n");
        return false;
    }

    // build a path by greedily following successors
    // sort of line 34
    std::vector<state> n;
    state cur = s_start;
    int max_iters = 10000000;
    // std::cout << "goal: " << s_goal << " has g value " << getG(s_goal) << " and rhs " << getRHS(s_goal) << std::endl;
    while (cur != s_goal)
    {
        path.push_back(cur);
        // std::cout << " added cur: " << cur << " with g value " << getG(cur) <<  std::endl;
        getPred(cur, n); // getPred == getSucc for undirected graphs
        if (n.empty())
        {
            fprintf(stderr, "NO PATH TO GOAL, no predesessors of a state on path\n");
            return false;
        }
        double cmin = std::numeric_limits<double>::infinity();
        double tmin = std::numeric_limits<double>::infinity();
        state smin;
        for (auto i = n.begin(); i != n.end(); i++)
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
        // if (cur == s_start) {
        //     std::cout << "found cost to goal of " << cmin << std::endl;
        // }
        n.clear();
        cur = smin;
        // std::cout << "cur: " << cur << " with g value " << getG(cur) <<  std::endl;
        if (max_iters-- < 0)
        {
            fprintf(stderr, "NO PATH TO GOAL, max iters hit while extracting path\n");
            return false;
        }
    }
    path.push_back(s_goal);
    return true;
}
