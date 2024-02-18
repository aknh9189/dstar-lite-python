/* Dstar.cpp
 * FROM https://github.com/ArekSredzki/dstar-lite/tree/master
 * James Neufeld (neufeld@cs.ualberta.ca)
 * Compilation fixed by Arek Sredzki (arek@sredzki.com)
 */

#include "dstar.h"


std::ostream& operator<<(std::ostream &os, const state& s)
{
    os << "i: "; //<< s.i << " j: " << s.j << " k: " << s.k.first << " " << s.k.second;
    return os;
}

/* void Dstar::Dstar()
 * --------------------------
 * mapArg must be a double numpy 2d array
 * Each cell contains the cost of moving through it. So the heuristic is valid all cells must have a cost >= 1
 * Untraversable cells have a cost of infinity
 */
Dstar::Dstar(const py::array_t<double, py::array::c_style>& mapArg)
{
    maxSteps = 80; // node expansions before we give up
    init_called = false;

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
    return static_cast<float>(u.k.first + MAX_FIRST_KEY_VALUE * u.k.second);
}

/* bool Dstar::isValid(state u)
 * --------------------------
 * Returns true if state u is on the open list or not by checking if
 * it is in the hash table.
 */
bool Dstar::isValid(const state& u)
{

    ds_oh::iterator cur = openHash.find(u);
    if (cur == openHash.end())
        return false;
    if (!close(keyHashCode(u), cur->second))
        return false;
    return true;
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
    if (!isStateInMap(s_start) || !isStateInMap(s_goal))
    {
        throw std::runtime_error("Start or goal is outside of the map");
    }

    cellInfo tmp;
    tmp.rhs = 0; // line 05
    cellHash[s_goal] = tmp;

    openList.push(calculateKey(s_goal)); // line 06

    init_called = true;
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

    if (cellHash.find(u) == cellHash.end())
        return heuristic(u, s_goal);
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

    if (cellHash.find(u) == cellHash.end())
        return heuristic(u, s_goal);
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
    list<state>::iterator i;
    std::cout << "openList empty check" << std::endl;

    if (openList.empty())
        return 1;
    std::cout << "openList not empty" << std::endl;
    int k = 0;
    while (((!openList.empty()) &&
               (openList.top() < (s_start = calculateKey(s_start)))) ||
           (getRHS(s_start) != getG(s_start)))
    {
        std::cout << "starting while loop" << std::endl;

        if (k++ > maxSteps)
        {
            fprintf(stderr, "At maxsteps\n");
            return -1;
        }

        std::cout << "starting while loop of compute shortest path" << std::endl;

        state u;

        bool test = (getRHS(s_start) != getG(s_start));
        std::cout << "test: " << test << std::endl;

        // lazy remove
        while (1)
        {
            if (openList.empty())
                return 1;
            u = openList.top();
            openList.pop();

            if (!isValid(u))
                continue;
            if (!(u < s_start) && (!test))
                return 2;
            break;
        }
        std::cout << "finished lazy remove, starting with state: " << u << std::endl;

        ds_oh::iterator cur = openHash.find(u);
        openHash.erase(cur);

        state k_old = u;

        if (k_old < calculateKey(u))
        { // u is out of date
            std::cout << "u is out of date" << std::endl;
            insert(u);
        }
        else if (getG(u) > getRHS(u))
        { // needs update (got better)
            std::cout << "needs update (got better)" << std::endl;
            setG(u, getRHS(u));
            getPred(u, s);
            for (i = s.begin(); i != s.end(); i++)
            {
                updateVertex(*i);
            }
        }
        else
        { // g <= rhs, state has got worse
            std::cout << "g <= rhs, state has got worse" << std::endl;
            setG(u, std::numeric_limits<double>::infinity());
            getPred(u, s);
            for (i = s.begin(); i != s.end(); i++)
            {
                updateVertex(*i);
            }
            updateVertex(u);
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

    list<state> s;
    list<state>::iterator i;

    if (u != s_goal)
    {
        getSucc(u, s);
        double tmp = std::numeric_limits<double>::infinity();
        double tmp2;

        for (i = s.begin(); i != s.end(); i++)
        {
            tmp2 = getG(*i) + cost(u, *i);
            if (tmp2 < tmp)
                tmp = tmp2;
        }
        if (!close(getRHS(u), tmp)) {
            setRHS(u, tmp);
        }
    }

    if (!close(getG(u), getRHS(u)))
        insert(u);
}

/* void Dstar::insert(state u)
 * --------------------------
 * Inserts state u into openList and openHash.
 */
void Dstar::insert(state u)
{

    ds_oh::iterator cur;
    float csum;

    u = calculateKey(u);
    cur = openHash.find(u);
    csum = keyHashCode(u);
    // return if cell is already in list. TODO: this should be
    // uncommented except it introduces a bug, I suspect that there is a
    // bug somewhere else and having duplicates in the openList queue
    // hides the problem...
    // if ((cur != openHash.end()) && (close(csum,cur->second))) return;

    openHash[u] = csum;
    openList.push(u);
}

/* void Dstar::remove(state u)
 * --------------------------
 * Removes state u from openHash. The state is removed from the
 * openList lazilily (in replan) to save computation.
 */
void Dstar::remove(const state& u)
{

    ds_oh::iterator cur = openHash.find(u);
    if (cur == openHash.end())
        return;
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
    return 0.0;
    return eightCondist(a, b);
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
        throw std::runtime_error("k.first > MAX_FIRST_KEY_VALUE");
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

    int id = fabs(a.i - b.i);
    int jd = fabs(a.j - b.j);
    double scale = 1;

    if (id + jd > 1)
        scale = M_SQRT2;
    return scale * getMapCell(a);
}
/* void Dstar::updateCell(int i, int j, double val)
 * --------------------------
 * As per [S. Koenig, 2002]
 */
void Dstar::updateCell(int i, int j, double val)
{

    state u;

    u.i = i;
    u.j = j;

    if ((u == s_start) || (u == s_goal))
        return;

    makeNewCell(u);
    setMapCell(u, val);
    updateVertex(u);
}

/* void Dstar::getSucc(state u,list<state> &s)
 * --------------------------
 * Returns a list of successor states for state u, since this is an
 * 8-way graph this list contains all of a cells neighbours. Unless
 * the cell is occupied in which case it has no successors.
 */
void Dstar::getSucc(state u, list<state> &s)
{

    s.clear();
    u.k.first = -1;
    u.k.second = -1;

    if (occupied(u))
        return;

    u.i += 1;
    s.push_front(u);
    u.j += 1;
    s.push_front(u);
    u.i -= 1;
    s.push_front(u);
    u.i -= 1;
    s.push_front(u);
    u.j -= 1;
    s.push_front(u);
    u.j -= 1;
    s.push_front(u);
    u.i += 1;
    s.push_front(u);
    u.i += 1;
    s.push_front(u);
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

    s_start.i = i;
    s_start.j = j;

    if (!isStateInMap(s_start))
    {
        throw std::runtime_error("Start is outside of the map");
    }

    k_m += heuristic(s_last, s_start);

    s_start = calculateKey(s_start);
    s_last = s_start;
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
    std::cout << "starting replan" << std::endl;
    int res = computeShortestPath();
    printf("res: %d ols: %zu ohs: %zu tk: [%f %f] sk: [%f %f] sgr: (%f,%f)\n",res,openList.size(),openHash.size(),openList.top().k.first,openList.top().k.second, s_start.k.first, s_start.k.second,getRHS(s_start),getG(s_start));
    if (res < 0)
    {
        fprintf(stderr, "NO PATH TO GOAL\n");
        return false;
    }
    list<state> n;
    list<state>::iterator i;

    state cur = s_start;

    if (isinf(getG(s_start)))
    {
        fprintf(stderr, "NO PATH TO GOAL\n");
        return false;
    }

    while (cur != s_goal)
    {

        path.push_back(cur);
        getSucc(cur, n);

        if (n.empty())
        {
            fprintf(stderr, "NO PATH TO GOAL\n");
            return false;
        }

        double cmin = std::numeric_limits<double>::infinity();
        double tmin;
        state smin;

        for (i = n.begin(); i != n.end(); i++)
        {

            // if (occupied(*i)) continue;
            double val = cost(cur, *i);
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
    }
    path.push_back(s_goal);
    return true;
}
