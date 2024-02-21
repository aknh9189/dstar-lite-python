/* Dstar.h
 * FROM https://github.com/ArekSredzki/dstar-lite/tree/master
 * James Neufeld (neufeld@cs.ualberta.ca)
 * Compilation fixed by Arek Sredzki (arek@sredzki.com)
 */

#pragma once
#include <math.h>
#include <stack>
#include <queue>
#include <list>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>



#define MAX_MAP_DIM 999999 // for hash_map

using namespace std;
namespace py = pybind11;

class state
{
public:
    int i;
    int j;
    pair<double, double> k;

    bool operator==(const state &s2) const
    {
        return ((i == s2.i) && (j == s2.j));
    }

    bool operator!=(const state &s2) const
    {
        return ((i != s2.i) || (j != s2.j));
    }

    bool operator>(const state &s2) const
    {
        if (k.first - 0.000001 > s2.k.first)
            return true;
        else if (k.first < s2.k.first - 0.000001)
            return false;
        return (k.second - 0.000001) > s2.k.second;
    }

    bool operator<(const state &s2) const
    {
        if (k.first + 0.000001 < s2.k.first)
            return true;
        else if (k.first - 0.000001 > s2.k.first)
            return false;
        return (k.second + 0.000001) < s2.k.second;
    }
    friend std::ostream& operator<<(std::ostream &os, const state& s);

};

struct ipoint2 {
  int i,j;
};

struct cellInfo
{
    double g = std::numeric_limits<double>::infinity();
    double rhs = std::numeric_limits<double>::infinity();
};

class state_hash
{
public:
    size_t operator()(const state &s) const
    {
        return s.i + MAX_MAP_DIM * s.j;
    }
};

typedef priority_queue<state, vector<state>, greater<state>> ds_pq;

class Dstar
{

public:
    Dstar(const py::array_t<double, py::array::c_style>& mapArg, 
          unsigned long maxStepsArg,
          bool scale_diag_cost = true);
    void reset();
    void init(int sI, int sJ, int gI, int gJ);
    void updateStart(int i, int j);
    bool replan();

    void updateCells(const py::array_t<int>& indexes, const py::array_t<double>& values);
    void updateCell(int i, int j, double val);
    void printMap();

    py::list getPath();
    py::list getGValues();
    py::list getRHSValues();
    py::list getKeys();

private:
    double diag_cost_scale;
    bool init_called;
    bool edge_cost_changed;
    std::vector<double> map;
    int map_width, map_height;
    std::vector<state> path;

    double k_m;
    state s_start, s_goal, s_last;
    unsigned long maxSteps;

    ds_pq openList;
    // store the g and rhs values for each cell
    std::vector<double> g_values;
    std::vector<double> rhs_values;
    // store the key hashes, so we can lazy remove from the open list
    // default is inf 
    std::vector<double> firstKey;
    std::vector<double> secondKey;

    // calls init to reset structures 
    void setMap(const py::array_t<double, py::array::c_style>& mapArg); // NOTE: CAN SOMETIMES DECIDE TO SILENT COPY OR NOT SILENT COPY
    double getMapCell(const state& state);
    void setMapCell(const state& state, double value);
    bool close(double x, double y);
    double getG(const state& u);
    double getRHS(const state& u);
    void setG(const state& u, double g);
    void setRHS(const state& u, double rhs);
    double eightCondist(const state& a, const state& b);
    double radiusFromStart(const state& start, const state& b);
    int computeShortestPath();
    void updateVertex(const state& u);
    void insertOpen(const state& u);
    void removeOpen(const state& u);
    double trueDist(const state& a, const state& b);
    double heuristic(const state& a, const state& b);
    state calculateKey(const state& u);
    void getPred(const state& u, std::vector<state> &s);
    double cost(const state& a, const state& b);
    bool occupied(const state& u);
    bool isStateInOpenList(const state& u);
    bool isStateWithKeyInOpenList(const state& u);
    bool isStateInMap(const state& u);
};