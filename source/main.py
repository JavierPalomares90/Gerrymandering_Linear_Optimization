
# coding: utf-8

# In[1]:

from gurobipy import *
import utils
from math import radians, sin, cos, acos
import numpy as np



def readData():
    dataDir = "../census_data/"
    # file with population data
    populationFile = "/DEC_10_PL_P3_with_ann.csv"
    # file with geographical data
    geoFile = "/DEC_10_PL_G001_with_ann.csv"
    dirs = utils.getSubdirs(dataDir)
    blocks = []
    # get all of the blocks into one list
    count = 0
    for d in dirs:
        block,popData,geoData = utils.getBlocks(dataDir + d + populationFile,dataDir + d + geoFile)
        for b in block:
            blocks.append(b)
            count = count + 1
            if count > 6:
                break
    # read the political data
    fipsDataFile = "../census_data/st44_ri_cou.txt"
    poliDataFile = "../political_data/US_elect_county.csv"
    counties = utils.getPoliDataByCounty(poliDataFile,fipsDataFile)
    return (blocks, counties)


def getDistance(block, district):
    slat = radians(float(block["Latitude"]))
    slon = radians(float(block["Longitude"]))
    elat = radians(float(district["Latitude"]))
    elon = radians(float(district["Longitude"]))

    dist = 6371.01 * acos(sin(slat) * sin(elat) + cos(slat) * cos(elat) * cos(slon - elon))
    return dist


def getCostsArcsCapacity(blocks, districts):
    f = np.empty((len(blocks), len(districts)))
    arcs = tuplelist()
    numeric_Arcs = tuplelist()
    capacity = {}
    totalPop = 0
    for i,block in enumerate(blocks):
        totalPop += block["Population"]
        for j,district in enumerate(districts):
            f[i,j] = getDistance(block, district)
            arc = (block["Block"], j)
            arcs.append(arc)
            numeric_Arcs.append((i,j))
            capacity[arc] = block["Population"]
    return (f, arcs, capacity, numeric_Arcs, totalPop)



def assign(blocks, districts, counties):
    print(blocks)
    # gurobi network flow example: http://www.gurobi.com/documentation/7.5/examples/netflow_py.html
    #
    # cost function f(i,j) is the euclidian distance from block i to district j times population of block i
    # we can create a cost matrix f with rows being blocks and columns being districts
    # we can write the edges u as constrained to 0 or 1 indicating whether block i is assigned to district j
    #
    (f, arcs, capacity, numericArcs, totalPop) = getCostsArcsCapacity(blocks, districts)
    print(f)
    print(capacity)

    n = len(blocks)
    m = len(districts)
    pop_lower = 0.3 * totalPop
    pop_upper = 0.7 * totalPop
    M = 1000


    # Create optimization model
    model = Model('netflow')

    # Create variables
    flow = model.addVars(n, m, name="flow")
    indic = model.addVars(n, m, vtype=GRB.INTEGER, lb=0, ub=1, name="indic")


    # Arc capacity constraints
    model.addConstrs((flow.sum('*', i, j) <= capacity[i, j] for i, j in arcs), "cap")

    # to ensure all blocks are allocated and no splitting allowed
    model.addConstrs( (flow.sum(i,'*') == blocks[i]["Population"] for i in range(n)), "node")
    model.addConstrs((flow[i, j] <= M * indic[i, j] for i in range(n) for j in range(m)))
    model.addConstrs((indic.sum(i, '*') == 1 for i in range(n)))

    # to keep population "equal"
    model.addConstrs( (flow.sum('*', j) >= pop_lower for j in range(m)))
    model.addConstrs( (flow.sum('*', j) <= pop_upper for j in range(m)))


    model.optimize()

    # Print solution
    if model.status == GRB.Status.OPTIMAL:
        solution = model.getAttr('x', flow)
        indic_sol = model.getAttr('x', indic)
        print(solution)
        print(indic_sol)
        for j in range(m):
            totalPop = 0
            blocks_assigned = []
            for i in range(n):
                if indic_sol[i, j] > 0:
                    blocks_assigned.append(i)
                    totalPop += solution[i,j]
            print("For district " + str(j) + ": the population is " + str(totalPop)
                 + " and the assigned blocks are " + str(blocks_assigned) + "\n")


# In[5]:
(blocks, counties) = readData()
districts = [{'Latitude': '+41.1879323', 'Longitude': '-071.5828012'},
             {'Latitude': '+41.1686180', 'Longitude': '-071.5928347'}]
#This is a population block
#print(blocks[3])
#print(blocks[4])

# In[2]:

#who
# This is a vote by county
#print(counties[0])

assign(blocks[:6], districts, counties)

