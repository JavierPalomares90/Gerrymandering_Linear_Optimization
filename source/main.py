
# coding: utf-8

# In[1]:



# coding: utf-8

# In[1]:


# coding: utf-8

# In[1]:

from gurobipy import *
import utils
from math import radians, sin, cos, acos
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
# install basemap with: sudo -H pip3 install https://github.com/matplotlib/basemap/archive/v1.1.0.tar.gz




def readData():
    dataDir = "../census_data/"
    # file with population data
    populationFile = "/DEC_10_PL_P3_with_ann.csv"
    # file with geographical data
    geoFile = "/DEC_10_PL_G001_with_ann.csv"
    # file with the congressional districts per block
    cdFile = "../cd115/National_CD115.txt"
    dirs = utils.getSubdirs(dataDir)
    blocks = []
    counties = []
    # get all of the blocks into one list
    for d in dirs:
        block,popData,geoData = utils.getBlocks(dataDir + d + populationFile,dataDir + d + geoFile,cdFile)
        for b in block:
            blocks.append(b)
    # read the political data
    fipsDataFile = "../census_data/st44_ri_cou.txt"
    poliDataFile = "../political_data/US_elect_county.csv"
    #counties = utils.getPoliDataByCounty(poliDataFile,fipsDataFile)
    return (blocks, counties)


def getDistance(block, district):
    slat = radians(float(block["Latitude"]))
    slon = radians(float(block["Longitude"]))
    elat = radians(float(district["Latitude"]))
    elon = radians(float(district["Longitude"]))

    dist = 6371.01 * acos(sin(slat) * sin(elat) + cos(slat) * cos(elat) * cos(slon - elon))
    return dist


def connected(start, end):
    return True

def getCostsArcsCapacity(blocks, districts):
    f = np.empty((len(blocks), len(districts)))
    arcs = tuplelist()
    numeric_Arcs = tuplelist()
    capacity = {}
    totalPop = 0
    for i,block in enumerate(blocks):
    #     totalPop += block_start["Population"]
    #     for j,block_end in enumerate(blocks):
    #         if block_start is not block_end:
    #             if connected(block_start, block_end):
    #                 arc = (i, j)
    #                 arcs.append(arc)
    #                 #capacity[arc] = block["Population"]

        for j, district in enumerate(districts):
            f[i, j] = getDistance(block, district)
            arc = (i, j)
            #arc = (block["Block"], j)
            arcs.append(arc)
            #numeric_Arcs.append((i, j))
            capacity[arc] = block["Population"]
    return (f, arcs, capacity, totalPop)


def drawMap(solution, blocks, n, m):
    fig = plt.figure(figsize=(8, 8))
    map = Basemap(projection='merc', resolution='h',
                  width=5E1, height=5E1,
                  lat_0=40, lon_0=-70, )
    #map.etopo(scale=0.5, alpha=0.5)
    for j in range(m):
        for i in range(n):
            if solution[i, j] > 0:
                block = blocks[i]
                x, y = map(float(block["Longitude"]), float(block["Latitude"]))
                plt.plot(x, y, 'ok', markersize=5)
                plt.text(x, y, str(j), fontsize=12)
    plt.show()


def assign(blocks, districts, counties):
    #print(blocks)
    # gurobi network flow example: http://www.gurobi.com/documentation/7.5/examples/netflow_py.html
    #
    # cost function f(i,j) is the euclidian distance from block i to district j times population of block i
    # we can create a cost matrix f with rows being blocks and columns being districts
    # we can write the edges u as constrained to 0 or 1 indicating whether block i is assigned to district j
    #
    (f, arcs, capacity, totalPop) = getCostsArcsCapacity(blocks, districts)
    #print(f)
    #print(capacity)

    n = len(blocks)
    m = len(districts)
    print(totalPop)
    pop_lower = 0.1 * totalPop
    pop_upper = 0.9 * totalPop
    M = 1000 # should be as large as the largest block population. might calculate dynamically


    # Create optimization model
    model = Model('netflow')

    # Create variables
    flow = model.addVars(n, m, name="flow")
    indic = model.addVars(n, m, vtype=GRB.INTEGER, lb=0, ub=1, name="indic")


    # d[i,j] * population of block i * indic[i,j]
    model.setObjective((quicksum(f[i,j] * blocks[i]["Population"] * indic[i,j] for i in range(n) for j in range(m))), GRB.MINIMIZE)
    #model.setObjective((f[i,j] * blocks[i]["Population"] for i in range(n) for j in range(m)), GRB.MINIMIZE)

    # Arc capacity constraints
    #model.addConstrs((flow.sum('*', i, j) <= capacity[i, j] for i, j in arcs), "cap")

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
        drawMap(indic_sol, blocks, n, m)

# get the districtCenter by minimizing the population weighted squared
# distances between electoralDistricts and blocks
def getDistrictCenters(blocks,u,districtList):
    I = len(blocks);
    J = len(districtList);
    
    # collection of the x and y coordinates of the district centers
    districtCenterCoords={};
    for j in range(J):
        districtId = districtList[j];
        # finding the center for each district is an optimization problem
        model = Model('District_%d_Centers' % j)
        # by default gurobi wants optimal solutions to be non negative
        x = model.addVar(lb = -360,name="x_%d" % j);
        y = model.addVar(name="y_%d" % j);
        objective = QuadExpr();
        for i in range(I):
            block = blocks[i];
            blockId = block['Id2'];
            x_i = float(block['Longitude']);
            y_i = float(block['Latitude']);
            u_i = u[i];
            if(u_i['Id2'] != blockId):
                raise ValueError('There is a mismatch in the blockIds at i=%d' % i)
            u_i_j = u_i[districtId];
            objective.add(u_i_j * ( (x-x_i)*(x-x_i) + (y-y_i)*(y-y_i)));
        model.setObjective(objective)
        model.setParam( 'OutputFlag', False );
        model.optimize();
        districtCenterCoords[districtId] = (x.X,y.X);
    return districtCenterCoords
        

# In[5]:
(blocks, counties) = readData()
#This is a population block
print(blocks[3])
# This is a vote by county
#print(counties[0])



# In[2]:
u = utils.getNumBlockInDistrict(blocks)
#TODO: write method to figure out all districts
districtList= [1,2]
districtCenters = getDistrictCenters(blocks,u,districtList);
districts = [{'Latitude': districtCenters[1][1], 'Longitude': districtCenters[1][0]},
             {'Latitude': districtCenters[2][1], 'Longitude': districtCenters[2][0]}];
#districts = [{'Latitude': '+41.1879323', 'Longitude': '-071.5828012'},
#             {'Latitude': '+41.1686180', 'Longitude': '-071.5928347'}]
assign(blocks[:6], districts, counties)

