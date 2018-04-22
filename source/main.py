
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
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


totalVoters = 429697

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
    # get all of the blocks into one list
    for d in dirs:
        block,popData,geoData = utils.getBlocks(dataDir + d + populationFile,dataDir + d + geoFile,cdFile)
        blocks = blocks + block
        #for b in block:
        #    blocks.append(b)
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


def connected(start, end):
    return True


def getCostsArcsCapacity(blocks, districts):
    f = np.empty((len(blocks), len(districts)))
    arcs = tuplelist()
    capacity = {}
    totalPop = 0
    maxPop = 0
    for i,block in enumerate(blocks):
    #     totalPop += block_start["Population"]
    #     for j,block_end in enumerate(blocks):
    #         if block_start is not block_end:
    #             if connected(block_start, block_end):
    #                 arc = (i, j)
    #                 arcs.append(arc)
    #                 #capacity[arc] = block["Population"]
        pop = block["Population"]
        totalPop += pop
        if pop > maxPop:
            maxPop = pop
        for j, district in enumerate(districts):
            f[i, j] = getDistance(block, district)
            arc = (i, j)
            arcs.append(arc)
            capacity[arc] = pop

    return (f, arcs, capacity, totalPop, maxPop)


def drawMap(solution, m):
    colors = ['m', 'b']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    map = Basemap(llcrnrlon=-72, llcrnrlat=40.9, urcrnrlon=-70.9, urcrnrlat=42.2,
                  resolution='i', area_thresh=5000., projection='lcc',
                  lat_1=40., lon_0=-071., ax=ax)
    map.drawcoastlines()
    map.drawcountries()
    map.drawstates()

    map.readshapefile('../census_block_shape_files/tl_2010_44_tabblock10', 'tl_2010_44_tabblock10', drawbounds=False)

    patches = []
    for i in range(m):
        patches.append([])

    for info, shape in zip(map.tl_2010_44_tabblock10_info, map.tl_2010_44_tabblock10):
        id = info['GEOID10']
        district = solution[id]
        patches[district].append(Polygon(np.array(shape), True))

    for i in range(m):
        ax.add_collection(PatchCollection(patches[i], facecolor=colors[i], edgecolor='k', linewidths=0., zorder=2))

    plt.show()


def analyzeSolution(counties, blocks, solution, indic_sol, n, m):
    sol_map = {}
    popResult = []
    raceResult = []
    polResult = []
    county_result = []
    county_total = {}
    for county in counties:
        county_total[county["County"]] = 0
    for j in range(m):
        totalPop = 0
        #blocks_assigned = []
        white = 0
        black = 0
        asian = 0
        native = 0
        island = 0
        other = 0
        county_distr = {}
        for county in counties:
            county_distr[county["County"]] = 0
        for i in range(n):
            if indic_sol[i, j] > 0:
                pop = solution[i, j]
                block = blocks[i]
                #blocks_assigned.append(i)
                totalPop += pop
                sol_map[str(block['Id2'])] = j
                county_distr[block["County"]] += pop
                county_total[block["County"]] += pop
                white += block["Race"]["white"]
                black += block["Race"]["black"]
                asian += block["Race"]["asian"]
                native += block["Race"]["native"]
                island += block["Race"]["island"]
                other += block["Race"]["other"]
        popResult.append(totalPop)
        raceResult.append({"white": white, "black": black, "asian": asian, "native": native, "island": island, "other": other})
        county_result.append(county_distr)
        print("For district " + str(j) + ": the population is " + str(totalPop))
        # + " and the assigned blocks are " + str(blocks_assigned) + "\n")
    for i in range(m):
        county_distr = county_result[i]
        obama_total = 0
        romney_total = 0
        for county in counties:
            name = county["County"]
            percent = float(county_distr[name]) / county_total[name]
            obama_total += county["Obama vote"] * percent
            romney_total += county["Romney vote"] * percent
        polResult.append({"Obama": obama_total, "Romney": romney_total, "total": obama_total+romney_total})
    return (sol_map, popResult, raceResult, polResult)


def calcMetrics(popResult, raceResult, polResult, m):
    metrics = []
    for i in range(m):
        pop = popResult[i]
        perWhite = raceResult[i]["white"] / pop * 100.0
        perBlack = raceResult[i]["black"] / pop * 100.0
        perAsian = raceResult[i]["asian"] / pop * 100.0
        perNative = raceResult[i]["native"] / pop * 100.0
        perIsland = raceResult[i]["island"] / pop * 100.0
        perOther = raceResult[i]["other"] / pop * 100.0
        perObama = polResult[i]["Obama"] / polResult[i]["total"] * 100.0
        perRomney = polResult[i]["Romney"] / polResult[i]["total"] * 100.0
        metrics.append({"perWhite": perWhite, "perBlack": perBlack, "perAsian": perAsian,
                        "perNative": perNative, "perIsland": perIsland, "perOther": perOther,
                        "perObama": perObama, "perRomney": perRomney})
    totalMetrics = {"white": quicksum(raceResult[i]["white"] for i in range(m)) / totalPop * 100.0,
                    "black": quicksum(raceResult[i]["black"] for i in range(m)) / totalPop * 100.0,
                    "asian": quicksum(raceResult[i]["asian"] for i in range(m)) / totalPop * 100.0,
                    "native": quicksum(raceResult[i]["native"] for i in range(m)) / totalPop * 100.0,
                    "island": quicksum(raceResult[i]["island"] for i in range(m)) / totalPop * 100.0,
                    "other": quicksum(raceResult[i]["other"] for i in range(m)) / totalPop * 100.0,
                    "Obama": quicksum(polResult[i]["Obama"] for i in range(m)) / totalVoters * 100.0,
                    "Romney": quicksum(polResult[i]["Romney"] for i in range(m)) / totalVoters * 100.0,
                    }
    return (metrics, totalMetrics)


def assign(blocks, districts, counties, n, m):
    #print(blocks)
    # gurobi network flow example: http://www.gurobi.com/documentation/7.5/examples/netflow_py.html
    #
    # cost function f(i,j) is the euclidian distance from block i to district j 
    # we can create a cost matrix f with rows being blocks and columns being districts
    # we can write the edges u as constrained to 0 or 1 indicating whether block i is assigned to district j
    #
    (f, arcs, capacity, totalPop, maxPop) = getCostsArcsCapacity(blocks, districts)
    #print(f)
    #print(capacity)

    print(totalPop)
    pop_lower = 0.1 * totalPop
    pop_upper = 0.9 * totalPop
    M = maxPop * 10 # should be as large as the largest block population. might calculate dynamically

    pop_cost = 3
    distance_cost = 1

    # Create optimization model
    model = Model('netflow')

    # Create variables
    flow = model.addVars(n, m, name="flow")
    indic = model.addVars(n, m, vtype=GRB.INTEGER, lb=0, ub=1, name="indic")
    smallest = model.addVar(name="min")
    largest = model.addVar(name="max")
    gap = model.addVar(name="gap")


    # f[i,j] * population of block i * indic[i,j]
    model.setObjective((distance_cost * quicksum(f[i,j] * blocks[i]["Population"] * indic[i,j] for i in range(n) for j in range(m)) +
                       pop_cost * gap), GRB.MINIMIZE)
    #model.setObjective((f[i,j] * blocks[i]["Population"] for i in range(n) for j in range(m)), GRB.MINIMIZE)

    # Arc capacity constraints
    #model.addConstrs((flow.sum('*', i, j) <= capacity[i, j] for i, j in arcs), "cap")

    # to ensure all blocks are allocated and no splitting allowed
    model.addConstrs( (flow.sum(i,'*') == blocks[i]["Population"] for i in range(n)), "node")
    model.addConstrs((flow[i, j] <= M * indic[i, j] for i in range(n) for j in range(m)))
    model.addConstrs((indic.sum(i, '*') == 1 for i in range(n)))

    model.addConstrs( smallest <= flow.sum('*', j) for j in range(m))
    model.addConstrs( largest >= flow.sum('*', j) for j in range(m))
    model.addConstr(gap == largest - smallest)

    # to keep population "equal"
    #model.addConstrs( (flow.sum('*', j) >= pop_lower for j in range(m)))
    #model.addConstrs( (flow.sum('*', j) <= pop_upper for j in range(m)))


    model.optimize()

    # Print solution
    if model.status == GRB.Status.OPTIMAL:
        solution = model.getAttr('x', flow)
        indicSol = model.getAttr('x', indic)
        #print(solution)
        #print(indicSol)
    else:
        solution = []
        indicSol = []
    return(totalPop, solution, indicSol)


# get the districtCenter by minimizing the population weighted squared
# distances between electoralDistricts and blocks
def getDistrictCenters(blocks,u,districtList):
    I = len(blocks)
    J = len(districtList)
    
    # collection of the x and y coordinates of the district centers
    districtCenterCoords={}
    for j in range(J):
        districtId = districtList[j]
        # finding the center for each district is an optimization problem
        model = Model('District_%d_Centers' % j)
        # by default gurobi wants optimal solutions to be non negative
        x = model.addVar(lb = -360,name="x_%d" % j)
        y = model.addVar(name="y_%d" % j)
        objective = QuadExpr()
        for i in range(I):
            block = blocks[i]
            blockId = block['Id2']
            x_i = float(block['Longitude'])
            y_i = float(block['Latitude'])
            u_i = u[i]
            if(u_i['Id2'] != blockId):
                raise ValueError('There is a mismatch in the blockIds at i=%d' % i)
            u_i_j = u_i[districtId]
            objective.add(u_i_j * ( (x-x_i)*(x-x_i) + (y-y_i)*(y-y_i)))
        model.setObjective(objective)
        model.setParam( 'OutputFlag', False )
        model.optimize()
        districtCenterCoords[districtId] = (x.X,y.X)
    return districtCenterCoords

# dummy method to formulate constraints for continuous districts.
# need to add these constrainst to assign() method
# dummy method to formulate constraints for continuous districts.
# need to add these constrainst to assign() method
def getContinuousDistricts(blocks,districts,neighbors):
    numBlocks = len(blocks);
    numDistricts = len(districts);
    m = numBlocks;

    model = Model('Continuous_Districts');
    #w_i_k is a decision variable that equals 1 if the block i is the hub of district k
    w = model.addVars(numBlocks, numDistricts, name="hub_indicators",vtype = GRB.BINARY);
    # only one hub per district. This is constraint (2)
    for k in range(numDistricts):
        model.addConstr(quicksum(w[i,k] for i in range(numBlocks)) == 1);
    # x_i_k is a decision variable that equals 1 if the block i is assigned to district k
    # This should be the same var as "indic"
    x = model.addVars(numBlocks,numDistricts, name="assignment_indicator", vtype=GRB.BINARY);
    # y_i+j is a decision variable that indicates the amount of flow from block i to block j
    # y must be nonnegative
    # will add y variables dynamically only when we find a pair
    y = {};

    # constraint (3) (specifically  (21) from the examples)
    for k in range(numDistricts):
        for i in range(numBlocks):
            # neighbors is the list of blocks adjacent to i.
            flowInto = LinExpr();
            neighborsOfI = neighbors[i];
            for j in neighborsOfI:
                # add the flow variable dynamically for the found pairs
                # variable must be non negative
                y[(j,i,k)] = model.addVar(name="flow_%d_%d_%d" % (j,i,k),lb=0);
                flowInto.add(y[(j,i,k)])
            model.addConstr(flowInto <= (m - 1) * x[i,k]);
    # constraint (1) ((20) from the examples)
    for k in range(numDistricts):
        for i in range(numBlocks):
            neighborsOfI = neighbors[i];
            netFlow = LinExpr();
            for j in neighborsOfI:
                netFlow.add(y[(i,j,k)] - y[(j,i,k)])
            model.addConstr(netFlow >= (x[i,k] - m * w[i,k]));
    model.update();
    model.setObjective(0);
    model.optimize(); 
        


# In[5]:
(blocks, counties) = readData()
#This is a population block
print(blocks[3])
# This is a vote by county
print(counties[0])



# In[2]:
u = utils.getNumBlockInDistrict(blocks)
#TODO: write method to figure out all districts
districtList= [1,2]
districtCenters = getDistrictCenters(blocks,u,districtList)
print(districtCenters)
districts = [{'Latitude': districtCenters[1][1], 'Longitude': districtCenters[1][0]},
             {'Latitude': districtCenters[2][1], 'Longitude': districtCenters[2][0]}];

n = len(blocks)
m = len(districts)

(totalPop, solution, indicSol) = assign(blocks, districts, counties, n, m)
(sol_map, popResult, raceResult, polResult) = analyzeSolution(counties, blocks, solution, indicSol, n, m)
print(popResult)
print(raceResult)
print(polResult)
(metrics, totalMetrics) = calcMetrics(popResult, raceResult, polResult, m)
print(metrics)
print(totalMetrics)
drawMap(sol_map, m)


# In[3]:

shapesDir = "../census_block_shape_files/tl_2010_44_tabblock10";
indexMapping = utils.getIndexMapping(blocks);
neighbors,neighborsByIndex = utils.getNeighbors(shapesDir,indexMapping);
getContinuousDistricts(blocks,districts,neighborsByIndex)
        

