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


def readDataBlock():
    dataDir = "../census_data/"
    # file with population data
    populationFile = "/DEC_10_PL_P3_with_ann.csv"
    # file with geographical data
    geoFile = "/DEC_10_PL_G001_with_ann.csv"
    # file with the congressional districts per block
    cdFile = "../cd115/44_RI_CD115.txt"

    dirs = utils.getSubdirs(dataDir)
    blocks = []
    # get all of the blocks into one list
    for d in dirs:
        block,popData,geoData = utils.getBlocks(dataDir + d + populationFile,dataDir + d + geoFile,cdFile)
        blocks = blocks + block

    # read the political data
    fipsDataFile = "../census_data/st44_ri_cou.txt"
    poliDataFile = "../political_data/US_elect_county.csv"
    counties = utils.getPoliDataByCounty(poliDataFile, fipsDataFile)
    return (blocks, counties)


def readData():
    dataDir = "../census_data_tract"
    # file with population data
    populationFile = "/DEC_10_PL_P3_with_ann.csv"
    # file with geographical data
    geoFile = "/DEC_10_PL_G001_with_ann.csv"
    # file with the congressional districts per block
    cdFile = "../cd115/44_RI_CD115.txt"

    blocks, popData, geoData = utils.getBlocks(dataDir + populationFile, dataDir + geoFile, cdFile)

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
    capacity = {}
    totalPop = 0
    maxPop = 0
    for i,block in enumerate(blocks):
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

#    map.readshapefile('../census_block_shape_files/tl_2010_44_tabblock10', 'tl_2010', drawbounds=False)
    map.readshapefile('../census_tract_shape_files/tl_2010_44_tract10', 'tl_2010', drawbounds=False)

    patches = []
    for i in range(m):
        patches.append([])

    for info, shape in zip(map.tl_2010_info, map.tl_2010):
        id = info['GEOID10']
        district = solution[id]
        patches[district].append(Polygon(np.array(shape), True))

    for i in range(m):
        ax.add_collection(PatchCollection(patches[i], facecolor=colors[i], edgecolor='k', linewidths=0., zorder=2))
    map.readshapefile('../Congressional_Districts/Congressional_Districts', 'Congressional_Districts', drawbounds=True,color='g')

    plt.show()


def analyzeSolution(counties, blocks, solution, n, m):
    sol_map = {}
    tractResult = []
    popResult = []
    raceResult = []
    polResult = []
    county_result = []
    county_total = {}
    for county in counties:
        county_total[county["County"]] = 0
    for j in range(m):
        totalPop = 0
        numAssigned = 0
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
            if solution[i, j] > 0:
                block = blocks[i]
                pop = block["Population"]
                #blocks_assigned.append(i)
                totalPop += pop
                numAssigned += 1
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
        tractResult.append(numAssigned)
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
    return (sol_map, popResult, tractResult, raceResult, polResult)


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


def assign(blocks, districts, neighbors, n, m):
    # cost function f(i,j) is the euclidian distance from block i to district j 

    (f, arcs, capacity, totalPop, maxPop) = getCostsArcsCapacity(blocks, districts)
    #print(f)
    #print(capacity)

    print(totalPop)
    M = maxPop * 10
    alpha = .05
    p = totalPop / m
    pop_cost = 5
    distance_cost = 100

    # Create optimization model
    model = Model('netflow')

    # Create variables
    flow = model.addVars(n, m, name="flow") # flow(i,j) = number of people from block i assigned to district j
    indic = model.addVars(n, m, vtype=GRB.BINARY, name="indic") # indic(i,j) is an indicator of whether block i is assigned to district j
    smallest = model.addVar(name="min")     # smallest district
    largest = model.addVar(name="max")      # largest district
    gap = model.addVar(name="gap")          # difference between largest and smallest

    # Objective function
    model.setObjective((distance_cost * quicksum(f[i,j] * blocks[i]["Population"] * indic[i,j] for i in range(n) for j in range(m)) +
                       pop_cost * gap), GRB.MINIMIZE)

    # Constraints
    # to ensure all blocks are allocated and no splitting allowed
    model.addConstrs( (flow.sum(i,'*') == blocks[i]["Population"] for i in range(n)), "node")
    model.addConstrs((flow[i, j] <= M * indic[i, j] for i in range(n) for j in range(m)))
    model.addConstrs((indic.sum(i, '*') == 1 for i in range(n)))

    # Population related constraints to help achieve equal sizes
    model.addConstrs( smallest <= flow.sum('*', j) for j in range(m))
    model.addConstrs( largest >= flow.sum('*', j) for j in range(m))
    model.addConstr(gap == largest - smallest)
    for k in range(m):
        # constraint (10)
        model.addConstr( quicksum(blocks[i]['Population'] * indic[i,k] for i in range(n)) >= (1-alpha)*p )
        # constraint (11)
        model.addConstr( quicksum(blocks[i]['Population'] * indic[i,k] for i in range(n)) <= (1+alpha)*p )

    # add continuity constraints
    w = model.addVars(n,m, name="hub_indicators",vtype = GRB.BINARY)
    # only one hub per district. This is constraint (2)
    for k in range(m):
        model.addConstr(quicksum(w[i,k] for i in range(n)) == 1)
        
    # y_i_j is a decision variable that indicates the amount of flow from block i to block j
    # y must be nonnegative
    # will add y variables dynamically only when we find a pair
    y = {}

    # constraint (3) (specifically  (21) from the examples)
    for k in range(m):
        for i in range(n):
            # neighbors is the list of blocks adjacent to i.
            flowInto = LinExpr()
            neighborsOfI = neighbors[i]
            for j in neighborsOfI:
                # add the flow variable dynamically for the found pairs
                # variable must be non negative
                y[(j,i,k)] = model.addVar(name="flow_%d_%d_%d" % (j,i,k),lb=0)
                flowInto.add(y[(j,i,k)])
            model.addConstr(flowInto <= (n - 1) * indic[i,k])

    # constraint (1) ((20) from the examples)
    for k in range(m):
        for i in range(n):
            neighborsOfI = neighbors[i]
            netFlow = LinExpr()
            for j in neighborsOfI:
                netFlow.add(y[(i,j,k)] - y[(j,i,k)])
                #model.addConstr(netFlow >= (indic[i,k] - n * w[i,k]))
                # IS THIS SUPPOSED TO BE COMMENTED? if so, we can remove this entire chunk

    model.optimize()

    # Print solution
    if model.status == GRB.Status.OPTIMAL:
        solution = model.getAttr('x', indic)
        #print(solution)
    else:
        solution = []
    return(totalPop, solution)


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

        


# In[5]:
(blocks, counties) = readData()

# In[2]:
u = utils.getNumBlockInDistrict(blocks)

# convert districts into correct format
districtList= [1,2]
districtCenters = getDistrictCenters(blocks,u,districtList)
print(districtCenters)
districts = [{'Latitude': districtCenters[1][1], 'Longitude': districtCenters[1][0]},
             {'Latitude': districtCenters[2][1], 'Longitude': districtCenters[2][0]}]

n = len(blocks)
m = len(districts)

#shapesDir = "../census_block_shape_files/tl_2010_44_tabblock10"
shapesDir = "../census_tract_shape_files/tl_2010_44_tract10"
indexMapping = utils.getIndexMapping(blocks)
neighbors,neighborsByIndex = utils.getNeighbors(shapesDir,indexMapping)

(totalPop, solution) = assign(blocks, districts, neighborsByIndex, n, m)
(sol_map, popResult, tractResult, raceResult, polResult) = analyzeSolution(counties, blocks, solution, n, m)
print(popResult)
print(tractResult)
print(raceResult)
print(polResult)
(metrics, totalMetrics) = calcMetrics(popResult, raceResult, polResult, m)
print(metrics)
print(totalMetrics)
drawMap(sol_map, m)


