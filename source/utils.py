
# coding: utf-8

# In[2]:



# coding: utf-8

# In[1]:


import re
import csv
import os
import shapefile
import pysal as ps
from math import asin, sqrt, cos, pi


race_keys = {
    "Population of one race: - White alone": {"white": 100, "black": 0, "native": 0, "asian": 0, "island": 0, "other": 0},
    "Population of one race: - Black or African American alone": {"white": 0, "black": 100, "native": 0, "asian": 0, "island": 0, "other": 0},
    "Population of one race: - American Indian and Alaska Native alone": {"white": 0, "black": 0, "native": 100, "asian": 0, "island": 0, "other": 0},
    "Population of one race: - Asian alone": {"white": 0, "black": 0, "native": 0, "asian": 100, "island": 0, "other": 0},
    "Population of one race: - Native Hawaiian and Other Pacific Islander alone": {"white": 0, "black": 0, "native": 0, "asian": 0, "island": 100, "other": 0},
    "Population of one race: - Some Other Race alone": {"white": 0, "black": 0, "native": 0, "asian": 0, "island": 0, "other": 100},
    "Two or More Races: - Population of two races: - White; Black or African American" : {"white": 50, "black": 50, "native": 0, "asian": 0, "island": 0, "other": 0},
    "Two or More Races: - Population of two races: - White; American Indian and Alaska Native": {"white": 50, "black": 0, "native": 50, "asian": 0, "island": 0, "other": 0},
    "Two or More Races: - Population of two races: - White; Asian": {"white": 50, "black": 0, "native": 0, "asian": 50, "island": 0, "other": 0},
    "Two or More Races: - Population of two races: - White; Native Hawaiian and Other Pacific Islander": {"white": 50, "black": 0, "native": 0, "asian": 0, "island": 50, "other": 0},
    "Two or More Races: - Population of two races: - White; Some Other Race": {"white": 50, "black": 0, "native": 0, "asian": 0, "island": 0, "other": 50},
    "Two or More Races: - Population of two races: - Black or African American; American Indian and Alaska Native": {"white": 0, "black": 50, "native": 50, "asian": 0, "island": 0, "other": 0},
    "Two or More Races: - Population of two races: - Black or African American; Asian": {"white": 0, "black": 50, "native": 0, "asian": 50, "island": 0, "other": 0},
    "Two or More Races: - Population of two races: - Black or African American; Native Hawaiian and Other Pacific Islander": {"white": 0, "black": 50, "native": 0, "asian": 0, "island": 50, "other": 0},
    "Two or More Races: - Population of two races: - Black or African American; Some Other Race": {"white": 0, "black": 50, "native": 0, "asian": 0, "island": 0, "other": 50},
    "Two or More Races: - Population of two races: - American Indian and Alaska Native; Asian": {"white": 0, "black": 0, "native": 50, "asian": 50, "island": 0, "other": 0},
    "Two or More Races: - Population of two races: - American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander": {"white": 0, "black": 0, "native": 50, "asian": 0, "island": 50, "other": 0},
    "Two or More Races: - Population of two races: - American Indian and Alaska Native; Some Other Race": {"white": 0, "black": 0, "native": 50, "asian": 0, "island": 0, "other": 50},
    "Two or More Races: - Population of two races: - Asian; Native Hawaiian and Other Pacific Islander": {"white": 0, "black": 0, "native": 0, "asian": 50, "island": 50, "other": 0},
    "Two or More Races: - Population of two races: - Asian; Some Other Race": {"white": 0, "black": 0, "native": 0, "asian": 50, "island": 0, "other": 50},
    "Two or More Races: - Population of two races: - Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 0, "black": 0, "native": 0, "asian": 0, "island": 50, "other": 50},
    "Two or More Races: - Population of three races: - White; Black or African American; American Indian and Alaska Native": {"white": 34, "black": 33, "native": 33, "asian": 0, "island": 0, "other": 0},
    "Two or More Races: - Population of three races: - White; Black or African American; Asian": {"white": 34, "black": 33, "native": 0, "asian": 33, "island": 0, "other": 0},
    "Two or More Races: - Population of three races: - White; Black or African American; Native Hawaiian and Other Pacific Islander": {"white": 34, "black": 33, "native": 0, "asian": 0, "island": 33, "other": 0},
    "Two or More Races: - Population of three races: - White; Black or African American; Some Other Race": {"white": 34, "black": 33, "native": 0, "asian": 0, "island": 0, "other": 33},
    "Two or More Races: - Population of three races: - White; American Indian and Alaska Native; Asian": {"white": 34, "black": 0, "native": 33, "asian": 33, "island": 0, "other": 0},
    "Two or More Races: - Population of three races: - White; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander": {"white": 34, "black": 0, "native": 33, "asian": 0, "island": 33, "other": 0},
    "Two or More Races: - Population of three races: - White; American Indian and Alaska Native; Some Other Race": {"white": 34, "black": 0, "native": 33, "asian": 0, "island": 0, "other": 33},
    "Two or More Races: - Population of three races: - White; Asian; Native Hawaiian and Other Pacific Islander": {"white": 34, "black": 0, "native": 0, "asian": 33, "island": 33, "other": 0},
    "Two or More Races: - Population of three races: - White; Asian; Some Other Race": {"white": 34, "black": 0, "native": 0, "asian": 33, "island": 0, "other": 33},
    "Two or More Races: - Population of three races: - White; Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 34, "black": 0, "native": 0, "asian": 0, "island": 33, "other": 33},
    "Two or More Races: - Population of three races: - Black or African American; American Indian and Alaska Native; Asian": {"white": 0, "black": 34, "native": 33, "asian": 33, "island": 0, "other": 0},
    "Two or More Races: - Population of three races: - Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander": {"white": 0, "black": 34, "native": 33, "asian": 0, "island": 33, "other": 0},
    "Two or More Races: - Population of three races: - Black or African American; American Indian and Alaska Native; Some Other Race": {"white": 0, "black": 34, "native": 33, "asian": 0, "island": 0, "other": 33},
    "Two or More Races: - Population of three races: - Black or African American; Asian; Native Hawaiian and Other Pacific Islander": {"white": 0, "black": 34, "native": 0, "asian": 33, "island": 33, "other": 0},
    "Two or More Races: - Population of three races: - Black or African American; Asian; Some Other Race": {"white": 0, "black": 34, "native": 0, "asian": 33, "island": 0, "other": 33},
    "Two or More Races: - Population of three races: - Black or African American; Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 0, "black": 34, "native": 0, "asian": 0, "island": 33, "other": 33},
    "Two or More Races: - Population of three races: - American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander": {"white": 0, "black": 0, "native": 33, "asian": 33, "island": 33, "other": 0},
    "Two or More Races: - Population of three races: - American Indian and Alaska Native; Asian; Some Other Race": {"white": 0, "black": 0, "native": 33, "asian": 33, "island": 0, "other": 33},
    "Two or More Races: - Population of three races: - American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 0, "black": 0, "native": 33, "asian": 0, "island": 33, "other": 33},
    "Two or More Races: - Population of three races: - Asian; Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 0, "black": 0, "native": 0, "asian": 33, "island": 33, "other": 33},
    "Two or More Races: - Population of four races: - White; Black or African American; American Indian and Alaska Native; Asian": {"white": 25, "black": 25, "native": 25, "asian": 25, "island": 0, "other": 0},
    "Two or More Races: - Population of four races: - White; Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander": {"white": 25, "black": 25, "native": 25, "asian": 0, "island": 25, "other": 0},
    "Two or More Races: - Population of four races: - White; Black or African American; American Indian and Alaska Native; Some Other Race": {"white": 25, "black": 25, "native": 25, "asian": 0, "island": 0, "other": 25},
    "Two or More Races: - Population of four races: - White; Black or African American; Asian; Native Hawaiian and Other Pacific Islander": {"white": 25, "black": 25, "native": 0, "asian": 25, "island": 25, "other": 0},
    "Two or More Races: - Population of four races: - White; Black or African American; Asian; Some Other Race": {"white": 25, "black": 25, "native": 0, "asian": 25, "island": 0, "other": 25},
    "Two or More Races: - Population of four races: - White; Black or African American; Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 25, "black": 25, "native": 0, "asian": 0, "island": 25, "other": 25},
    "Two or More Races: - Population of four races: - White; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander": {"white": 25, "black": 0, "native": 25, "asian": 25, "island": 25, "other": 0},
    "Two or More Races: - Population of four races: - White; American Indian and Alaska Native; Asian; Some Other Race": {"white": 25, "black": 0, "native": 25, "asian": 25, "island": 0, "other": 25},
    "Two or More Races: - Population of four races: - White; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 25, "black": 0, "native": 25, "asian": 0, "island": 25, "other": 25},
    "Two or More Races: - Population of four races: - White; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 25, "black": 0, "native": 0, "asian": 25, "island": 25, "other": 25},
    "Two or More Races: - Population of four races: - Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander": {"white": 0, "black": 25, "native": 25, "asian": 25, "island": 25, "other": 0},
    "Two or More Races: - Population of four races: - Black or African American; American Indian and Alaska Native; Asian; Some Other Race": {"white": 0, "black": 25, "native": 25, "asian": 25, "island": 0, "other": 25},
    "Two or More Races: - Population of four races: - Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 0, "black": 25, "native": 25, "asian": 0, "island": 25, "other": 25},
    "Two or More Races: - Population of four races: - Black or African American; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 0, "black": 25, "native": 0, "asian": 25, "island": 25, "other": 25},
    "Two or More Races: - Population of four races: - American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 0, "black": 0, "native": 25, "asian": 25, "island": 25, "other": 25},
    "Two or More Races: - Population of five races: - White; Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander": {"white": 20, "black": 20, "native": 20, "asian": 20, "island": 20, "other": 0},
    "Two or More Races: - Population of five races: - White; Black or African American; American Indian and Alaska Native; Asian; Some Other Race": {"white": 20, "black": 20, "native": 20, "asian": 20, "island": 20, "other": 0},
    "Two or More Races: - Population of five races: - White; Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 20, "black": 20, "native": 20, "asian": 0, "island": 20, "other": 20},
    "Two or More Races: - Population of five races: - White; Black or African American; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 20, "black": 20, "native": 0, "asian": 20, "island": 20, "other": 20},
    "Two or More Races: - Population of five races: - White; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander;  Some Other Race": {"white": 20, "black": 0, "native": 20, "asian": 20, "island": 20, "other": 20},
    "Two or More Races: - Population of five races: - Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 0, "black": 20, "native": 20, "asian": 20, "island": 20, "other": 20},
    "Two or More Races: - Population of six races: - White; Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race": {"white": 16, "black": 16, "native": 17, "asian": 17, "island": 17, "other": 17},

}


def getGeography(s):
    strings = s.split(',')
    if(len(strings) < 5):
        return
    #get the block number and block group using regEx
    blockId = int(re.search(r'\d+', strings[0]).group())
    blockGroupId = int(re.search(r'\d+', strings[1]).group())
    censusTract = int(re.search(r'\d+', strings[2]).group())
    county = strings[3].lstrip()
    state = strings[4].lstrip()
    return blockId,blockGroupId,censusTract,county,state


def getTotalPopulation(data):
    pop = 0
    for key in data:
        # sum up all of the population for a block
        if (key.find("Population") != -1):
            pop += int(data[key]);
    return pop


def getFipsCountyCode(stateCode,countyCode):
    code = "%d%03d" %(stateCode,countyCode);
    return int(code)


def getRaceData(data):
    #geoId = int(data[i][geoIdKey]);
    white = 0
    black = 0
    native = 0
    asian = 0
    island = 0
    other = 0
    total = int(data["Total:"])
    running = 0
    if total > 0:
        for key in race_keys:
            value = int(data[key])
            running += value
            percentages = race_keys[key]
            white += percentages["white"] / 100.0 * value
            black += percentages["black"] / 100.0 * value
            native += percentages["native"] / 100.0 * value
            island += percentages["island"] / 100.0 * value
            other += percentages["other"] / 100.0 * value
            asian += percentages["asian"] / 100.0 * value
            if running == total:
                break
        sum = white + black + native + island + other + asian
        if sum < total - 1:
            print("something wrong!")
            print("sum is " + str(sum) + " and total is " + str(total))
            exit(0)
    return {"white": white, "black": black, "native": native, "island": island, "other": other, "asian": asian}


def getBlocks(popFilename,geoFilename,cdFilename):
    # read the poulation data
    with open(popFilename) as f:
        # skip the first "header" row
        next(f)
        # read the data as a dictionary
        reader = csv.DictReader(f)
        data = [r for r in reader]
    # read the geo data
    with open(geoFilename) as geoFile:
        # skip the first "header" row
        next(geoFile)
        # read the data as a dictionary
        geoReader = csv.DictReader(geoFile)
        geoData = [r for r in geoReader]
    # read the congressional district data
    cdMapping = {}
    with open(cdFilename) as cdFile:
        # skip the header row
        next(cdFile)
        for line in cdFile:
            blockId, districtId = line.partition(",")[::2]
            cdMapping[int(blockId)] = districtId.rstrip()
    numCensusBlocks = len(data)
    # populationBlocks is a list of maps, one map per census block
    populationBlocks = []
    # geography keys
    # use the second id as the identifier
    geoIdKey = 'Id2'
    geoKey = 'Geography'
    blockKey = 'Block'
    populationKey = 'Population'
    countyKey = 'County'
    censusTractKey = "Census_Tract"
    latKey = "Latitude"
    latLookup = "AREA CHARACTERISTICS - Internal Point (Latitude)"
    longLookup = "AREA CHARACTERISTICS - Internal Point (Longitude)"
    longKey = "Longitude"
    fipsCodeKey = "FIPS Code"
    stateLookup = "GEOGRAPHIC AREA CODES - State (FIPS)"
    countyLookup = "GEOGRAPHIC AREA CODES - County"
    districtKey = "Congressional District"
    raceKey = "Race"

    for i in range(numCensusBlocks):
        block = {}
        geoId = int(data[i][geoIdKey])
        # make sure the id's match
        if(geoId != int(geoData[i][geoIdKey])):
            raise("There is a mismatch in ids in files:" + popFilename + " " + geoFilename)
            return
        block[latKey] = geoData[i][latLookup]
        block[longKey] = geoData[i][longLookup]
        block[geoIdKey] = geoId
        blockId,blockGroupId,censusTract,county,state = getGeography(data[i][geoKey])
        block[blockKey] = blockId
        #totalPopulation = getTotalPopulation(data[i])
        #block[populationKey] = totalPopulation
        block[populationKey] = int(data[i]["Total:"])
        block[countyKey] = county
        block[censusTractKey] = censusTract
        # Add lat/long data
        block[latKey] = geoData[i][latLookup]
        block[longKey] = geoData[i][longLookup]
        # Add fips code
        state = int(geoData[i][stateLookup])
        county = int(geoData[i][countyLookup])
        fipsCode = getFipsCountyCode(state, county)
        block[fipsCodeKey] = fipsCode
        districtId = cdMapping.get(geoId)
        if districtId == None:
            print(geoId)
        # Add the district this block is currently part of
        block[districtKey] = districtId
        race_data = getRaceData(data[i])
        block[raceKey] = race_data
        populationBlocks.append(block)    
    return populationBlocks,data,geoData
def getSubdirs(dir):
    #"Get a list of immediate subdirectories"
    return next(os.walk(dir))[1]
# compute the distance between 2 lat/long points measured in degrees using the haversine formulae

def distanceInKm(lat1, lon1, lat2, lon2):
    earthDiameterKm = 12742
    p = pi/180
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return earthDiameterKm * asin(sqrt(a))

# return the block if a block is in a district. False otherwise
def blockInDistrict(block,district):
    idKey = 'Id2'
    blockId = block[idKey]
    return next((item for item in district if item[idKey] == blockId), False)
# measure of compactness of a district
# obtained by computing the percentange of sites in the circle centered at s 
# of radius r that are not in the district
def circularCompactness(blocks,district,radius,centerLong,centerLat):
    numBlocks = len(blocks)
    numBlocksInDistrict = len(district)
    # the number of blocks within the 
    numInRadius = 0
    numNotInD = 0
    latKey = "Latitude"
    longKey = "Longitude"
    for block in blocks:
        latitude = block[latKey]
        longitude = block[longKey]
        if( distanceInKm(latitude,longitude,centerLong,centerLat) <= radius):
            numInRadius += 1
            if(blockInDistrict(block,district) == False):
                numNotInD += 1
    return numNotInD / numInRadius

#returns the latitude and longitude of the geographical center of a district
def geographicalCenter(district):
    numBlocks = len(district)
    latitude = 0
    longitude = 0
    latKey = "Latitude"
    longKey = "Longitude"
    for block in district:
        latitude += block[latKey]
        longitude += block[longKey]
    return latitude/numBlocks,longitude/numBlocks

# geographicalCenter weighed by the population of each block
def centerOfPopulation(district):
    numBlocks = len(district)
    latitude = 0
    longitude = 0
    population = 0
    latKey = "Latitude"
    longKey = "Longitude"
    populationKey = 'Population'
    for block in district:
        blockPopulation = block[populationKey]
        latitude += block[latKey] * population
        longitude += block[longKey] * population
        population += blockPopulation
    return latitude/population, longitude/population

# return the moment of inertia for a district
# result has units person * km^2
def momentOfInertia(district):
    moment = 0
    centerLat,centerLong = centerOfPopulation(district)
    latKey = "Latitude"
    longKey = "Longitude"
    populationKey = 'Population'
    for block in district:
        moment += block[populationKey] * (distanceInKm(centerLat,centerLong,block[latKey], block[longKey]))**2
    return moment

def populationInDistrict(district):
    population = 0
    populationKey = 'Population'
    for block in district:
        population += block[populationKey]
    return population
        
#districts is a list of list of dictionary
def meanDistrictPopulation(districts):
    numDistricts = len(districts)
    population = 0
    populationKey = 'Population'
    for district in districts:
        population += populationInDistrict(district)
    return population / numDistricts

# districts is a list of list of dictionary
# norm is normalized
def populationEqualityL1Norm(districts):
    numDistricts = len(districts)
    meanPop = meanDistrictPopulation(districts)
    norm = 0
    for district in districts:
        norm += abs(populationInDistrict(district) - meanPop)
    normalizationFactor = 2 *(numDistricts - 1 ) * meanPop
    return norm/numDistricts

def populationEqualityL2Norm(districts):
    numDistricts = len(districts)
    meanPop = meanDistrictPopulation(districts)
    norm = 0
    for district in districts:
        norm += (populationInDistrict(district) - meanPop)**2
    normalizationFactor = 2 *(numDistricts - 1 ) * meanPop
    return norm/numDistricts

def populationEqualityL1Normalized(districts):
    numDistricts = len(districts)
    meanPop = meanDistrictPopulation(districts)
    norm = 0
    for district in districts:
        norm += abs(populationInDistrict(district) - meanPop)
    normalizationFactor = 2 *(numDistricts - 1 ) * meanPop
    return norm/normalizationFactor

def invCoeffVar(districts):
    numDistricts = len(districts)
    meanPop = meanDistrictPopulation(districts)
    norm = 0
    for district in districts:
        norm += (populationInDistrict(district) / meanPop - 1)**2
    return sqrt(norm/numDistricts)
def getFipsPerCounty(filename):
    # read the data
    with open(filename) as f:
        # read the data as a dictionary
        reader = csv.DictReader(f)
        data = [r for r in reader]
    return data
    
# get the political data for the counties
def getPoliDataByCounty(filename,fipsFileName):
    # read the data
    with open(filename) as f:
        # read the data as a dictionary
        reader = csv.DictReader(f)
        data = [r for r in reader]
    fipsData = getFipsPerCounty(fipsFileName)
    counties = []
    for fip in fipsData:
        county = {}
        numObamaVoters = 0
        numRomneyVoters = 0
        fipCode = fip['State(Fips)']+fip['County(Fips)']
        for votes in data:
            if(votes['FIPS'] != fipCode):
                continue
            # get rid of commas in the vote count
            numObamaVoters += int(votes['Obama vote'].replace(',',''))
            numRomneyVoters += int(votes['Romney vote'].replace(',',''))
        fip['Obama vote'] = numObamaVoters
        fip['Romney vote'] = numRomneyVoters
    return fipsData

# get u_i_j = number people in block i assigned to district j
def getNumBlockInDistrict(blocks):
    u= []
    for block in blocks:
        u_i_j = {}
        population = block['Population']
        district = int(block['Congressional District'])
        nd=1
        if(district==1):
            nd = 2
        # blocks have all their population assigned to a single district
        u_i_j[district] = population
        u_i_j[nd] = 0
        u_i_j['Id2'] = block['Id2']
        u.append(u_i_j)
    return u

def getNeighbors(shapefileDir):
    neighborsMap = {}
    sf = shapefile.Reader(shapefileDir);
    records = sf.records();
    # choose the "queen" neighbors. This means shapes are neighbors as long as they share a vertex.
    # alternative is the "rook", where shapes must share an edge.
    w = ps.queen_from_shapefile(shapefileDir+".shp");
    N = w.n;
    for i in range(N):
        # blockId is the field in index 4
        blockId = int(records[i][4]);
        # this var is a map containing the neighbors of block i, where the key is the neighbor, and value is the weight
        neighbors = w[i];
        # map everything by blockIds instead of indices in this list
        # since this ordering is different from the blocksList
        neighborList = []
        for n in neighbors.keys():
            neighborId = int(records[n][4]);
            neighborList.append(neighborId);
        neighborsMap[blockId] = neighborList;
    return neighborsMap

def getNeighborPairs(shapefileDir):
    neighborPairs = [];
    tmp = set();
    sf = shapefile.Reader(shapefileDir);
    records = sf.records();
    # choose the "queen" neighbors. This means shapes are neighbors as long as they share a vertex.
    # alternative is the "rook", where shapes must share an edge.
    w = ps.queen_from_shapefile(shapefileDir+".shp");
    N = w.n;
    for i in range(N):
        # blockId is the field in index 4
        blockId = int(records[i][4]);
        # this var is a map containing the neighbors of block i, where the key is the neighbor, and value is the weight
        neighbors = w[i];
        for n in neighbors.keys():
            neighborId = int(records[n][4]);
            # pairs are unordered, we don't want to double add pairs
            pair = (blockId,neighborId);
            revPair = (neighborId, blockId);
            if revPair in tmp:
                continue;
            tmp.add(pair);
            neighborPairs.append(pair);
    return neighborPairs

 # the neighborsMap is by blockId,
# return pairs by their index according the order in the blocks list
def getPairsFromMap(blocks,neighborsMap):
    pairs = [];
    indexMapping = {};
    for i in range(len(blocks)):
        blockId = blocks[i]['Id2']
        indexMapping[blockId] = i;
    for block in neighborsMap:
        index = indexMapping[block];
        neighborList = neighborsMap[block];
        for neighbor in neighborList:
            neighborIndex = indexMapping[neighbor];
            pairs.append((index,neighborIndex))
    return pairs

# Find the nodes that "flow" into node i
# This is finding all the pairs with i as the second value
def getFlowInto(i,pairs):
    matches = [x for x in pairs if pairs[1] == i]
    return matches


# Find the nodes that "flow" from node i
# This is finding all the pairs with i as the first value   
def getFlowOutof(i,pairs):
    matches = [x for x in pairs if pairs[0] == i]
    return matches
            


