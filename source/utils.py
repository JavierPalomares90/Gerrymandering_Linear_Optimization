
# coding: utf-8

# In[27]:


import re
import csv
import os
def getGeography(s):
    strings = s.split(',')
    if(len(strings) < 5):
        return
    #get the block number and block group using regEx
    blockId = int(re.search(r'\d+', strings[0]).group())
    blockGroupId = int(re.search(r'\d+', strings[1]).group())
    censusTract = int(re.search(r'\d+', strings[2]).group())
    county = strings[3]
    state = strings[4]
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

def getBlocks(popFilename,geoFilename):
    # read the poulation data
    with open(popFilename) as f:
        # skip the first "header" row
        next(f);
        # read the data as a dictionary
        reader = csv.DictReader(f)
        data = [r for r in reader]
    # read the geo data
    with open(geoFilename) as geoFile:
        # skip the first "header" row
        next(geoFile);
        # read the data as a dictionary
        geoReader = csv.DictReader(geoFile)
        geoData = [r for r in geoReader]
    numCensusBlocks = len(data)
    # populationBlocks is a list of maps, one map per census block
    populationBlocks = [];
    # geography keys
    # use the second id as the identifier
    geoIdKey = 'Id2';
    geoKey = 'Geography';
    blockKey = 'Block';
    populationKey = 'Population'
    countyKey = 'County';
    censusTractKey = "Census_Tract"
    latKey = "Latitude";
    latLookup = "AREA CHARACTERISTICS - Internal Point (Latitude)";
    longLookup = "AREA CHARACTERISTICS - Internal Point (Longitude)";
    longKey = "Longitude";
    fipsCodeKey = "FIPS Code";
    stateLookup = "GEOGRAPHIC AREA CODES - State (FIPS)";
    countyLookup = "GEOGRAPHIC AREA CODES - County";
    for i in range(numCensusBlocks):
        block = {};
        geoId = int(data[i][geoIdKey]);
        # make sure the id's match
        if(geoId != int(geoData[i][geoIdKey])):
            raise("There is a mismatch in ids in files:" + popFilename + " " + geoFilename)
            return
        block[latKey] = geoData[i][latLookup];
        block[longKey] = geoData[i][longLookup];
        block[geoIdKey] = geoId;
        blockId,blockGroupId,censusTract,county,state = getGeography(data[i][geoKey]);
        block[blockKey] = blockId;
        totalPopulation = getTotalPopulation(data[i]);
        block[populationKey] = totalPopulation;
        block[countyKey] = county;
        block[censusTractKey] = censusTract;
        # Add lat/long data
        block[latKey] = geoData[i][latLookup];
        block[longKey] = geoData[i][longLookup];
        # Add fips code
        state = int(geoData[i][stateLookup]);
        county = int(geoData[i][countyLookup]);
        fipsCode = getFipsCountyCode(state, county);
        block[fipsCodeKey] = fipsCode;
        populationBlocks.append(block)    
    return populationBlocks,data,geoData
def getSubdirs(dir):
    #"Get a list of immediate subdirectories"
    return next(os.walk(dir))[1]
# compute the distance between 2 lat/long points measured in degrees using the haversine formulae
import math
def distanceInKm(lat1, lon1, lat2, lon2):
    earthDiameterKm = 12742;
    p = math.pi/180 
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return earthDiameterKm * asin(sqrt(a))

# return the block if a block is in a district. False otherwise
def blockInDistrict(block,district):
    idKey = 'Id2'
    blockId = block[idKey];
    return next((item for item in district if item[idKey] == blockId), False)
# measure of compactness of a district
# obtained by computing the percentange of sites in the circle centered at s 
# of radius r that are not in the district
def circularCompactness(blocks,district,radius,centerLong,centerLat):
    numBlocks = len(blocks);
    numBlocksInDistrict = len(district);
    # the number of blocks within the 
    numInRadius = 0;
    numNotInD = 0;
    latKey = "Latitude";
    longKey = "Longitude";
    for block in blocks:
        latitude = block[latKey];
        longitude = block[longKey];
        if( distanceInKm(latitude,longitude,centerLong,centerLat) <= radius):
            numInRadius += 1;
            if(blockInDistrict(block,district) == False):
                numNotInD += 1;
    return numNotInD / numInRadius;

#returns the latitude and longitude of the geographical center of a district
def geographicalCenter(district):
    numBlocks = len(district);
    latitude = 0;
    longitude = 0;
    latKey = "Latitude";
    longKey = "Longitude";
    for block in district:
        latitude += block[latKey];
        longitude += block[longKey];
    return latitude/numBlocks,longitude/numBlocks;

# geographicalCenter weighed by the population of each block
def centerOfPopulation(district):
    numBlocks = len(district);
    latitude = 0;
    longitude = 0;
    population = 0;
    latKey = "Latitude";
    longKey = "Longitude";
    populationKey = 'Population'
    for block in district:
        blockPopulation = block[populationKey]
        latitude += block[latKey] * population;
        longitude += block[longKey] * population;
        population += blockPopulation;
    return latitude/population, longitude/population

# return the moment of inertia for a district
# result has units person * km^2
def momentOfInertia(district):
    moment = 0
    centerLat,centerLong = centerOfPopulation(district);
    latKey = "Latitude";
    longKey = "Longitude";
    populationKey = 'Population'
    for block in district:
        moment += block[populationKey] * (distanceInKm(centerLat,centerLong,block[latKey], block[longKey]))**2
    return moment

def populationInDistrict(district):
    population = 0;
    populationKey = 'Population';
    for block in district:
        population += block[populationKey];
    return population
        
#districts is a list of list of dictionary
def meanDistrictPopulation(districts):
    numDistricts = len(districts);
    population = 0;
    populationKey = 'Population';
    for district in districts:
        population += populationInDistrict(district);
    return population / numDistricts;

# districts is a list of list of dictionary
# norm is normalized
def populationEqualityL1Norm(districts):
    numDistricts = len(districts)
    meanPop = meanDistrictPopulation(districts);
    norm = 0
    for district in districts:
        norm += abs(populationInDistrict(district) - meanPop)
    normalizationFactor = 2 *(numDistricts - 1 ) * meanPop;
    return norm/numDistricts;

def populationEqualityL2Norm(districts):
    numDistricts = len(districts)
    meanPop = meanDistrictPopulation(districts);
    norm = 0
    for district in districts:
        norm += (populationInDistrict(district) - meanPop)**2
    normalizationFactor = 2 *(numDistricts - 1 ) * meanPop;
    return norm/numDistricts;

def populationEqualityL1Normalized(districts):
    numDistricts = len(districts)
    meanPop = meanDistrictPopulation(districts);
    norm = 0
    for district in districts:
        norm += abs(populationInDistrict(district) - meanPop)
    normalizationFactor = 2 *(numDistricts - 1 ) * meanPop;
    return norm/normalizationFactor

def invCoeffVar(districts):
    numDistricts = len(districts)
    meanPop = meanDistrictPopulation(districts);
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
    fipsData = getFipsPerCounty(fipsFileName);
    counties = [];
    for fip in fipsData:
        county = {};
        numObamaVoters = 0;
        numRomneyVoters = 0;
        fipCode = fip['State(Fips)']+fip['County(Fips)']
        for votes in data:
            if(votes['FIPS'] != fipCode):
                continue;
            # get rid of commas in the vote count
            numObamaVoters += int(votes['Obama vote'].replace(',',''));
            numRomneyVoters += int(votes['Romney vote'].replace(',',''));
        fip['Obama vote'] = numObamaVoters;
        fip['Romney vote'] = numRomneyVoters;
    return fipsData

        

