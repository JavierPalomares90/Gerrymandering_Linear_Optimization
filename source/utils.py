
# coding: utf-8

# In[6]:


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
def getPopulationPerBlock(filename):
    with open(filename) as f:
        # skip the first "header" row
        next(f);
        # read the data as a dictionary
        reader = csv.DictReader(f)
        data = [r for r in reader]
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
    for i in range(numCensusBlocks):
        block = {};
        geoId = int(data[i][geoIdKey]);
        block[geoIdKey] = geoId;
        blockId,blockGroupId,censusTract,county,state = getGeography(data[i][geoKey]);
        block[blockKey] = blockId;
        totalPopulation = getTotalPopulation(data[i]);
        block[populationKey] = totalPopulation;
        block[countyKey] = county;
        block[censusTractKey] = censusTract;
        populationBlocks.append(block)
    return populationBlocks,data
def getSubdirs(dir):
    #"Get a list of immediate subdirectories"
    return next(os.walk(dir))[1]

