
# coding: utf-8

# In[4]:


# coding: utf-8

# In[1]:

from gurobipy import *
import utils

def readData():
    dataDir = "../census_data/";
    # file with population data
    populationFile = "/DEC_10_PL_P3_with_ann.csv"
    # file with geographical data
    geoFile = "/DEC_10_PL_G001_with_ann.csv";
    # file with the congressional districts per block
    cdFile = "../cd115/National_CD115.txt"
    dirs = utils.getSubdirs(dataDir)
    blocks = []
    # get all of the blocks into one list
    for d in dirs:
        block,popData,geoData = utils.getBlocks(dataDir + d + populationFile,dataDir + d + geoFile,cdFile)
        for b in block:
            blocks.append(b)
    # read the political data
    fipsDataFile = "../census_data/st44_ri_cou.txt";
    poliDataFile = "../political_data/US_elect_county.csv";
    counties = utils.getPoliDataByCounty(poliDataFile,fipsDataFile);
    return (blocks, counties)

def assign():
    # gurobi network flow example: http://www.gurobi.com/documentation/7.5/examples/netflow_py.html
    #
    # cost function f(i,j) is the euclidian distance from block i to district j times population of block i
    # we can create a cost matrix f with rows being blocks and columns being districts
    # we can write the edges u as constrained to 0 or 1 indicating whether block i is assigned to district j
    # 


    # Create optimization model
    m = Model('netflow')

# In[5]:
(blocks, counties) = readData()
#This is a population block
print(blocks[3])
# This is a vote by county
#print(counties[0])


