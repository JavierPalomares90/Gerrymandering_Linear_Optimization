
# coding: utf-8

# In[1]:


import utils
dataDir = "../census_data/"
# file with population data
populationFile = "/DEC_10_PL_P3_with_ann.csv"
# file with geographical data
geoFile = "/DEC_10_PL_G001_with_ann.csv"
dirs = utils.getSubdirs(dataDir)
blocks = []
# get all of the blocks into one list
for d in dirs:
    block,popData,geoData = utils.getBlocks(dataDir + d + populationFile,dataDir + d + geoFile)
    for b in block:
        blocks.append(b)
# read the political data
fipsDataFile = "../census_data/st44_ri_cou.txt";
poliDataFile = "../political_data/US_elect_county.csv";
counties = utils.getPoliDataByCounty(poliDataFile,fipsDataFile);


# In[5]:


#This is a population block
print(blocks[3])

# In[2]:


#who
# This is a vote by county
print(counties[0])

