
# coding: utf-8

# In[1]:


import utils
dataDir = "../census_data/";
# file with population data
populationFile = "/DEC_10_PL_P3_with_ann.csv"
# file with geographical data
geoFile = "/DEC_10_PL_G001_with_ann.csv";
dirs = utils.getSubdirs(dataDir)
blocks = []
# get all of the blocks into one list
for d in dirs:
    block,popData,geoData = utils.getBlocks(dataDir + d + populationFile,dataDir + d + geoFile)
    for b in block:
        blocks.append(b)


# In[2]:


#This is a population block
blocks[3]


# In[2]:


who

