
# coding: utf-8

# In[1]:


import utils
dataDir = "../census_data/";
annFile = "/DEC_10_PL_P3_with_ann.csv"
dirs = utils.getSubdirs(dataDir)
blocks = []
# get all of the blocks into one list
for d in dirs:
    block,data = utils.getPopulationPerBlock(dataDir + d + annFile)
    for b in block:
        blocks.append(b)


# In[2]:


#This is a population block
blocks[3]

