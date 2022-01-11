#!/usr/bin/env python
# coding: utf-8

# In[1]:


first_time = 0


# # Change Directory - Revise for your file system.

# In[2]:


import re
s = get_ipython().run_line_magic('pwd', '')
if not re.search('Downloads$',s):
    get_ipython().run_line_magic('cd', 'Downloads')


# # Usual Package Imports

# In[3]:


import networkx as nx, numpy as np, matplotlib.pyplot as plt, sys, shelve
import graphutils as gutils, make_graph_dict as mkgraph
from datetime import datetime,date
from collections import defaultdict
from functools import reduce
from importlib import reload
today = re.sub('-','_',str(date.today()))
today,nx.__version__


# In[4]:


WRITE = False


# ## Open a shelve and load a graph and a term dictionary

# In[5]:


with shelve.open('definitions') as f:
    graphdict = dict([(k,v) for k,v in f.items()])
print(graphdict.keys())


# In[6]:


term = 'Senior Reduction Amount'
term0 = 'Tranche Write-up Amount' # a second term
baseobjs = [graphdict[k] for k in 'graph term_dictionary termdefdict acrodict racrodict formuladict'.split()]
graph,termdict,termdefdict,acrodict,racrodict,formuladict = baseobjs


# In[7]:


print('\n\n-----\n\n'.join([str(obj[term]) for obj in (acrodict,termdict,termdefdict,formuladict)]))


# In[8]:


g = graph.copy()


# In[9]:


graph.__dict__.keys()


# In[10]:


list(g.in_edges(term))


# In[11]:


termdict['Credit Event Net Gain']
termdict[term]


# In[12]:


g.in_edges(term)


# # Eliminate Unecessary Nodes

# In[13]:


with open('nodes2delete.txt') as f: # background on how I selected these on request
    xnodes = f.read().split(' ; ')
xnodes


# In[14]:


for n in xnodes:
    g.remove_node(n)
    print('removed node %s' % n)


# # Draw Graph

# In[15]:


reload(gutils)
GU = gutils.GraphUtils
acro = gutils.acro


# In[16]:


acro('',reset = 1) # empty the cache
acrodict = dict([k,acro(k)] for k in termdict.keys()) # a dcitionary
racrodict = dict(((v,k) for k,v in acrodict.items()))
list(acrodict.items())[:5]#,racrodict['RCRNot']


# In[17]:


sra,mcet,rprp,twua = [acrodict[k] for k in 'Senior Reduction Amount;Minimum Credit Enhancement Test;Recovery Principal;Tranche Write-up Amount'.split(';')]
sra,mcet,rprp,twua


# In[18]:


graphs = {}
for k in (sra,mcet,rprp,twua):
    graphs[k] = GU.traceback(g,racrodict[k],num=1)


# In[19]:


def draw_g(g,sz = (15,10)):
    plt.figure(1,figsize=sz)
    nx.draw_networkx(g)
    plt.show()


# In[20]:


draw_g(graphs[sra])


# In[21]:


draw_g(graphs[mcet])


# In[22]:


draw_g(graphs[rprp])


# In[23]:


draw_g(graphs[twua])


# In[24]:


termdict['Minimum Credit Enhancement Test'],termdict['Group 1 Subordinate Percentage'],termdict['Group 1 Senior Percentage']


# In[25]:


print(formuladict[term0])


# TrWrAm: [Tranche Write-up Amount]: max(PrRecAm - PrLosAm,0)
# 
# 

# ## How the formula works
# 
# TrWrAm: [Tranche Write-up Amount]: max (PrRecAm - PrLosAm,0)
# 
# Suppose PrRecAm, PrLosAm = 10,5
# 
# Then TrWrAm = max( 10 - 5,0) = 5
# 
# But suppose  PrRecAm, PrLosAm = 5,10
# 
# Then TrWrAm = max(5 - 10 ,0) = 0

# In[26]:


print('''TrWrAm: [Tranche Write-up Amount]: max (PrRecAm - PrLosAm,0)
	PrRecAm: [Principal Recovery Amount]: sum(CrEvNetLos,CrEvNetGa,RepWarSetAm)
		CrEvNetGa: [Credit Event Net Gain]: max(NetLiqPr – sum(CrEvUPB,see_def),0)
			CrEvUPB: [Credit Event UPB]: (CrEv, PayDat, Per, RepPer, CrEvRefObl)
			NetLiqPr: [Net Liquidation Proceeds]:  sum(LiqPr, MorInsCrAm,see_def)
		NetLiqPr: [Net Liquidation Proceeds]: sum(LiqPr,MorInsCrAm), less expensese
			LiqPr: [Liquidation Proceeds]: (CrEvRefObl, CrEv)
			MorInsCrAm: [Mortgage Insurance Credit Amount]: (CrEvRefObl, CrEv)
		LiqPr: [Liquidation Proceeds]: (CrEvRefObl, CrEv)
			CrEvRefObl: [Credit Event Reference Obligation]: (CrEv,
		CrEvNetLos: [Credit Event Net Loss]: max(sum(CrEvUPB,see_def) - NetLiqPr, 0)
			NetLiqPr: [Net Liquidation Proceeds]: above
			CrEvUPB: [Credit Event UPB]: above
	PrLosAm: [Principal Loss Amount]: sum(CrEvNetLos, cramdowns, parts of ModLosAm)
		ModLosAm: [Modification Loss Amount]: (CurAccRat, ModEv, OrAccRat, PayDat)
			CurAccRat: [Current Accrual Rate]: (PayDat, ModEv)
			ModEv: [Modification Event]: (PayDat, OrAccRat, ModLosAm, CurAccRat)
			OrAccRat: [Original Accrual Rate]: (CutDat)
			PayDat: [Payment Date]: (BusDay, GlAg, RecDat, BoNot, DefNot)
		CrEvNetLos: [Credit Event Net Loss]: (NetLiqPr, CurAccRat, CrEv, CrEvUPB, CrEvRefObl, LiqPr)
			NetLiqPr: [Net Liquidation Proceeds]: above
			CrEvUPB: [Credit Event UPB]: above
			LiqPr: [Liquidation Proceeds]: above
		CrEvRefObl: [Credit Event Reference ''')

