#!/usr/bin/env python
# coding: utf-8

# # Usual Package Imports

# In[70]:


first_time = 0


# In[50]:


import networkx as nx, numpy as np, matplotlib.pyplot as plt, re, yaml,sys, shelve
import graphutils as gutils, make_graph_dict as mkgraph
from datetime import datetime,date
from collections import defaultdict
from functools import reduce
from importlib import reload
today = re.sub('-','_',str(date.today()))
today,nx.__version__


# In[35]:


WRITE = False


# # Change Directory - Revise for your file system.

# In[36]:


s = get_ipython().run_line_magic('pwd', '')
if not re.search('Downloads$',s):
    get_ipython().run_line_magic('cd', 'Downloads')


# ## Open a shelve and load a graph and a term dictionary

# In[100]:


with shelve.open('definitions') as f:
    graphdict = dict([(k,v) for k,v in f.items()])
print(graphdict.keys())


# In[106]:


term = 'Senior Reduction Amount'
term0 = 'Tranche Write-up Amount' # a second term
g.in_edges(term)
baseobjs = [graphdict[k] for k in 'graph term_dictionary termdefdict acrodict racrodict formuladict'.split()]
graph,termdict,termdefdict,acrodict,racrodict,formuladict = baseobjs


# In[120]:


print('\n\n-----\n\n'.join([str(obj[term]) for obj in (acrodict,termdict,termdefdict,formuladict)]))


# In[121]:


g = graph.copy()


# In[122]:


graph.__dict__.keys()


# In[123]:


list(g.in_edges('Senior Reduction Amount'))


# In[124]:


termdict['Credit Event Net Gain']
termdict['Senior Reduction Amount']


# # Eliminate Unecessary Nodes

# In[125]:


with open('nodes2delete.txt') as f: # background on how I selected these on request
    xnodes = f.read().split(' ; ')
xnodes


# In[126]:


for n in xnodes:
    g.remove_node(n)
    print('removed node %s' % n)


# # Draw Graph

# In[127]:


reload(gutils)
GU = gutils.GraphUtils
acro = gutils.acro


# In[128]:


acro('',reset = 1) # empty the cache
acrodict = dict([k,acro(k)] for k in termdict.keys()) # a dcitionary
racrodict = dict(((v,k) for k,v in acrodict.items()))
list(acrodict.items())[:5]#,racrodict['RCRNot']


# In[142]:


sra,mcet,rprp,twua = [acrodict[k] for k in 'Senior Reduction Amount;Minimum Credit Enhancement Test;Recovery Principal;Tranche Write-up Amount'.split(';')]
sra,mcet,rprp,twua


# In[143]:


graphs = {}
for k in (sra,mcet,rprp,twua):
    graphs[k] = GU.traceback(g,racrodict[k],num=1)


# In[144]:


def draw_g(g,sz = (15,10)):
    plt.figure(1,figsize=sz)
    nx.draw_networkx(g)
    plt.show()


# In[145]:


draw_g(graphs[sra])


# In[146]:


draw_g(graphs[mcet])


# In[147]:


draw_g(graphs[rprp])


# In[148]:


draw_g(graphs[twua])


# In[99]:


termdict['Minimum Credit Enhancement Test'],termdict['Group 1 Subordinate Percentage'],termdict['Group 1 Senior Percentage']


# ## Test the Graph

# In[134]:


term = 'Senior Reduction Amount'
term0 = 'Tranche Write-up Amount' # a second term
g.in_edges(term)


# In[135]:


[(edge[0],termdict[edge[0]]) for edge in g.in_edges('Senior Reduction Amount')]


# # Import our Custom Code for Tracing Back Definitions through other Definitions

# In[16]:


import graphutils
reload(graphutils)
gutils = graphutils.GraphUtils


# ## Test Traceback

# In[17]:


ar = []
lst = lambda x: ar.append(x)
lg = gutils.traceback(g.copy(),term0,num=5,it=0,f=lst)
print(len(lg.edges()),len(g.edges()),len(g.in_edges('Tranche Write-up Amount')))
#lg = gutils.traceback(g.copy(),'Senior Reduction Amount',num=3)
#draw_g(lg)
g.in_edges(term0),termdict[term0]


# In[18]:


ar.sort()
ar


# In[19]:


term0,termdict[term0]


# # A sketch of "memoizing"

# In[20]:


def memo(f):
    """Decorator that caches the return value for each call to f(args).
    Then when called again with same args, we can just look it up."""
    cache = {}
    def _f(*args):
        try:
            if args in cache: print("found it")
            return cache[args]
        except KeyError:
            cache[args] = result = f(*args)
            return result
        except TypeError:
            # some element of args can't be a dict key
            return f(args)
    return _f
@memo # f = memo(f) = _f
def f(x):return x*2
f(3)
f(3)


# # Converting definition lists to readable text and to formulae

# # A General Recursion with an (Ac)Cumulator

# In[21]:


# decorate a function (see below) with an (ac)cumulator
# control the number of recursion with limit = num
# include a function newargs_ for generating new arguments from the old args
def decf(f,newargs_,cum = None,num=2, it = 0):
    cum = cum or [] # cum is the accumulation of the recursion
    def f_(*args,num=num,it=it):
        #print('args',args) # DEBUG
        cum.append(f(*args)) # cache the result of f applied to the arguments
        if it <= num: # recurse if under the limit
            newargs = newargs_(*args)
            for args in newargs:
                f_(*args,num=num,it=it+1)
    return f_,cum

# A toy example with trivial functions as arguments
# We need to have access to the cumulator so the decorator returns it 
func,cum = decf(lambda x: x, lambda x: ([x+1],))
print(cum) # empty
func(0)
print(cum )
func(0)
print(cum)
# the accumulator is persistent
print(cum) # accumulator
func,cum = decf(lambda x: x, lambda x: ([x+1],))
# the (only) way to get a new accumulator
func(0)
print(cum)


# In[22]:


dlm = (': ',', ','\n') # various delimiters for use in string formation


# ## Turn a definition into a string

# In[23]:


get_def = lambda term,g=g,dct = termdict: dlm[0].join((term, dlm[-1].join((tpl[1] for tpl in dct[term]))))
get_def(term0)


# In[39]:


get_def('Tranche Write-up Amount')


# ## Indent the definition string for readability

# In[47]:


# indent based on level
get_def_indent = lambda x,indent=0: '\t'*indent + get_def(x)
get_def_indent(term0,2)


# ## Using the recursion to write an indented dictionary of terms

# In[41]:


func,cum = decf(get_def_indent,lambda term,it: ([indedge[0],it+1]
                for indedge in g.in_edges(term)))
print(cum) # empty cumulator
func(term, 0,num = 0) # fill it
cum[:5]


# In[42]:


# using the recursion 
func,cum = decf(get_def_indent,lambda term,it: ([indedge[0],it+1]
                for indedge in g.in_edges(term)))
print(cum)
func(term0,0)
cum[:5]


# In[25]:


if 0 or WRITE: # Writing a term dictionary with indented definitions: execute by replace 0 with 1
    with open('term_dict.txt','w') as f:
        for line in cum: f.write(line+"\n")


# ## Using the recursion to write a series of indented dependent formulae

# In[26]:


termdict[term0]


# In[27]:


#How to make a tentative formula, for editing and encoding
get_formula = lambda term,line=False,g=g: dlm[0].join((gutils.acro(term),'['+term+']')) + dlm[0] +'('+ dlm[1].join((gutils.acro(edge[0]) 
                                                        for edge in g.in_edges(term)) )+')'+ ("\n" if line else '')
get_formula(term0,1)


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

# In[28]:


# for easier reading, we indent
get_formula_indent = lambda x,indent=0: '\t'*indent + get_formula(x)
get_formula_indent(term0,2)


# In[29]:


# set up temporary formulae for all definitions needed for term = Tranche Write-up Amount
func,cum_form = decf(get_formula_indent,lambda term,it: ([indedge[0],it+1]
                for indedge in g.in_edges(term)))
func(term0,0)


# In[30]:


# write a formula dictionary: execute by replace 0 with 1
if 0:
    with open('formula_dict.txt','w') as f:
        for line in cum_form: f.write(line+"\n")


# # Revising Graph - Delete nodes for terms not used in formulae 

# We've seen some nodes, like Credit Event which, although important, do not appear in formulae.  We want to delete them systematically from our graph for purposes of creating the formulae which will become our pseudocode.  This will allow us to delete once, and have the deletion propogate through all our definitions, which is a benefit of starting with a graph.

# In[31]:


cleangraph = graph.copy() # want to preserve the original for future reference.

terms2delete = 'Reference Obligations;Reference Pool;Clearstream;Group 2 Termination Date;Participants'.split(';')
terms2delete += 'Global Agency Agreement;Wells Fargo Bank;Treasury;BofA Merrill;Group 1 Notes'.split(';')
terms2delete += 'Group Termination Date; Indirect Participants;Euroclear Participants'.split(';')
terms2delete += 'Closing Date;Note Owners;Dealer;Dealer Agreement;Warrant;Wells Fargo'.split(';')
terms2delete += 'Credit Suisse;JP Morgan;Barclays;Clearstream International'.split(';')
terms2delete +='Citigroup;Euroclear;Group 1 Termination Date'.split(';')
terms2delete+= 'Liquidation Proceeds;Credit Event;Payment Date;Business Day;Reporting Period;Period;Credit Event Reference Obligation'.split(';')

print(len(cleangraph))

for term_ in terms2delete:
    if term_ in cleangraph.nodes():
        print("Removing node %s" % term_)
        cleangraph.remove_node(term_)
print(len(cleangraph))


# In[32]:


make_draw_graph(term0,cleangraph,(18,5))


# ## Set up some automated acronyms for easy reading

# In[33]:


def nocollision(f):
    cache = []
    # this time, we allow ourselves to clear the cache manually
    def _f(arg,reset=False):
        if reset: 
            cache.clear()
            return
        #print(cache)
        i = 0
        baseresult = f(arg)
        result = baseresult
        while result in cache:
            assert i < 100, 'Problems with arg %s, f(arg) %s, and i %i' %(arg,result,i)
            #print(result)
            result = baseresult + str(i)
            i+=1
        cache.append(result)
        return result
    return _f
print(term0)

@nocollision
def acro(term): return gutils.acro(term)

print([acro(term_) for term_ in 'Principal Amount;Private Amount;Press Amonia'.split(';')])
print([acro(term_) for term_ in 'Principal Amount;Private Amount;Press Amonia'.split(';')])
acro(term0,1) # resets ('clears') the cache to an empty list
print([acro(term_) for term_ in 'Principal Amount;Private Amount;Press Amonia'.split(';')])


# ## Make an acronym dictionary and reverse dictionary.

# In[34]:


acro('',reset = 1) # empty the cache
acrodict = dict([k,acro(k)] for k in termdict.keys()) # a dcitionary
racrodict = dict(((v,k) for k,v in acrodict.items()))
list(acrodict.items())[:5]#,racrodict['RCRNot']


# In[35]:


len(cleangraph),len(graph)


# In[36]:


term0,'full graph',graph.in_edges(term0),'cleaned graph', cleangraph.in_edges(term0)


# In[37]:


cleangraph.in_edges('Credit Event')


# In[38]:


# Confirm values 
term = 'Senior Reduction Amount'
term0 = 'Tranche Write-up Amount' # a second term
cleangraph.in_edges(term)


# In[54]:


#How to make a tentative formula, for editing and encoding
get_formula = lambda term,line=False,g=cleangraph: dlm[0].join((gutils.acro(term),'['+term+']')) + dlm[0] +'('+ dlm[1].join((gutils.acro(edge[0]) 
                                                        for edge in g.in_edges(term)) )+')'+ ("\n" if line else '')
print(get_formula(term,1))
print(get_formula(term,1,g=g))
for edge in g.in_edges(term): print(acrodict[edge[0]]," : ", edge[0])
print(('\n' * 2).join((termdict[term][0][1],gutils.acro('Senior Percentage'),termdict['Group 1 Senior Percentage'][0][1])))

get_formula_indent = lambda x,indent=0: '\t'*indent + get_formula(x)
term0, get_formula_indent(term0,2)


# In[101]:


func,cum_form = decf(lambda x: x, lambda x: ([x+1],))
print(cum_form)


# In[102]:


func,cum_form = decf(get_formula_indent,lambda term,it: ([indedge[0],it+1]
                for indedge in cleangraph.in_edges(term)))
func(term0,0)
cum_form[:3]


# ## Write the pre-formula to file

# In[106]:


if 1:
    with open('formula_dict_clean0.txt','w') as f:
        for line in cum_form: f.write(line+"\n")


# ## Check that empty formulae should be empty

# In[104]:


for term_ in 'LiqPr CrEvUPB LiqPr'.split():
    print(termdict[racrodict[term_]])


# In[108]:


get_formula('LiqPr')


# In[105]:


termdict[racrodict['ModEv']]


# In[107]:


get_formula('ModEv')


# In[ ]:




