import networkx as nx

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

@nocollision
def acro(term): return GraphUtils.acro(term)



class GraphUtils(object):
    @classmethod
    def traceback(kls,g,n,g_ = None,num = 0, it=0,f = None):
        if it==0 or not g_: g_ = nx.DiGraph()
        if it == 0: assert not g_.edges()
        #print('starting at',str((num,it)),str(len(g_.nodes())))
        newedges = list(set([edge for edge in list(g.in_edges(n)) 
                             if  edge not in g_.edges() and edge[0] not in g_.nodes()])).copy()
        if newedges and it < num:
            #print(num,it)
            g_.add_edges_from(newedges)
            for edge in newedges:
                g_ = GraphUtils.traceback(g,edge[0],g_,num=num,it=it+1,f=f)
                if f: f((it,edge[0]))
        return g_
        
    @classmethod
    def acro_w(kls,word):
        return word if len(word)< 4 else (word[:2] if word[2].lower() in 'aeiou' else word[:3])      
    @classmethod
    def acro(kls,words):
        return ''.join((kls.acro_w(word) for word in words.split()))