import networkx as nx, numpy as np, matplotlib.pyplot as plt, re, yaml,sys, shelve
from datetime import datetime,date
from collections import defaultdict
from functools import reduce
from importlib import reload
today = re.sub('-','_',str(date.today()))
today

quotere = '''["']'''
quoted_term_re = quotere + '(\w[\w \-]*\w)\W?' + quotere # all quoted terms
quoted_term_re_mod = quotere + '\w[\w \-]*\W?' + quotere # all quoted terms
start_definition_re = '^\s*([Tt]he|[Aa])? *(' + quoted_term_re + ')'#formal definition at a line's beginning
#altdefinition_re =  quotere + '?([\w\s\-]+)' + quotere + '?\.\.\.+' # works with findall to get the defined term
altdefinition_re =  '([\w \-]+\w)' + ' *\.\.\.+' # works with findall to get the defined term
altdefinitionsplit_re =  '[\w \-]+\.\.\.+ *' # for splitting

compact = lambda l: [e for e in l if e]
get_words = lambda s: re.split('[^\w\-]',s)
numwords = lambda s: len(get_words(s))
initcap = lambda w: w[0] and w[0] == w[0].upper()

test_initial_capitals_in_term = lambda x: x.strip() and DictMaker.icappc(x.strip()) > 0.9
get_all_defined_terms = lambda regex: lambda s:list(filter(test_initial_capitals_in_term,re.findall(regex,s)))
get_all_quoted_terms = get_all_defined_terms(quoted_term_re)

# a convenience function to identify True object values
idty = lambda x: x


class DictMaker(object):

    sent_rex = '''(\.)\s+(['"A-Z1-9])'''
    seglength = 3
    
    @classmethod
    def partition(kls,l,f):
        a,b = [],[]
        for e in l: (a if f(e) else b).append(e)
        return a,b

    @classmethod
    def icappc(kls,s):
        wds = compact(get_words(s))
        total = len(wds)
        if total > 0: 
            up,low = kls.partition(wds,initcap)
            return float(len(up))/total

    
    def __init__(self,sent_rex=None,seglength=None):
        self.srex = sent_rex or self.sent_rex
        self.seglength = seglength or self.seglength
        self.segindex = self.seglength-1
        
    # returns a period, a capital or number
    # and the split text.  See the assertions below    
    def get_sentences(self,s,return_result = True):
        parts = re.split(self.srex,s)
        lastInitial,self.sentences =     '',[]
        for i in range(0,len(parts)- self.seglength,self.seglength):
            currentparts = parts[i:i+self.seglength]
            current = ''.join(currentparts[:self.segindex])
            if lastInitial: current = lastInitial + current
            self.sentences.append(current)
            lastInitial = currentparts[self.segindex]
        self.sentences.append(lastInitial + parts[-1])
        if return_result: return self.sentences


def tests():
    print("Running tests")
    test_s = 'start here. This is one. Here is another. and here a third.'
    dmkr = DictMaker()
    print(dmkr.seglength)
    assert dmkr.get_sentences(test_s,return_result = True).__repr__() == "['start here.', 'This is one.', 'Here is another. and here a third.']"
    assert test_initial_capitals_in_term('Senior Reduction Amount')
    assert not test_initial_capitals_in_term('Senior Reduction amount')
    
if __name__ == "__main__": tests()