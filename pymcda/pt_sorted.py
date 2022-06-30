from __future__ import division
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
import bisect
from pymcda.types import AlternativePerformances



class SortedPerformanceTable():

    def __init__(self, pt):
        """ Initializes a sorted performance table instance """
        
        self.pt = pt
        self.__sort()
        self.__build_index()

    def __getitem__(self, id):
        return self.pt[id]

    def __sort(self):
        self.cids = next(self.pt.itervalues()).performances.keys()
        self.n = len(self.pt)
        self.sorted_pt = { cid: list() for cid in self.cids }
        for ap in self.pt:
            p = ap.performances
            aid = ap.id
            for cid in self.cids:
                bisect.insort(self.sorted_pt[cid], (p[cid], aid))

    def __build_index(self):
        self.sorted_values = { cid: list() for cid in self.cids }
        self.sorted_altid = { cid: list() for cid in self.cids }
        for cid in self.cids:
            self.sorted_values[cid] = [ r[0] for r in self.sorted_pt[cid] ]
            self.sorted_altid[cid] = [ r[1] for r in self.sorted_pt[cid] ]

    def get_all(self, cid):
        return self.sorted_altid[cid], self.sorted_values[cid]

    def get_below(self, cid, val, b=True):
        if b is True:
            i = bisect.bisect(self.sorted_values[cid], val)
        else:
            i = bisect.bisect_left(self.sorted_values[cid], val)
        return self.sorted_altid[cid][:i], self.sorted_values[cid][:i]

    def get_above(self, cid, val, a=True):
        if a is True:
            i = bisect.bisect_left(self.sorted_values[cid], val)
        else:
            i = bisect.bisect(self.sorted_values[cid], val)
        return self.sorted_altid[cid][i:], self.sorted_values[cid][i:]

    def get_middle(self, cid, val_l, val_r, l=True, r=True):
        """ Constructs a middle value between two alternatives values 
        following a given criterion """
        
        if val_l > val_r:
            val_l, val_r = val_r, val_l
            l, r = r, l
            reverse = True
        else:
            reverse = False

        if l is True:
            i = bisect.bisect_left(self.sorted_values[cid], val_l)
        else:
            i = bisect.bisect(self.sorted_values[cid], val_l)
        if r is True:
            i2 = bisect.bisect(self.sorted_values[cid], val_r)
        else:
            i2 = bisect.bisect_left(self.sorted_values[cid], val_r)

        altids = self.sorted_altid[cid][i:i2]
        values = self.sorted_values[cid][i:i2]

        if reverse is True:
            altids.reverse()
            values.reverse()

        return altids, values

    def get_below_len(self, cid, val, r=True):
        if r is True:
            return bisect.bisect(self.sorted_values[cid], val)
        else:
            return bisect.bisect_left(self.sorted_values[cid], val)

    def get_above_len(self, cid, val, l=True):
        if l is True:
            return self.n - bisect.bisect_left(self.sorted_values[cid], val)
        else:
            return self.n - bisect.bisect(self.sorted_values[cid], val)

    def get_middle_len(self, cid, val_l, val_r, l=True, r=True):
        if val_l > val_r:
            val_l, val_r = val_r, val_l
            l, r = r, l

        if l is True:
            i = bisect.bisect_left(self.sorted_values[cid], val_l)
        else:
            i = bisect.bisect(self.sorted_values[cid], val_l)
        if r is True:
            i2 = bisect.bisect(self.sorted_values[cid], val_r)
        else:
            i2 = bisect.bisect_left(self.sorted_values[cid], val_r)

        return i2-i

    def get_worst_ap(self):
        a = AlternativePerformances('worst', {})
        for cid in self.cids:
            a.performances[cid] = self.sorted_values[cid][0]
        return a

    def get_best_ap(self):
        a = AlternativePerformances('best', {})
        for cid in self.cids:
            a.performances[cid] = self.sorted_values[cid][-1]
        return a

