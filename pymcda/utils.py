from __future__ import division
from __future__ import print_function
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
import random
from pymcda.types import AlternativesAssignments



def add_errors_in_assignments(aa, category_ids, errors_pc):
    """ Introduces some noise in the given assignment examples.
    (which consists in interchanging adjacent categories of alternatives)"""
    
    n = int(len(aa)*errors_pc)
    aa_erroned = random.sample(list(aa), n)
    l = AlternativesAssignments([])
    
    for a in aa_erroned:
        cat = a.category_id
        new_cat = a.category_id
        cats = set()
        if category_ids.index(new_cat)-1>=0:
            cats.add(category_ids[category_ids.index(new_cat)-1])
        if category_ids.index(new_cat)+1<len(category_ids):
            cats.add(category_ids[category_ids.index(new_cat)+1])

        while new_cat == cat:
            new_cat = random.sample(cats, 1)[0]
        a.category_id = new_cat
        l.append(a)

    return l

def add_errors_in_assignments_proba(aa, category_ids, proba):
    l = AlternativesAssignments([])

    for a in aa:
        r = random.random()
        if r <= proba:
            cat = a.category_id
            new_cat = a.category_id
            while new_cat == cat:
                new_cat = random.sample(category_ids, 1)[0]
            a.category_id = new_cat
            l.append(a)

    return l


def compute_ca(aa, aa2, alist=None):
    if alist is None:
        alist = aa.keys()

    total = len(alist)
    ok = 0
    for aid in alist:
        af = aa(aid)
        af2 = aa2(aid)
        if af == af2:
            ok += 1

    return ok / total

def compute_ca_good(aa, aa2, alist=None):
    if alist is None:
        alist = aa.keys()

    total = 0
    ok = 0
    for aid in alist:
        af = aa(aid)
        af2 = aa2(aid)
        if af == 'recommend' or 'cat1':
            total += 1
            if af2 == 'recommend' or 'cat1':
                ok += 1
    if total == 0:
        return 1
    return ok / total

