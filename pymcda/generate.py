from __future__ import division
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
import random
from pymcda.electre_tri import MRSort
from pymcda.types import Alternative, Alternatives
from pymcda.types import AlternativePerformances, PerformanceTable
from pymcda.types import Criterion, Criteria
from pymcda.types import CriterionValue, CriteriaValues
from pymcda.types import Category, Categories
from pymcda.types import CategoryProfile, CategoriesProfiles, Limits
import numpy as np



def generate_alternatives(number, prefix = 'a', names = None):
    """ Generates alternatives instances """
    
    alts = Alternatives()
    for i in range(number):
        aid = names[i] if names is not None else "%s%d" % (prefix, i+1)
        a = Alternative(aid)
        alts.append(a)

    return alts


def generate_criteria_msjp(number, prefix = 'c', random_direction = False,
                      names = None, random_directions = None):
    """ Generates criteria instances """

    crits = Criteria()
    if random_directions is None:
        for i in range(number):
            cid = names[i] if names is not None else "%s%d" % (prefix, i+1)
            c = Criterion(cid)
            if random_direction is True:
                c.direction = random.choice([-1, 1])
            crits.append(c)
    else :
        for i in range(number):
            cid = names[i] if names is not None else "%s%d" % (prefix, i+1)
            c = Criterion(cid)
            c.direction = random_directions[i]
            crits.append(c)
            
    return crits


def generate_random_criteria_weights_msjp(crits, seed = None, k = 3, fixed_w1 = None):
    """ Constructs weights values for an MR-Sort model """
    
    if seed is not None:
        random.seed(seed)

    if fixed_w1 is None:
        weights = [ random.random() for i in range(len(crits) - 1) ]
    else :
        weights = [ round(random.uniform(0,1-fixed_w1),k) for i in range(len(crits) - 2) ]
        weights = [(fixed_w1 + i) for i in weights]
        weights = weights + [round(fixed_w1,k)]
    weights.sort()

    cvals = CriteriaValues()
    for i, c in enumerate(crits):
        cval = CriterionValue()
        cval.id = c.id
        if i == 0:
            cval.value = round(weights[i], k)
        elif i == len(crits) - 1:
            cval.value = round(1 - weights[i - 1], k)
        else:
            cval.value = round(weights[i] - weights[i - 1], k)
        cvals.append(cval)

    return cvals


def generate_random_performance_table_msjp(alts, crits, seed = None, k = 3,
                                      worst = None, best = None, dupl_crits = None):
    """ Generates random values as performance table 
    for the use of the META and META-SP algorithms """

    if seed is not None:
        random.seed(seed)

    pt = PerformanceTable()
    pt_dupl = PerformanceTable()
    tmp_ids = [i.id for i in dupl_crits]
    for a in alts:
        perfs = {}
        perfs_dupl = {}
        for c in crits:
            if worst is None or best is None:
                #random.seed()
                rdom = round(random.random(), k)
            else:
                rdom = round(random.uniform(worst.performances[c.id],
                                            best.performances[c.id]), k)

            perfs[c.id] = rdom
            perfs_dupl[c.id] = rdom
            if (c.id+"d") in tmp_ids:
                perfs_dupl[(c.id+"d")] = rdom


        ap = AlternativePerformances(a.id, perfs)
        ap_dupl = AlternativePerformances(a.id, perfs_dupl)
        pt.append(ap)
        pt_dupl.append(ap_dupl)

    return pt,pt_dupl


def generate_random_performance_table_msjp_mip(alts, crits, seed = None, k = 2,
                                      worst = None, best = None, dupl_crits = None, cardinality=10):
    """ Generates random values as performance table 
    for the use of the MIP algorithm """
    
    if seed is not None:
        random.seed(seed)

    pt = PerformanceTable()
    pt_dupl = PerformanceTable()
    tmp_ids = [i.id for i in dupl_crits]
    for a in alts:
        perfs = {}
        perfs_dupl = {}
        for c in crits:
            rdom = round(random.choice([f for f in np.arange(0,1.+1./(10**k),1./(cardinality-1))]),k)
            perfs[c.id] = rdom
            perfs_dupl[c.id] = rdom
            if (c.id+"d") in tmp_ids:
                perfs_dupl[(c.id+"d")] = rdom

        ap = AlternativePerformances(a.id, perfs)
        ap_dupl = AlternativePerformances(a.id, perfs_dupl)
        pt.append(ap)
        pt_dupl.append(ap_dupl)

    return pt,pt_dupl



def duplicate_performance_table_msjp(pt, alts, crits, seed = None, k = 3,
                                      worst = None, best = None, dupl_crits = None):
    """ Generate random values as a performance table taking into account 
    the duplication of performances of 
    unknown criteria preference directions """
    
    if seed is not None:
        random.seed(seed)

    pt_dupl = PerformanceTable()
    tmp_ids = [i.id for i in dupl_crits]
    for ap in pt:
        ap_perfs = ap.performances
        for c in crits:
            if (c.id+"d") in tmp_ids:
                ap_perfs[(c.id+"d")] = ap_perfs[c.id]
        
        ap_dupl = AlternativePerformances(ap.altid, ap_perfs)
        pt_dupl.append(ap_dupl)

    return pt_dupl



def generate_categories_msjp(number, prefix = 'cat', names = None):
    """ Generates categories instances """

    cats = Categories()
    for i in range(number):
        cid = names[i] if names is not None else "%s%d" % (prefix, i+1)
        c = Category(cid, rank = number - i)
        cats.append(c)
    return cats


def generate_random_profiles(alts, crits, seed = None, k = 3,
                             worst = None, best = None, prof_threshold = 0.05, fixed_profc1 = None):
    """ Generates random values for profiles taking into account 
    single-peaked criteria in order to build  ground truth models """
    
    if seed is not None:
        random.seed(seed)

    if worst is None:
        worst = generate_worst_ap(crits)
    if best is None:
        best = generate_best_ap(crits)

    crit_random = {}
    n = len(alts)
    pt = PerformanceTable()
    for c in crits:
        rdom = []
        for i in range(n):
            minp = worst.performances[c.id]
            maxp = best.performances[c.id]
            if minp > maxp:
                minp, maxp = maxp, minp
            if fixed_profc1 is not None and c.id == "c1":
                rdom.append(round(random.uniform(0.5-fixed_profc1, 0.5+fixed_profc1), k))
            else:
                if c.direction == 2 or c.direction == -2:
                    b_sp =tuple(sorted([round(random.uniform(max(minp,prof_threshold), min(1-prof_threshold,maxp)), k),round(random.uniform(max(minp,prof_threshold), min(1-prof_threshold,maxp)), k)]))
                    rdom.append(b_sp)
                else:
                    rdom.append(round(random.uniform(max(minp,prof_threshold), min(1-prof_threshold,maxp)), k))
        
        if c.direction == -1:
            rdom.sort(reverse = True)
        else:
            rdom.sort()
        crit_random[c.id] = rdom
    
    for i, a in enumerate(alts):
        perfs = {}
        for c in crits:
            perfs[c.id] = crit_random[c.id][i]
        ap = AlternativePerformances(a, perfs)
        pt.append(ap)
    
    return pt


def generate_random_profiles_msjp(alts, crits, seed = None, k = 3,
                             worst = None, best = None, fct_percentile = [], nb_unk_criteria = 0):
    """ Generates random values as an initialization for 
    the profiles of a new MR-Sort model for the learning process 
    considering monotone and unkwown preference directions """
    
    if seed is not None:
        random.seed(seed)

    if worst is None:
        worst = generate_worst_ap(crits)
    if best is None:
        best = generate_best_ap(crits)

    crit_random = {}
    n = len(alts)
    pt = PerformanceTable()
    random.seed(seed)
    for c in crits:
        rdom = []
        random.seed(seed)
        for i in range(n):
            minp = worst.performances[c.id]
            maxp = best.performances[c.id]
            if minp > maxp:
                minp, maxp = maxp, minp
            if (c.id[-1] != "d") and (int(c.id[1:]) <= nb_unk_criteria):
                rdom.append(round(random.uniform(max(minp,fct_percentile[int(c.id[1:])-1][0]), min(maxp,fct_percentile[int(c.id[1:])-1][1])), k))
            else:
                rdom.append(round(random.uniform(minp,maxp),k))

        if c.direction == -1:
            rdom.sort(reverse = True)
        else:
            rdom.sort()
        crit_random[c.id] = rdom
    
    for i, a in enumerate(alts):
        perfs = {}
        for c in crits:
            perfs[c.id] = crit_random[c.id][i]
        ap = AlternativePerformances(a, perfs)
        pt.append(ap)
    
    return pt


def generate_random_profiles_msjp_sp(alts, crits, seed = None, k = 3,
                             worst = None, best = None, fct_percentile = [], nb_unk_criteria = 0):
    """ Initializes profiles regarding monotone or single-peaked criteria
    of learned models """
    
    if seed is not None:
        random.seed(seed)

    if worst is None:
        worst = generate_worst_ap(crits)
    if best is None:
        best = generate_best_ap(crits)

    crit_random = {}
    n = len(alts)
    pt = PerformanceTable()
    random.seed(seed)
    for c in crits:
        rdom = []
        random.seed(seed)
        for i in range(n):
            minp = worst.performances[c.id]
            maxp = best.performances[c.id]
            if minp > maxp:
                minp, maxp = maxp, minp
            if c.direction == 2 or c.direction == -2:
                b_sp =tuple(sorted([round(random.uniform(minp,maxp), k),round(random.uniform(minp,maxp), k)]))
                rdom.append(b_sp)
            else:
                rdom.append(round(random.uniform(minp, maxp), k))
                
        if c.direction == -1:
            rdom.sort(reverse = True)
        else:
            rdom.sort()
        crit_random[c.id] = rdom
    
    for i, a in enumerate(alts):
        perfs = {}
        for c in crits:
            perfs[c.id] = crit_random[c.id][i]
        ap = AlternativePerformances(a, perfs)
        pt.append(ap)
    return pt


def generate_categories_profiles(cats, prefix='b'):
    cat_ids = cats.get_ordered_categories()
    cps = CategoriesProfiles()
    for i in range(len(cats)-1):
        l = Limits(cat_ids[i], cat_ids[i+1])
        cp = CategoryProfile("%s%d" % (prefix, i + 1), l)
        cps.append(cp)
    return cps



def generate_worst_ap(crits, value = 0):
    """ Establishes a worst alternative profile for criteria """
    
    return AlternativePerformances("worst", {c.id: value
                                             if c.direction == 1 else 1
                                             for c in crits})

def generate_best_ap(crits, value = 1):
    """ Establishes a best alternative profile for criteria """

    return AlternativePerformances("best", {c.id: value
                                             if c.direction == 1 else 0
                                             for c in crits})



def generate_random_mrsort_model(ncrit, ncat, seed = None, k = 3,
                                 worst = None, best = None,
                                 random_direction = False, random_directions = None, fixed_w1 = None, fixed_profc1 = None, min_w = 0.05, prof_threshold = 0.05):
    """ Generates uniformly random values of weights, profiles and lambda 
    and returns such parameters as an instance of an MR-Sort model """
    
    # Contributes to randomize the draw
    if seed is not None:
        random.seed(int(seed))
    random.seed()
    
    c = generate_criteria_msjp(ncrit, random_direction = random_direction, random_directions = random_directions)
    random.seed()
    if worst is None:
        worst = generate_worst_ap(c)
    if best is None:
        best = generate_best_ap(c)

    # Construction of weights values
    no_min_w =  True
    while no_min_w:
        random.seed()
        cv = generate_random_criteria_weights_msjp(c, None, k, fixed_w1)
        if [i for i in list(cv.values()) if i.value < min_w] == []:
            no_min_w = False
        
    random.seed()
    cat = generate_categories_msjp(ncat)
    random.seed()
    cps = generate_categories_profiles(cat)
    random.seed()
    b = cps.get_ordered_profiles()
    random.seed()
    bpt = generate_random_profiles(b, c, None, k, worst, best, prof_threshold = prof_threshold, fixed_profc1=fixed_profc1)
    random.seed()
    lbda = round(random.uniform(max(0.5,cv.max())+0.01, 1), k)

    return MRSort(c, cv, bpt, lbda, cps)

