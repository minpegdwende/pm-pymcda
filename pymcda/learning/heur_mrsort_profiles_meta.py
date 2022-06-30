from __future__ import division
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
from itertools import product
import random
import numpy as np



class MetaMRSortProfiles5():

    def __init__(self, model, pt_sorted, aa_ori):
        """ Initializes sub-parameters for the learning of profiles values """
        
        self.na = len(aa_ori)
        self.na_good = len([aa_ori(ff) for ff in aa_ori.keys() if aa_ori(ff)=='recommend'])
        self.nc = len(model.criteria)
        self.it_meta = np.inf
        self.model = model
        self.nprofiles = len(model.profiles)
        self.pt_sorted = pt_sorted
        self.aa_ori = aa_ori
        self.cat = self.categories_rank()
        self.cat_ranked = self.model.categories
        self.aa_by_cat = self.sort_alternative_by_category(aa_ori)
        self.b0 = pt_sorted.pt.get_worst(model.criteria)
        self.bp = pt_sorted.pt.get_best(model.criteria)
        self.rebuild_tables()

    def categories_rank(self):
        return { cat: i + 1 for i, cat in enumerate(self.model.categories) }

    def sort_alternative_by_category(self, aa):
        aa_by_cat = {}
        for a in aa:
            aid = a.id
            cat = self.cat[a.category_id]
            if cat in aa_by_cat:
                aa_by_cat[cat].append(aid)
            else:
                aa_by_cat[cat] = [ aid ]
        return aa_by_cat

    def compute_above_histogram(self, cid, perf_profile, perf_above,
                                cat_b, cat_a, ct):
        """ Compute possible moves above a given profile value """
        
        w = self.model.cv[cid].value
        lbda = self.model.lbda
        direction = self.model.criteria[cid].direction
        delta = 0.00001 * direction

        h_above = {}
        num = total = 0
        alts, perfs = self.pt_sorted.get_middle(cid,
                                                perf_profile, perf_above,
                                                True, True)

        for i, a in enumerate(alts):
            if (perfs[i] + delta) * direction > perf_above * direction:
                continue
            conc = ct[a]
            aa_ori = self.aa_ori._d[a].category_id
            aa = self.aa._d[a].category_id
            diff = conc - w
            if aa_ori == cat_a:
                if aa == cat_a and diff < lbda:
                    # --
                    total += 5
                elif aa == cat_b:
                    # -
                    total += 1
            elif aa_ori == cat_b and aa == cat_a:
                if diff >= lbda:
                    # +
                    num += 0.5
                    total += 1
                    h_above[perfs[i] + delta] = num / total
                else:
                    # ++
                    num += 2
                    total += 1
                    h_above[perfs[i] + delta] = num / total
            elif aa_ori != aa and \
                 self.cat[aa] < self.cat[cat_a] and \
                 self.cat[aa_ori] < self.cat[cat_a]:
                num += 0.1
                total += 1
                h_above[perfs[i] + delta] = num / total

        return h_above

    def compute_below_histogram(self, cid, perf_profile, perf_below, cat_b, cat_a, ct):
        """ Compute possible moves below a given profile value """

        w = self.model.cv[cid].value
        lbda = self.model.lbda

        h_below = {}
        num = total = 0
        alts, perfs = self.pt_sorted.get_middle(cid,perf_profile, perf_below,True, True)

        for i, a in enumerate(alts):
            conc = ct[a]
            aa_ori = self.aa_ori._d[a].category_id
            aa = self.aa._d[a].category_id
            diff = conc + w
            if aa_ori == cat_a and aa == cat_b:
                if diff >= lbda:
                    # ++
                    num += 3
                    total += 1
                    h_below[perfs[i]] = num / total
                else:
                    # +
                    num += 1.0
                    total += 1
                    h_below[perfs[i]] = num / total
            elif aa_ori == cat_b:
                if aa == cat_b and diff >= lbda:
                    # --
                    total += 5
                elif aa == cat_a:
                    # -
                    total += 1
            elif aa_ori != aa and \
                 self.cat[aa] > self.cat[cat_b] and \
                 self.cat[aa_ori] > self.cat[cat_b]:
                num += 0.1
                total += 1
                h_below[perfs[i]] = num / total

        return h_below

    def histogram_choose(self, h, current):
        """ Chooses the approriate move (key,value)
        in function of the desirability score and the distance 
        between the current and the next move """
        
        key = random.choice(list(h.keys()))
        val = h[key]
        diff = abs(current - key)
        for k, v in h.items():
            if v >= val:
                tmp = abs(current - k)
                if tmp > diff:
                    key = k
                    val = v
                    diff = tmp
        return key

    def get_alternative_assignment(self, aid):
        ap = self.pt_sorted.pt[aid]
        return self.model.get_assignment(ap).category_id

    def build_assignments_table(self):
        self.good = 0
        self.good_good = 0
        self.aa = self.model.get_assignments(self.pt_sorted.pt)
        for a in self.aa:
            cat1 = a.category_id
            cat2 = self.aa_ori[a.id].category_id
            if cat1 == cat2:
                self.good += 1
                if cat2 == 'recommend':
                    self.good_good += 1

    def build_concordance_table(self):
        self.ct = { bp.id: dict() for bp in self.model.bpt }
        for aid, bp in product(self.aa_ori.keys(), self.model.bpt):
            ap = self.pt_sorted[aid]
            conc = self.model.concordance(ap, bp)
            self.ct[bp.id][aid] = conc

    def build_veto_table(self):
        if self.model.vpt is None:
            self.vt = None
            return

        self.vt = { bp.id: dict() for bp in self.model.vpt }
        for aid, bp in product(self.aa_ori.keys(), self.model.vpt):
            ap = self.pt_sorted[aid]
            conc = self.model.veto_concordance(ap, bp)
            self.vt[bp.id][aid] = conc

    def rebuild_tables(self):
        self.build_concordance_table()
        self.build_veto_table()
        self.build_assignments_table()

    def update_tables(self, profile, cid, old, new):
        direction = self.model.criteria[cid].direction
        if old > new:
            w = self.model.cv[cid].value * direction
        else:
            w = -self.model.cv[cid].value * direction

        alts, perfs = self.pt_sorted.get_middle(cid, old, new, True, True)

        for a in alts:
            self.ct[profile][a] += w
            old_cat = self.aa[a].category_id
            new_cat = self.get_alternative_assignment(a)
            ori_cat = self.aa_ori[a].category_id

            if old_cat == new_cat:
                continue
            elif old_cat == ori_cat:
                self.good -= 1
                if ori_cat == 'recommend':
                    self.good_good -= 1
            elif new_cat == ori_cat:
                self.good += 1
                if ori_cat == 'recommend':
                    self.good_good += 1

            self.aa[a].category_id = new_cat

    def optimize_profile(self, profile, below, above, cat_b, cat_a):
        """ Optimizes profiles values per criteria (monotone), per model,
        computes next moves of profiles 
        if it is relevant and updates learning parameters"""
        
        cids = list(self.model.criteria.keys())
        random.shuffle(cids)

        for cid in cids:
            ct = self.ct[profile.id]

            perf_profile = profile.performances[cid]
            perf_above = above.performances[cid]
            perf_below = below.performances[cid]

            h_below = self.compute_below_histogram(cid, perf_profile,
                                                   perf_below, cat_b,
                                                   cat_a, ct)
            h_above = self.compute_above_histogram(cid, perf_profile,
                                                   perf_above, cat_b,
                                                   cat_a, ct)
            h = h_below
            h.update(h_above)

            if not h:
                continue

            i = self.histogram_choose(h, perf_profile)

            r = random.uniform(0, 1)

            if r <= h[i]:
                profile.performances[cid] = i
                self.update_tables(profile.id, cid, perf_profile, i)

    def get_profile_limits(self, i):
        profiles = self.model.profiles
        above = self.model.get_profile_upper_limit(profiles[i])
        below = self.model.get_profile_lower_limit(profiles[i])

        if above is None:
            above = self.bp

        if below is None:
            below = self.b0

        return below, above

    def optimize(self, it_meta = np.inf):
        """ Optimizes profiles values per categories """
        
        self.it_meta = it_meta
        profiles = self.model.profiles
        for i, profile in enumerate(profiles):
            pperfs = self.model.bpt[profile]
            below, above = self.get_profile_limits(i)
            cat_b, cat_a = self.cat_ranked[i], self.cat_ranked[i+1]
            self.optimize_profile(pperfs, below, above, cat_b, cat_a)

        return (self.good / self.na,1) if (self.na_good==0) else (self.good / self.na, self.good_good / self.na_good)

