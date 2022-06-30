from __future__ import division
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
from itertools import product
import itertools
import random
import numpy as np

LIMIT_EVAL_SCALE_INF = 0
LIMIT_EVAL_SCALE_SUP = 1


class MetaMRSortProfiles5():

    def __init__(self, model, pt_sorted, aa_ori, strat_heur=[]):
        """ Initializes sub-parameters for the learning of profiles values 
        considering eventually single-peaked/single valley criteria """
        
        dec = 5
        self.na = len(aa_ori)
        self.na_good = len([aa_ori(ff) for ff in aa_ori.keys() if aa_ori(ff)=='recommend'])
        self.nc = len(model.criteria)
        self.it_meta = np.inf
        self.model = model
        self.strat_heur = strat_heur
        self.nprofiles = len(model.profiles)
        self.pt_sorted = pt_sorted
        self.aa_ori = aa_ori
        self.cat = self.categories_rank()
        self.cat_ranked = self.model.categories
        self.aa_by_cat = self.sort_alternative_by_category(aa_ori)
        self.b0 = pt_sorted.pt.get_worst(model.criteria)
        self.bp = pt_sorted.pt.get_best(model.criteria)
        self.rebuild_tables()

        # Initialization of profiles values of each model
        self.gt = dict()
        self.perfs = dict()
        for c in model.criteria:
            if abs(c.direction) == 2:
                self.gt[c.id] = []
                self.perfs[c.id] = set(pt_sorted.sorted_values[c.id])
                #import pdb; pdb.set_trace()
                for ev in self.perfs[c.id]:
                    self.gt[c.id] += [(ev,[b for a,b in pt_sorted.sorted_pt[c.id] if a==ev and self.aa_ori._d[b].category_id=="cat1"],[b for a,b in pt_sorted.sorted_pt[c.id] if a==ev and self.aa_ori._d[b].category_id=="cat2"])]
        
        # Recalibrating profiles values for single-peaked criteria 
        if self.strat_heur[1] == 1:
            for c in model.criteria:
                avg_b_mid = []
                avg_b_inf = []
                avg_b_sup = []
                if abs(c.direction) == 2:
                    if c.direction == 2:
                        for  a,b in pt_sorted.sorted_pt[c.id]:
                            if self.aa_ori._d[b].category_id == "cat2":
                                avg_b_mid += [a]
                        avg_b_mid=np.mean(avg_b_mid) if not avg_b_mid==[] else round(random.random(),dec)
                        for  a,b in pt_sorted.sorted_pt[c.id]:
                            if self.aa_ori._d[b].category_id == "cat1" and a<avg_b_mid:
                                avg_b_inf += [a]
                            if self.aa_ori._d[b].category_id == "cat1" and a>avg_b_mid:
                                avg_b_sup += [a]
                    if c.direction == -2:
                        for  a,b in pt_sorted.sorted_pt[c.id]:
                            if self.aa_ori._d[b].category_id == "cat1":
                                avg_b_mid += [a]
                        avg_b_mid=np.mean(avg_b_mid) if not avg_b_mid==[] else round(random.random(),dec)
                        for  a,b in pt_sorted.sorted_pt[c.id]:
                            if self.aa_ori._d[b].category_id == "cat2" and a<avg_b_mid:
                                avg_b_inf += [a]
                            if self.aa_ori._d[b].category_id == "cat2" and a>avg_b_mid:
                                avg_b_sup += [a]
    
                    avg_b_inf = np.mean(avg_b_inf) if not avg_b_inf==[] else round(random.random(),dec)
                    avg_b_sup = np.mean(avg_b_sup) if not avg_b_sup==[] else round(random.random(),dec)
                    self.model.bpt["b1"].performances[c.id] =  sorted([round(avg_b_inf,dec),round(avg_b_sup,dec)])
                elif abs(c.direction) == 1:
                    for  a,b in pt_sorted.sorted_pt[c.id]:
                        avg_b_mid += [a]
                    avg_b_mid=np.mean(avg_b_mid) if not avg_b_mid==[] else round(random.random(),dec)
                    self.model.bpt["b1"].performances[c.id] = round(avg_b_mid,dec)



    def categories_rank(self):
        return { cat: i + 1 for i, cat in enumerate(self.model.categories) }


    def sort_alternative_by_category(self, aa):
        """ Returns a dict where for each entry (category_id) 
        the value is the list of alternatives assigned to this category """
        
        aa_by_cat = {}
        for a in aa:
            aid = a.id
            cat = self.cat[a.category_id]
            if cat in aa_by_cat:
                aa_by_cat[cat].append(aid)
            else:
                aa_by_cat[cat] = [ aid ]
        return aa_by_cat



    def compute_histogram_sp(self, cid, perf_profile, perf_limits, ct):
        """ Returns a histogram with for each deplacement, 
        the associated distribution probability 
        regarding the considered profile, considering SP criteria """
        
        w = self.model.cv[cid].value
        lbda = self.model.lbda
        direction = self.model.criteria[cid].direction
        h = {}
        
        num = total = 0
        self.curr = dict()
        for c in self.model.criteria:
            if abs(c.direction) == 2:
                self.curr[c.id] = []
                for ev in self.perfs[c.id]:
                    self.curr[c.id] += [(ev,[b for a,b in self.pt_sorted.sorted_pt[c.id] if a==ev and self.aa._d[b].category_id=="cat1"],[b for a,b in self.pt_sorted.sorted_pt[c.id] if a==ev and self.aa._d[b].category_id=="cat2"])]        
        
        for depl in itertools.combinations(self.perfs[cid],2):
            num = total = 0
            depl=sorted(depl)
            for ((ev,l1,l2),(ev2,cu1,cu2)) in zip(self.gt[cid],self.curr[cid]):
                if direction == 2:
                    if ev < perf_profile[0] and (ev < depl[0] or ev > depl[1]): #based only on the criterion cid, cat_c=cat1 cat_d=cat1,  => but true cat_c is cu1,cu2
                        if (ev < depl[0] and abs(ev-perf_profile[0]) > abs(ev-depl[0])) or (abs(ev-perf_profile[0]) > abs(ev-depl[1])) : # deplacement is the closest to ev
                            num +=   1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #R
                        elif (abs(ev-perf_profile[0]) < abs(ev-depl[0])) or (abs(ev-perf_profile[0]) < abs(ev-depl[1])): #deplacement is the fartest from ev
                            num +=   1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #R
                            total += 1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #W
                        #print("W and R")
                    elif (ev < perf_profile[0] or ev > perf_profile[1]) and ev >= depl[0] and ev <= depl[1]: #cat_c=cat1 cat_d=cat2, => increases the coalition by w
                        num += 2.0*len([e for e in set.intersection(set(l2),set(cu1)) if ct[e]+w>=lbda]) #V
                        num +=     len([e for e in set.intersection(set(l2),set(cu1)) if ct[e]+w<lbda]) #W
                        total += 5.0*len([e for e in set.intersection(set(l1),set(cu1)) if ct[e]+w>=lbda]) #Q
                        total +=     len([e for e in set.intersection(set(l1),set(cu2))]) #R
                        total +=     len([e for e in set.intersection(set(l2),set(cu1)) if ct[e]+w>=lbda]) #V
                        total +=     len([e for e in set.intersection(set(l2),set(cu1)) if ct[e]+w<lbda]) #W
                        #print("Q and V")
                    elif ev >= perf_profile[0] and ev <= perf_profile[1] and (ev < depl[0] or ev > depl[1]): #cat_c=cat2 cat_d=cat1, => decreases the coalition by w
                        num += 2.0*len([e for e in set.intersection(set(l1),set(cu2)) if ct[e]-w<lbda]) #V
                        num +=     len([e for e in set.intersection(set(l1),set(cu2)) if ct[e]-w>=lbda]) #W
                        total += 5.0*len([e for e in set.intersection(set(l2),set(cu2)) if ct[e]-w<lbda]) #Q
                        total +=     len([e for e in set.intersection(set(l2),set(cu1))]) #R
                        total +=     len([e for e in set.intersection(set(l1),set(cu2)) if ct[e]-w<lbda]) #V
                        total +=     len([e for e in set.intersection(set(l1),set(cu2)) if ct[e]-w>=lbda]) #W
                        #print("Q and V")
                    elif ev >= perf_profile[0] and ev <= perf_profile[1] and ev >= depl[0] and ev <= depl[1]: #cat_c=cat2 cat_d=cat2, => Wij and Rij
                        if min(abs(ev-perf_profile[0]),abs(ev-perf_profile[1])) > min(abs(ev-depl[0]),abs(ev-depl[1])) : # deplacement is the closest to ev
                            num +=   1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #R
                            total += 1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #W
                        elif min(abs(ev-perf_profile[0]),abs(ev-perf_profile[1])) < min(abs(ev-depl[0]),abs(ev-depl[1])): #le deplacement is the fartest from ev
                            num +=   1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #R
                        #print("W and R")
                    elif ev > perf_profile[1] and (ev < depl[0] or ev > depl[1]):  #cat_c=cat1 cat_d=cat1, => 
                        if (abs(ev-perf_profile[1]) > abs(ev-depl[0])) or (abs(ev-perf_profile[1]) > abs(ev-depl[1])): # deplacement is the closest to ev
                            num +=   1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #R
                        elif (abs(ev-perf_profile[1]) < abs(ev-depl[0])) or (abs(ev-perf_profile[1]) < abs(ev-depl[1])): # deplacement is the fartest from ev
                            num +=   1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #R
                            total += 1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #W
                        #print("W and R")
                if direction == -2:
                    if ev <= perf_profile[0] and (ev <= depl[0] or ev >= depl[1]): #cat_c=cat2 cat_d=cat2, 
                        if (abs(ev-perf_profile[0]) > abs(ev-depl[0])) or (abs(ev-perf_profile[0]) > abs(ev-depl[1])): # deplacement is the closest to ev
                            num +=   1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #R
                            total += 1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #W
                        elif (abs(ev-perf_profile[0]) < abs(ev-depl[0])) or (abs(ev-perf_profile[0]) < abs(ev-depl[1])): # deplacement is the fartest from ev
                            num +=   1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #R
                        #print("W and R")
                    elif (ev <= perf_profile[0] or ev >= perf_profile[1]) and ev > depl[0] and ev < depl[1]: #cat_c=cat2 cat_d=cat1, =>  decreases the coalition by w
                        num += 2.0*len([e for e in set.intersection(set(l1),set(cu2)) if ct[e]-w<lbda]) #V
                        num +=     len([e for e in set.intersection(set(l1),set(cu2)) if ct[e]-w>=lbda]) #W
                        total += 5.0*len([e for e in set.intersection(set(l2),set(cu2)) if ct[e]-w<lbda]) #Q
                        total +=     len([e for e in set.intersection(set(l2),set(cu1))]) #R
                        total +=     len([e for e in set.intersection(set(l1),set(cu2)) if ct[e]-w<lbda]) #V
                        total +=     len([e for e in set.intersection(set(l1),set(cu2)) if ct[e]-w>=lbda]) #W
                        #print("QVWR")
                    elif ev > perf_profile[0] and ev < perf_profile[1] and (ev <= depl[0] or ev >= depl[1]): #cat_c=cat1 cat_d=cat2, => increases the coalition by w
                        num += 2.0*len([e for e in set.intersection(set(l2),set(cu1)) if ct[e]+w>=lbda]) #V
                        num +=     len([e for e in set.intersection(set(l2),set(cu1)) if ct[e]+w<lbda]) #W
                        total += 5.0*len([e for e in set.intersection(set(l1),set(cu1)) if ct[e]+w>=lbda]) #Q
                        total +=     len([e for e in set.intersection(set(l1),set(cu2))]) #R
                        total +=     len([e for e in set.intersection(set(l2),set(cu1)) if ct[e]+w>=lbda]) #V
                        total +=     len([e for e in set.intersection(set(l2),set(cu1)) if ct[e]+w<lbda]) #W
                        #print("QVWR")
                    elif ev > perf_profile[0] and ev < perf_profile[1] and ev > depl[0] and ev < depl[1]: #cat_c=cat1 cat_d=cat1, => Qij and Vij
                        if min(abs(ev-perf_profile[0]),abs(ev-perf_profile[1])) > min(abs(ev-depl[0]),abs(ev-depl[1])) : # deplacement is the closest to ev
                            num +=   1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #R
                        elif min(abs(ev-perf_profile[0]),abs(ev-perf_profile[1])) < min(abs(ev-depl[0]),abs(ev-depl[1])): # deplacement is the fartest from ev
                            num +=   1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #R
                            total += 1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #W
                        #print("W and R")
                    elif ev >= perf_profile[1] and (ev <= depl[0] or ev >= depl[1]): #cat_c=cat2 cat_d=cat2, 
                        if (abs(ev-perf_profile[1]) > abs(ev-depl[0])) or (abs(ev-perf_profile[1]) > abs(ev-depl[1])): # deplacement is the closest to ev
                            num +=   1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #R
                            total += 1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #W
                        elif (abs(ev-perf_profile[1]) < abs(ev-depl[0])) or (abs(ev-perf_profile[1]) < abs(ev-depl[1])): # deplacement is the fartest from ev
                            num +=   1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l2),set(cu1))]) #W
                            total += 1.0*len([e for e in set.intersection(set(l1),set(cu2))]) #R
                        #print("W and R")
            if total !=0 :
                h[tuple(depl)] = num / total
        return h

    def compute_above_histogram(self, cid, perf_profile, perf_limits, cat_b, cat_a, ct, ind_prof= None):
        """ Compute possible moves above a given profile value """
        
        w = self.model.cv[cid].value
        lbda = self.model.lbda
        direction = self.model.criteria[cid].direction
        if abs(direction) == 2:
            if direction == 2 and ind_prof == 0:
                delta = 0.001 * (1)
                perf_above = perf_profile[1]
            elif direction == 2 and ind_prof == 1:
                delta = 0.001 * (-1)
                perf_above = perf_profile[0]
            elif direction == -2 and ind_prof == 0:
                delta = 0.001 * (-1)
                perf_above = perf_limits[1]  # choosing best_fictive and worse_fictive alternative is based on the sign of the direction.
            elif direction == -2 and ind_prof == 1:
                delta = 0.001 * (1)
                perf_above = perf_limits[0]
            perf_profile = perf_profile[ind_prof]
        elif abs(direction)==1:
            delta = 0.001 * direction
            perf_above = perf_limits
        
        h_above = {}
        num = total = 0
        alts, perfs = self.pt_sorted.get_middle(cid, perf_profile, perf_above, True, True)
        
        for i, a in enumerate(alts):
            if (perfs[i] + delta) > LIMIT_EVAL_SCALE_SUP or (perfs[i] + delta) < LIMIT_EVAL_SCALE_INF:
                continue

            conc = ct[a]
            aa_ori = self.aa_ori._d[a].category_id
            aa = self.aa._d[a].category_id
            diff = conc - w
            if aa_ori == cat_a:
                if aa == cat_a and diff < lbda:
                    # Qij
                    # --
                    total += 5
                elif aa == cat_b:
                    # Rij..
                    # -
                    total += 1
            elif aa_ori == cat_b and aa == cat_a:
                if diff >= lbda:
                    # Wij...
                    # +
                    num += 0.5
                    total += 1
                    h_above[perfs[i] + delta] = num / total
                else:
                    # Vij 
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
            #print(num,total,h_above)
        return h_above


    def compute_below_histogram(self, cid, perf_profile, perf_limits, cat_b, cat_a, ct, ind_prof= None):
        """ Compute possible moves below a given profile value """
        
        w = self.model.cv[cid].value
        lbda = self.model.lbda
        direction = self.model.criteria[cid].direction
        if direction == 2:
            perf_below = perf_limits[ind_prof]
            perf_profile = perf_profile[ind_prof]
        if direction == -2:
            if ind_prof == 0:
                perf_below = perf_profile[1]
            if ind_prof == 1:
                perf_below = perf_profile[0]
            perf_profile = perf_profile[ind_prof]
        if abs(direction) == 1:
            perf_below = perf_limits
        
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
                    # Wij 
                    # +
                    num += 1.0
                    total += 1
                    h_below[perfs[i]] = num / total
            elif aa_ori == cat_b:
                if aa == cat_b and diff >= lbda:
                    # Qij
                    # --
                    total += 5
                elif aa == cat_a:
                    # Represents Rij 
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
        if self.strat_heur[2] == 1:
            for k, v in h.items():
                if v >= val:
                    tmp = abs(current - k)
                    if random.random() < 5:
                        if tmp > diff:
                            key = k
                            val = v
                            diff = tmp
                    else:
                        key = k
                        val = v
                        diff = tmp
        else:
            for k, v in h.items():
                if v >= val:
                    tmp = abs(current - k)
                    if tmp > diff:
                        key = k
                        val = v
                        diff = tmp
        
        return key


    def histogram_choose_sp(self, h, current):
        """ Chooses the approriate move (key,value)
        in function of the desirability score and the distance 
        between the current and the next move considering
        single-peaked/single-valley criteria """
        
        key = random.choice(list(h.keys()))
        val = h[key]
        diff = abs(current[0] - key[0])+abs(current[1] - key[1])
        if self.strat_heur[2] == 1:
            rand_it_meta = self.it_meta * random.random()
            for k, v in h.items():
                if v >= val:
                    tmp = abs(current[0] - k[0])+abs(current[1] - k[1])
                    if rand_it_meta < 5:
                        if tmp > diff:
                            key = k
                            val = v
                            diff = tmp
                    else:
                        key = k
                        val = v
                        diff = tmp
        else:
            for k, v in h.items():
                if v >= val:
                    tmp = abs(current[0] - k[0])+abs(current[1] - k[1])
                    if tmp > diff:
                        key = k
                        val = v
                        diff = tmp
        
        return key



    def get_alternative_assignment(self, aid):
        ap = self.pt_sorted.pt[aid]
        return self.model.get_assignment_sp(ap).category_id


    def build_assignments_table(self):
        """  Aims at recalculating CA score with the updated profiles """
        
        self.good = 0
        self.good_good = 0
        self.aa = self.model.get_assignments_sp(self.pt_sorted.pt)
        #self.aa = self.model.get_assignments(self.pt_sorted.pt)
        #import pdb; pdb.set_trace()
        for a in self.aa:
            cat1 = a.category_id
            cat2 = self.aa_ori[a.id].category_id
            if cat1 == cat2:
                self.good += 1
                if cat2 == 'recommend':
                    self.good_good += 1



    def build_concordance_table(self):
        """ Compute per profiles, the sum of criteria 
        for each alternative that outranks the given profile """
        self.ct = { bp.id: dict() for bp in self.model.bpt }
        #import pdb; pdb.set_trace()
        for aid, bp in product(self.aa_ori.keys(), self.model.bpt):
            ap = self.pt_sorted[aid]
            conc = self.model.concordance_sp(ap, bp)
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

    def update_tables(self, profile, cid, old, new, ind_prof = None):
        """ It updates the ct (concordance table) given that profiles moved """
        
        direction = self.model.criteria[cid].direction
        if False and abs(direction)==1:
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
            
        if True or abs(direction)==2:
            self.rebuild_tables()



    def optimize_profile(self, profile, below, above, cat_b, cat_a, it_meta):
        """ Optimizes profiles values per criteria (SP and SV criteria), 
        per model, computes next moves of profiles 
        if it is relevant and updates learning parameters"""
        
        cids = list(self.model.criteria.keys())
        random.shuffle(cids)
        for cid in cids:
            ct = self.ct[profile.id]
            perf_profile = profile.performances[cid]
            perf_above = above.performances[cid]
            perf_below = below.performances[cid]
            if abs(self.model.criteria[cid].direction)== 1:
                h_below = self.compute_below_histogram(cid, perf_profile, perf_below, cat_b, cat_a, ct)
                h_above = self.compute_above_histogram(cid, perf_profile, perf_above, cat_b, cat_a, ct)
                h = h_below
                h.update(h_above)
    
                if not h:
                    continue
    
                i = self.histogram_choose(h, perf_profile)
                r = random.uniform(0, 1)
    
                if r <= h[i]:
                    profile.performances[cid] = i
                    self.update_tables(profile.id, cid, perf_profile, i)
            
            
            elif abs(self.model.criteria[cid].direction) == 2:
                #Case of SP,SV criteria implying successive update of the profiles moves
                if self.strat_heur[0]==1:
                    draw = [True,False]
                    for first in [draw, not draw]:
                        if first:
                            h_below1 = self.compute_below_histogram(cid, perf_profile, (perf_below,perf_above), cat_b, cat_a, ct, 0)
                            h_above1 = self.compute_above_histogram(cid, perf_profile, (perf_below,perf_above), cat_b, cat_a, ct, 0)
                            h1 = h_below1
                            h1.update(h_above1)
                            if not h1:
                                continue
                            i1 = self.histogram_choose(h1, perf_profile[0])
                            r = random.uniform(0, 1)
                            if r <= h1[i1] and i1 <= profile.performances[cid][1]:
                                profile.performances[cid] = (i1,profile.performances[cid][1])
                                self.update_tables(profile.id, cid, perf_profile, i1)
                        else:
                            h_below2 = self.compute_below_histogram(cid, perf_profile, (perf_below,perf_above), cat_b, cat_a, ct, 1)
                            h_above2 = self.compute_above_histogram(cid, perf_profile, (perf_below,perf_above), cat_b, cat_a, ct, 1)
                            h2 = h_below2
                            h2.update(h_above2)
                            if not h2:
                                continue
                            i2 = self.histogram_choose(h2, perf_profile[1])
                            r = random.uniform(0, 1)
                            if r <= h2[i2] and i2 >= profile.performances[cid][0]:
                                profile.performances[cid] = (profile.performances[cid][0],i2)
                                self.update_tables(profile.id, cid, perf_profile, i2)
                elif self.strat_heur[0] == 2 :
                    h = self.compute_histogram_sp(cid, perf_profile, (perf_below,perf_above), ct)
                    if not h:
                        continue
                    i = self.histogram_choose_sp(h, perf_profile)
                    r = random.uniform(0, 1)
                    if r <= h[i]:
                        profile.performances[cid] = i
                        self.update_tables(profile.id, cid, perf_profile, i)


    def get_profile_limits(self, i):
        """ Returns the surronding  profile (upper and lower) 
        of the given profile, possibly returns the worst/best 
        possible fictive alternative performances """
        
        profiles = self.model.profiles
        above = self.model.get_profile_upper_limit(profiles[i])
        below = self.model.get_profile_lower_limit(profiles[i])
        
        if above is None:
            above = self.bp
        if below is None:
            below = self.b0

        return below, above


    def optimize(self, it_meta = np.inf):
        """ Optimizes profiles values per categories 
        (fixed to 2 categories) """
        
        self.it_meta = it_meta
        profiles = self.model.profiles
        for i, profile in enumerate(profiles):
            pperfs = self.model.bpt[profile]
            below, above = self.get_profile_limits(i)
            cat_b, cat_a = self.cat_ranked[i], self.cat_ranked[i+1]
            self.optimize_profile(pperfs, below, above, cat_b, cat_a, it_meta)

        return (self.good / self.na,1) if (self.na_good==0) else (self.good / self.na, self.good_good / self.na_good)


