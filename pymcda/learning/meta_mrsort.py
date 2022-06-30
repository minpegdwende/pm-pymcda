from __future__ import division
import errno
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
import random
import math
import copy
from pymcda.electre_tri import MRSort
from multiprocessing import Process, Queue
from pymcda.learning.lp_mrsort_weights_meta import LpMRSortWeightsPositive
from pymcda.learning.heur_mrsort_profiles_meta import MetaMRSortProfiles5
from pymcda.generate import generate_random_profiles_msjp
from pymcda.generate import generate_categories_profiles


# Global variable which activate/desactivate 
# the multiprocessing options for debugging purposes
HEUR_MODE_DEBUG = False


class MetaMRSort():

    def __init__(self, nmodels, criteria, unk_pref_dir_criteria, categories, pt_sorted, aa_ori,
                 lp_weights = LpMRSortWeightsPositive, heur_profiles = MetaMRSortProfiles5,
                 seed = 0, gamma = 0.5, version_meta = 4, pretreatment_crit = None,duplication = True):
        """ Create an instance of META algorithm 
        for learning MR-Sort parameters considering a population of models """
        
        self.nmodels = nmodels
        self.criteria = copy.deepcopy(criteria)
        self.nb_unk_criteria = len(unk_pref_dir_criteria)
        self.categories = categories
        self.pt_sorted = pt_sorted
        self.aa_ori = aa_ori
        self.lp_weights = lp_weights
        self.heur_profiles = heur_profiles
        self.version_meta = version_meta
        self.gamma = gamma
        self.pretreatment_crit = pretreatment_crit
        self.duplication = duplication
        self.metas = list()
        for i in range(self.nmodels):
            meta = self.init_one_meta(i + seed)
            meta.num = i
            meta.cah = 0
            self.metas.append(meta)


    def init_one_meta(self, seed):
        """ Initializes one container 
        (containing a new individual i.e a MR-Sort model initialized) """
        
        cps = generate_categories_profiles(self.categories)
        model = MRSort(self.criteria, None, None, None, cps)
        model.id = 'model_%d' % seed
        meta = MetaMRSortOne(model, self.pt_sorted, self.aa_ori, self.lp_weights, self.heur_profiles, gamma = self.gamma,
                             version_meta = self.version_meta, pretreatment_crit = self.pretreatment_crit,
                             duplication = self.duplication)
        random.seed(seed)
        meta.random_state = random.getstate()
        meta.auc = meta.model.auc(self.aa_ori, self.pt_sorted.pt)
        return meta

    def sort_models(self, fct_ca=0):
        """ Adhoc sorting function that enables to sort the
        population of models according to several features.
        By default, the sorting is according to 
        the classification accuracy of models in the population"""        
        
        if fct_ca == 1:
            metas_sorted = sorted(self.metas, key = lambda k: k.ca_good,
                              reverse = True)
        elif fct_ca == 2:
            metas_sorted = sorted(self.metas, key = lambda k: k.ca_good + k.ca,
                              reverse = True)
        elif fct_ca ==3:
            metas_sorted = sorted(self.metas, key = lambda k: 1000*k.ca_good + k.ca,
                              reverse = True)
        else:
            metas_sorted = sorted(self.metas, key = lambda k: k.ca,
                              reverse = True)
        return metas_sorted



    def reinit_worst_models(self, cloning = False):
        """ Updates the population by removing/re-initializing 
        worst models """
        
        nmeta_to_reinit = int(math.ceil(self.nmodels / 2))
        metas_sorted = self.sort_models()
        for meta in metas_sorted[nmeta_to_reinit:]:
            meta.init_profiles()
            

    def queue_get_retry(self, queue):
        """ Ahdoc function that executes pop out operations on queues """
        
        while True:
            try:
                return queue.get()
            except IOError as e:
                if e.errno == errno.EINTR:
                    continue
                else:
                    raise
                
                
    def _process_optimize(self, meta, nmeta):
        """ Packaging the results of optimizations phases 
        for a given model contained in meta"""
        
        # If HEUR_MODE_DEBUG is True then it performs 
        # a sequential treatment of individuals in the population
        if HEUR_MODE_DEBUG:
            random.setstate(meta.random_state)
            ca,ca_good = meta.optimize(nmeta)
            return [ca, ca_good, meta.model.bpt, meta.model.cv, meta.model.lbda,
                            meta.model.vpt, meta.model.veto_weights,
                            meta.model.veto_lbda, random.getstate()]
        else:
            random.setstate(meta.random_state)
            ca,ca_good = meta.optimize(nmeta)
            meta.queue.put([ca, ca_good, meta.model.bpt, meta.model.cv, meta.model.lbda,
                            meta.model.vpt, meta.model.veto_weights,
                            meta.model.veto_lbda, random.getstate()])



    def optimize(self, nmeta, fct_ca, it_meta=1):
        """ Performs optimization one iteration of the outer loop of
        Sobrie heuristic"""
        
        if it_meta > 0:
            self.reinit_worst_models()

        if HEUR_MODE_DEBUG:
            res_metas = list()
            for meta in self.metas:
                meta.it_meta = it_meta
                res_metas.append(self._process_optimize(meta, nmeta))
            for meta,res_meta in zip(self.metas, res_metas):
                output = res_meta
                meta.ca = output[0]
                meta.ca_good = output[1]
                meta.model.bpt = output[2]
                meta.model.cv = output[3]
                meta.model.lbda = output[4]
                meta.model.vpt = output[5]
                meta.model.veto_weights = output[6]
                meta.model.veto_lbda = output[7]
                meta.random_state = output[8]
        else:
            for meta in self.metas:
                meta.it_meta = it_meta
                meta.queue = Queue()
                meta.p = Process(target = self._process_optimize, args = (meta, nmeta))
                meta.p.start()
            for meta in self.metas:
                output = self.queue_get_retry(meta.queue)
                meta.ca = output[0]
                meta.ca_good = output[1]
                meta.model.bpt = output[2]
                meta.model.cv = output[3]
                meta.model.lbda = output[4]
                meta.model.vpt = output[5]
                meta.model.veto_weights = output[6]
                meta.model.veto_lbda = output[7]
                meta.random_state = output[8]
                meta.auc = meta.model.auc(self.aa_ori, self.pt_sorted.pt)
 
        metas_sorted = self.sort_models(fct_ca)
                
        return metas_sorted[0].model, metas_sorted[0].ca, self.metas
    



class MetaMRSortOne():

    def __init__(self, model, pt_sorted, aa_ori, lp_weights = LpMRSortWeightsPositive, 
                 heur_profiles = MetaMRSortProfiles5, gamma = 0.5, version_meta = 4,
                 pretreatment_crit = None, duplication = False):
        """ Create an instance of META algorithm 
        for learning MR-Sort parameters for a single MR-Sort model """

        self.model = model
        self.pt_sorted = pt_sorted
        self.aa_ori = aa_ori
        self.lp_weights = lp_weights
        self.heur_profiles = heur_profiles
        self.duplication = duplication
        self.version_meta = version_meta
        self.gamma = gamma
        self.pretreatment_crit = pretreatment_crit
        self.it_meta = 0

        # Initialization of profiles values of the new MR-Sort model
        self.init_profiles()

        # Initialization/optimization of weight values 
        # of the new MR-Sort model
        self.lp = self.lp_weights(self.model, self.pt_sorted.pt, self.aa_ori, gamma = gamma, 
                                  version_meta = version_meta, pretreatment_crit = pretreatment_crit)
        self.lp.solve()
        
        # Optimization of profiles values of the new MR-Sort model
        self.meta = self.heur_profiles(self.model, self.pt_sorted, self.aa_ori)

        self.ca = self.meta.good / self.meta.na
        self.ca_good = 1 if (self.meta.na_good==0) else self.meta.good_good / self.meta.na_good
        

    def init_profiles(self):
        """ Initialization of profiles of an MR-Sort model """
        
        bpt = generate_random_profiles_msjp(self.model.profiles,self.model.criteria)
        self.model.bpt = bpt
        self.model.vpt = None


    def optimize(self, nmeta):
        """ Optimization of profiles of an MR-Sort model """

        self.lp.update_linear_program()
        self.lp.solve()
        self.meta.rebuild_tables()

        best_ca = self.meta.good / self.meta.na
        best_ca_good = 1 if (self.meta.na_good==0) else self.meta.good_good / self.meta.na_good
        best_bpt = self.model.bpt.copy()
        for i in range(nmeta):
            ca,ca_good = self.meta.optimize()
            if ca > best_ca:
                best_ca = ca
                best_ca_good = ca_good
                best_bpt = self.model.bpt.copy()
            if ca == 1:
                break

        self.model.bpt = best_bpt

        return best_ca,best_ca_good


