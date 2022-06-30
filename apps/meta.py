# -*- coding: utf-8 -*-

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
import csv
import datetime
import random
import time
import numpy as np
from pymcda.learning.meta_mrsort import MetaMRSort
from pymcda.types import CriterionValue, CriteriaValues
from pymcda.types import  Criteria, Criterion, PerformanceTable
from pymcda.types import AlternativePerformances
from pymcda.generate import generate_alternatives
from pymcda.generate import generate_random_mrsort_model
from pymcda.generate import generate_random_performance_table_msjp
from pymcda.pt_sorted import SortedPerformanceTable
from pymcda.utils import  add_errors_in_assignments
from copy import deepcopy

# Global variable DATADIR that contains the absolute path to the results directory named 'results-meta' located inside the project
DATADIR = os.getenv('DATADIR', '/Users/pegdwendestephaneminoungou/python_workspace/Spyder/pm-pymcda/results-meta')


class RandMRSortMetaLearning():
    
    def __init__(self, nb_alternatives, nb_categories, nb_criteria, dir_criteria, l_dupl_criteria, nb_tests, \
                nb_models, meta_l, meta_ll, meta_nb_models, noise = None, mu = 0.95, gamma = 0.5, fixed_profc1 = None, \
                version_meta = 4, pretreatment = False, fixed_w1 = None, model_with_right_nb_crit = 0, \
                model_heuristic = False, duplication = True):
        """ Create a new instance of test including a learning set,
        the parameters of the problem for the use algorithm META 
        in order to learn an MR-Sort model """

        self.nb_alternatives = nb_alternatives
        self.nb_categories = nb_categories
        self.nb_criteria = nb_criteria
        self.dir_criteria = dir_criteria
        self.nb_dupl_criteria = len(l_dupl_criteria)
        self.l_dupl_criteria = l_dupl_criteria
        self.nb_tests = nb_tests
        self.meta_l = meta_l
        self.meta_ll = meta_ll
        self.meta_nb_models = meta_nb_models
        self.nb_models = nb_models
        self.model = None
        self.pt = None
        self.noise = noise
        self.learned_models = []
        self.ca_avg = [0]*(self.nb_models)
        self.ca_tests_avg = [0]*(self.nb_models)
        self.ca_good_avg = 0
        self.ca_good_tests_avg = 0
        self.exec_time = [0]*(self.nb_models)
        self.mu = mu
        self.gamma = gamma
        self.nb_under_lim_prof_val = [None]*(self.nb_models)
        self.nb_under_lim_prof_test = [None]*(self.nb_models)
        self.stats_cav = 0
        self.stats_cag = 0
        self.stats_capd = 0
        self.stats_time = 0        


    def prepare_dupl_criteria_model(self):
        """ Re-encodes duplicated criteria 
        and assigns them opposite preference directions """
        
        lcriteria = []
        for j in range(self.nb_criteria):
            if j in self.l_dupl_criteria:
                lcriteria += [Criterion(id=list(self.model.criteria.values())[j].id , direction = self.dir_criteria[j], dupl_id=str(list(self.model.criteria.values())[j].id)+"d")]
            else :
                lcriteria += [Criterion(id=list(self.model.criteria.values())[j].id , direction = self.dir_criteria[j])]
        for j in self.l_dupl_criteria:
            j_id = list(self.model.criteria.values())[j].id
            lcriteria += [Criterion(id=str(j_id)+"d", direction = -self.dir_criteria[j], dupl_id=j_id)]
        return Criteria(lcriteria)

              
    
    def heuristic_preference_directions(self, crit):
        """ Heuristic for the choice of correct preference directions """
        
        prof_ord = self.pt_dupl_sorted.sorted_values[crit]
        prof_ord = [(prof_ord[0]/2)] + [(prof_ord[i-1]+prof_ord[i])/2 for i in range(1,len(prof_ord))] + [(prof_ord[-1]+1)/2]
        prof_max = {i : 0 for i in prof_ord}
        prof_min = {i : 0 for i in prof_ord}

        best_prof_max = {(i,j): None for i,j in zip(self.model.categories[:-1],self.model.categories[1:])}
        best_prof_min= {(j,i): None for i,j in zip(self.model.categories[1:][::-1],self.model.categories[:-1][::-1])}
        profiles = [(i,j,x,y) for (i,j),(x,y) in zip(best_prof_max,best_prof_min)]
        profiles = random.sample(profiles,len(profiles))

        for cat_i,cat_j,cat_x,cat_y in profiles:
            prof_max = {i : 0 for i in prof_ord}
            prof_min = {i : 0 for i in prof_ord}
            for eb in prof_max.keys():
                best_prof_max[cat_i,cat_j] = eb
                best_prof_min[cat_x,cat_y] = eb
                if sorted([i for i in best_prof_max.values() if i is not None])==[i for i in best_prof_max.values()  if i is not None]:
                    for alt in self.a:
                        k = self.model.categories.index(self.aa(alt.id))
                        lim_max = [None,None]
                        lim_min = [None,None]
                        if k == self.nb_categories-1:
                            l = [h for h in range(1,self.nb_categories) if best_prof_max[(self.model.categories[-1-h],self.model.categories[-h])] is not None][0]
                            lim_max = [best_prof_max[(self.model.categories[-1-l],self.model.categories[-l])],1]
                            l = [h for h in range(1,self.nb_categories) if best_prof_min[(self.model.categories[-1-h],self.model.categories[-h])] is not None][0]
                            lim_min = [0,best_prof_min[(self.model.categories[-1-l],self.model.categories[-l])]]
                        elif k==0:
                            u = [h for h in range(self.nb_categories-1) if best_prof_max[(self.model.categories[h],self.model.categories[h+1])] is not None][0]
                            lim_max = [0,best_prof_max[(self.model.categories[u],self.model.categories[u+1])]]
                            u = [h for h in range(self.nb_categories-1) if best_prof_min[(self.model.categories[h],self.model.categories[h+1])] is not None][0]
                            lim_min = [best_prof_min[(self.model.categories[u],self.model.categories[u+1])],1]
                        else:
                            l = [h for h in range(k) if best_prof_max[(self.model.categories[k-h-1],self.model.categories[k-h])] is not None]
                            lii = [h for h in range(k) if best_prof_max[(self.model.categories[k-h-1],self.model.categories[k-h])] is not None]
                            lim_max[0] = 0 if l == [] else best_prof_max[(self.model.categories[k-l[0]-1],self.model.categories[k-l[0]])]
                            u = [h for h in range(self.nb_categories-k-1) if best_prof_max[(self.model.categories[k+h],self.model.categories[k+h+1])] is not None]
                            uii = [h for h in range(self.nb_categories-k-1) if best_prof_max[(self.model.categories[k+h],self.model.categories[k+h+1])] is not None]
                            lim_max[1] = 1 if u == [] else best_prof_max[(self.model.categories[k+u[0]],self.model.categories[k+u[0]+1])]
                            
                            l = [h for h in range(self.nb_categories-k-1) if best_prof_min[(self.model.categories[k+h],self.model.categories[k+h+1])] is not None]
                            lim_min[0] = 0 if l == [] else best_prof_min[(self.model.categories[k+l[0]],self.model.categories[k+l[0]+1])]
                            u = [h for h in range(k) if best_prof_min[(self.model.categories[k-h-1],self.model.categories[k-h])] is not None]
                            lim_min[1] = 1 if u == [] else best_prof_min[(self.model.categories[k-u[0]-1],self.model.categories[k-u[0]])]
                        
                        if self.pt_dupl[alt.id].performances[crit]>=lim_max[0] and self.pt_dupl[alt.id].performances[crit]<=lim_max[1]:
                            prof_max[eb] += 1
                        if self.pt_dupl[alt.id].performances[crit]>=lim_min[0] and self.pt_dupl[alt.id].performances[crit]<=lim_min[1]:
                            prof_min[eb] += 1

            if sorted(prof_max.values(),reverse = True)>sorted(prof_min.values(),reverse = True):
                best_prof_max[cat_i,cat_j] = sorted(prof_max.items(),key=lambda k: k[1] ,reverse = True)[0][0]
                best_prof_min[cat_x,cat_y] = sorted(prof_max.items(),key=lambda k: k[1] ,reverse = True)[0][0]
            else:
                best_prof_max[cat_i,cat_j] = sorted(prof_min.items(),key=lambda k: k[1] ,reverse = True)[0][0]
                best_prof_min[cat_x,cat_y] = sorted(prof_min.items(),key=lambda k: k[1] ,reverse = True)[0][0]
        
        prof_max = sorted(prof_max.values(),reverse = True)
        prof_min = sorted(prof_min.values(),reverse = True)
        return 1 if (prof_max>prof_min) else -1


    def construct_intermediate_model(self):
        """ Assemble the intermediate model (profile, weights,lambda) 
        thanks to the deduced preference directions at the previous stage """
        
        right_criteria = []
        tmp_right_criteria = []
        wrong_w_sum = 0
        ref_crit = [i for i,j in self.model2.criteria.items()]
        for i,j in self.model2.criteria.items():
            if not j.dupl_id and i[-1] != "d":
                tmp_right_criteria += [(list(self.model2.cv.values())[ref_crit.index(i)].id, j.direction, float(list(self.model2.cv.values())[ref_crit.index(i)].value))]
            elif j.dupl_id and i[-1] != "d":
                if float(list(self.model2.cv.values())[ref_crit.index(i)].value) != 0 and float(list(self.model2.cv.values())[ref_crit.index(j.dupl_id)].value) == 0:
                    tmp_right_criteria += [(list(self.model2.cv.values())[ref_crit.index(i)].id ,1 , float(list(self.model2.cv.values())[ref_crit.index(i)].value))]
                elif float(list(self.model2.cv.values())[ref_crit.index(i)].value) == 0 and float(list(self.model2.cv.values())[ref_crit.index(j.dupl_id)].value) != 0:
                    tmp_right_criteria += [(list(self.model2.cv.values())[ref_crit.index(j.dupl_id)].id , -1, float(list(self.model2.cv.values())[ref_crit.index(j.dupl_id)].value))]
                elif float(list(self.model2.cv.values())[ref_crit.index(i)].value) != 0 and float(list(self.model2.cv.values())[ref_crit.index(j.dupl_id)].value) != 0:
                    tmp_max = dict()
                    tmp_min = dict()
                    for cat_i in self.model.bpt.keys():
                        tmp_max[cat_i] = 0
                        tmp_min[cat_i] = 0
                    for alt in self.a:
                        for cat_i in self.model.bpt.keys():
                            if self.model2.bpt[cat_i].performances[i] < self.pt_dupl[alt.id].performances[i]:
                                tmp_max[cat_i] += 1
                            if self.model2.bpt[cat_i].performances[j.dupl_id] < self.pt_dupl[alt.id].performances[j.dupl_id]:
                                tmp_min[cat_i] += 1
                    tmp_max=min([max(h,self.nb_alternatives-h) for h in tmp_max.values()])
                    tmp_min=min([max(h,self.nb_alternatives-h) for h in tmp_min.values()])
                    if (tmp_max > self.mu*self.nb_alternatives) and (tmp_min < self.mu*self.nb_alternatives):
                        tmp_right_criteria += [(list(self.model2.cv.values())[ref_crit.index(j.dupl_id)].id , -1, float(list(self.model2.cv.values())[ref_crit.index(j.dupl_id)].value))]
                        wrong_w_sum += float(list(self.model2.cv.values())[ref_crit.index(i)].value)
                    elif (tmp_max < self.mu*self.nb_alternatives) and (tmp_min > self.mu*self.nb_alternatives):
                        tmp_right_criteria += [(list(self.model2.cv.values())[ref_crit.index(i)].id , 1, float(list(self.model2.cv.values())[ref_crit.index(i)].value))]
                        wrong_w_sum += float(list(self.model2.cv.values())[ref_crit.index(j.dupl_id)].value)
                    else:
                        if self.heuristic_preference_directions(i) == 1:
                            tmp_right_criteria += [(list(self.model2.cv.values())[ref_crit.index(i)].id , 1, float(list(self.model2.cv.values())[ref_crit.index(i)].value))]
                            wrong_w_sum += float(list(self.model2.cv.values())[ref_crit.index(j.dupl_id)].value)
                        else:
                            tmp_right_criteria += [(list(self.model2.cv.values())[ref_crit.index(j.dupl_id)].id , -1, float(list(self.model2.cv.values())[ref_crit.index(j.dupl_id)].value))]
                            wrong_w_sum += float(list(self.model2.cv.values())[ref_crit.index(i)].value)
                else:
                    if self.heuristic_preference_directions(i) == 1:
                        tmp_right_criteria += [(list(self.model2.cv.values())[ref_crit.index(i)].id , 1, 0)]
                    else:
                        tmp_right_criteria += [(list(self.model2.cv.values())[ref_crit.index(j.dupl_id)].id , -1, 0)]
        cvals = CriteriaValues()
        for i,j,k in tmp_right_criteria:
            right_criteria += [Criterion(id=i ,direction=j)]
            cval = CriterionValue()
            cval.id = i
            cval.value = k/(1-wrong_w_sum)
            cvals.append(cval)
        self.model2.criteria = Criteria(right_criteria)
        self.model2.lbda = min(self.model2.lbda/(1-wrong_w_sum),1)
        self.model2.cv = cvals
        pt_tmp = PerformanceTable()
        for ap in self.model2.bpt:
            perfs_tmp = {}
            for c in self.model2.criteria:
                perfs_tmp[c.id] = ap.performances[c.id]
            ap_tmp = AlternativePerformances(ap.id, perfs_tmp)
            pt_tmp.append(ap_tmp)
        self.model2.bpt = pt_tmp

        

    def run_mrsort(self):
        """ Runs one trial of the META algorithm """
        
        # Setting for running a second time Sobrie heuristic
        categories = self.model.categories_profiles.to_categories()
        if self.noise is None:
            meta = MetaMRSort(self.meta_nb_models, self.dupl_model_criteria, self.l_dupl_criteria,\
                                        categories, self.pt_dupl_sorted, self.aa, gamma = self.gamma)
        else:
            meta = MetaMRSort(self.meta_nb_models, self.dupl_model_criteria, self.l_dupl_criteria,\
                                        categories, self.pt_dupl_sorted, self.aa_noisy, gamma = self.gamma)
        t1 = time.time()
        for i in range(self.meta_l):
            self.model2, ca, self.all_models = meta.optimize(self.meta_ll,0)
            if ca == 1:
                break

        # Constructs the intermediate model
        self.construct_intermediate_model()
        
        # Setting for running a second time Sobrie heuristic
        self.dupl_model_criteria = deepcopy(self.model2.criteria)
        if self.noise is None:
            meta = MetaMRSort(self.meta_nb_models, self.dupl_model_criteria, self.l_dupl_criteria,\
                                        categories, self.pt_dupl_sorted, self.aa, gamma = self.gamma)
        else:
            meta = MetaMRSort(self.meta_nb_models, self.dupl_model_criteria, self.l_dupl_criteria,\
                                        categories, self.pt_dupl_sorted, self.aa_noisy, gamma = self.gamma)
        for i in range(self.meta_l):
            self.model2, ca, self.all_models = meta.optimize(self.meta_ll,0)
            if ca == 1:
                break
        t2 = time.time()
        self.exec_time[self.num_model] = (t2-t1)
        
        
        return self.exec_time[self.num_model]


    def eval_model_validation(self):
        """ Evaluates metrics regarding the learning process 
        such as the learning classification accuracy """
        
        self.nb_under_lim_prof_val[self.num_model]=dict()
        self.aa_learned = self.model2.get_assignments(self.pt_dupl)
        total = len(self.a)
        nok = 0
        totalg = 0
        okg = 0
        for alt in self.a:
            if self.aa(alt.id) != self.aa_learned(alt.id):
                nok += 1
            if self.aa(alt.id) == "cat1":
                totalg += 1
                if self.aa_learned(alt.id) == "cat1":
                    okg += 1

        self.nb_under_lim_prof_val[self.num_model]["c1"] = self.nb_dupl_criteria
        for i,j in self.model2.criteria.items():
            if i[-1] == "d":
                self.nb_under_lim_prof_val[self.num_model]["c1"] -= 1
        
        totalg = 1 if totalg == 0 else totalg
        self.ca_avg[self.num_model] = float(total-nok)/total
        self.ca_good_avg += (float(okg)/totalg)

        return (float(total-nok)/total),(float(okg)/totalg)


    def eval_model_test(self):
        """ Evaluates metrics regarding the test process 
        such as the classification accuracy in generalization """

        self.nb_under_lim_prof_test[self.num_model]=dict()
        a_tests = generate_alternatives(self.nb_tests)
        pt_tests,pt_tests_dupl = generate_random_performance_table_msjp(a_tests, self.model.criteria, dupl_crits = self.dupl_model_criteria)
        ao_tests = self.model.get_assignments(pt_tests)
        
        al_tests = self.model2.get_assignments(pt_tests_dupl)
        total = len(a_tests)
        nok = 0
        totalg = 0
        okg = 0
        for alt in a_tests:
            if ao_tests(alt.id) != al_tests(alt.id):
                nok += 1
            if ao_tests(alt.id) == "cat1":
                totalg +=1
                if al_tests(alt.id) == "cat1":
                    okg += 1

        totalg = 1 if totalg == 0 else totalg
        self.ca_tests_avg[self.num_model] = float(total-nok)/total
        self.ca_good_tests_avg += (float(okg)/totalg)

        return ao_tests,al_tests,(float(total-nok)/total),(float(okg)/totalg)



    def run_mrsort_all_models(self):
        """ Repetes 'nb_models' times the MR-Sort learning process 
        with the same problem parameters 
        from randomly generated learning sets """
        
        # Generation of a new learning set 
        self.report_stats_parameters_csv()
        classif_tolerance_prop = 0.1
        self.a = generate_alternatives(self.nb_alternatives)
        for m in range(self.nb_models):
            self.num_model = m
            b_inf = (self.nb_alternatives * 1.0 /self.nb_categories)-(classif_tolerance_prop*self.nb_alternatives)
            b_sup = (self.nb_alternatives * 1.0 /self.nb_categories)+(classif_tolerance_prop*self.nb_alternatives)
            notfound = True
            while notfound :
                self.model = generate_random_mrsort_model(self.nb_criteria, self.nb_categories, random_directions = self.dir_criteria)
                self.dupl_model_criteria = self.prepare_dupl_criteria_model()
                self.pt,self.pt_dupl = generate_random_performance_table_msjp(self.a, self.model.criteria, dupl_crits = self.dupl_model_criteria)
                self.aa = self.model.get_assignments(self.pt)
                i = 1
                size = len(self.aa.get_alternatives_in_category('cat'+str(i)))
                while (size >= b_inf) and (size <= b_sup):
                    if i == self.nb_categories:
                        notfound = False
                        break
                    i += 1
                    size = len(self.aa.get_alternatives_in_category('cat'+str(i)))
            self.pt_dupl_sorted = SortedPerformanceTable(self.pt_dupl)
            self.report_original_model_param_csv()
            if self.noise != None:
                self.aa_noisy = deepcopy(self.aa)
                self.aa_err_only = add_errors_in_assignments(self.aa_noisy, self.model.categories, self.noise)
            
            # Execution of the META algorithm 
            self.run_mrsort()

            # Extraction of results on the learning phase
            ca_v,cag_v = self.eval_model_validation()
            
            # Collection of results on the test phase         
            ao_tests,al_tests,ca_t,cag_t = self.eval_model_test()

            # Exports the results into a csv file 
            self.report_stats_model_csv()
        self.report_summary_results_csv()




    def report_stats_parameters_csv(self):
        """ Report the global parameters of tests into a csv file """
        
        str_noise = "_err" + str(int(self.noise*100)) if self.noise != None else ""
        if self.dir_criteria is None:
            self.output_dir = "/random_tests_meta_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.nb_categories))  + "_dupl" + str(len(self.l_dupl_criteria)) + str_noise + "/"
        else:
            self.output_dir = "/random_tests_meta_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.dir_criteria.count(1))) + "-" + str(int(self.dir_criteria.count(-1))) + "_dupl" + str(len(self.l_dupl_criteria)) + str_noise + "/"
        if not os.path.exists(DATADIR + self.output_dir):
            os.mkdir(DATADIR + self.output_dir)
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.filename_valid = "%s/random_tests_details_meta-%d-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, len(self.l_dupl_criteria), dt)
        with open(self.filename_valid, "a") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")
            writer.writerow(['PARAMETERS'])
            writer.writerow([',algorithm,', 'META'])
            writer.writerow([',nb_alternatives,', self.nb_alternatives])
            writer.writerow([',nb_categories,', self.nb_categories])
            writer.writerow([',nb_criteria,', self.nb_criteria])
            writer.writerow([',nb_unknown_pref_dirs,', str(len(self.l_dupl_criteria))])
            writer.writerow([',nb_outer_loops_meta,', self.meta_l])
            writer.writerow([',nb_inner_loops_meta,', self.meta_ll])
            writer.writerow([',nb_models_pop,', self.meta_nb_models])
            writer.writerow([',nb_repetitions,', self.nb_models])
            writer.writerow([',nb_alternatives_test,', self.nb_tests])
            writer.writerow([',noise,', str(self.noise)])
            writer.writerow([',gamma,', self.gamma])
            writer.writerow([',mu,', self.mu])                
            
            
            
    def report_original_model_param_csv(self):
        """ Reports the statistics/details of the origin model
        and learning set of each learning trial """
        
        with open(self.filename_valid, "a") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")
            writer.writerow(['original_model ', self.num_model])
            writer.writerow([',criteria,', ",".join([str(i) for i,j in self.model.criteria.items()])])
            writer.writerow([',criteria_direction,', ",".join([str(i.direction) for i in self.model.criteria.values()])])
            for cat_i in self.model.bpt.keys():
                writer.writerow([',profiles_values_'+cat_i+',', ",".join([str(self.model.bpt[cat_i].performances[i]) for i,j in self.model.criteria.items()])])
            writer.writerow([',criteria_weights,', ",".join([str(i.value) for i in self.model.cv.values()])])
            writer.writerow([',original_lambda,',self.model.lbda])
            writer.writerow(['learning_set', self.num_model])
            writer.writerow([',assignments_id,', ",".join([str(i.id) for i in self.aa])])
            writer.writerow([',assignments_cat,',",".join([str(i.category_id) for i in self.aa])])
            writer.writerow([',nb_cat1',","+str([str(i.category_id) for i in self.aa].count("cat1"))])
            writer.writerow([',nb_cat2',","+str([str(i.category_id) for i in self.aa].count("cat2"))])
            
    


    def report_stats_model_csv(self):
        """ Reports the statistics/details of the learned models
        and metrics of each learning trial into a csv file """        
        
        with open(self.filename_valid, "a") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")
            writer.writerow(['result ', self.num_model])
            writer.writerow([',criteria,', ",".join([i for i,j in self.model2.criteria.items()])])
            writer.writerow([',criteria_direction,', ",".join([str(i.direction) for i in self.model2.criteria.values()])])
            for cat_i in self.model.bpt.keys():
                writer.writerow([',profiles_values_'+cat_i+',', ",".join([str(self.model2.bpt[cat_i].performances[i]) for i,j in self.model2.criteria.items()])])
            writer.writerow([',criteria_weights,', ",".join([str(i.value) for i in self.model2.cv.values()])])
            writer.writerow([',lambda,', self.model2.lbda])
            if self.aa_learned:
                writer.writerow([',assignments_id,', ",".join([str(i.id) for i in self.aa_learned])])
                writer.writerow([',assignments_cat,',",".join([str(i.category_id) for i in self.aa_learned])])
            writer.writerow([',execution_time,', str(self.exec_time[self.num_model])])
            writer.writerow([',CA,', str(self.ca_avg[self.num_model])])
            writer.writerow([',CA_tests,', str(self.ca_tests_avg[self.num_model])])
            if self.nb_dupl_criteria:
                writer.writerow([',PDCA1,', str((sum(self.nb_under_lim_prof_val[self.num_model].values()))//self.nb_dupl_criteria)])
                writer.writerow([',PDCA2,', str((sum(self.nb_under_lim_prof_val[self.num_model].values()))/self.nb_dupl_criteria)])


    def report_summary_results_csv(self):
        """ Reports the average/std metrics 
        over the 'nb_models' trials into a csv file """
        
        with open(self.filename_valid, "a") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")
            writer.writerow(['SUMMARY'])
            self.stats_time = self.exec_time
            writer.writerow([',exec_time_avg,' , str(np.mean(self.stats_time))])
            writer.writerow([',exec_time_std,' , str(np.std(self.stats_time))])
            self.stats_cav = self.ca_avg
            writer.writerow([',CAv_avg,', str(np.mean(self.ca_avg))])
            writer.writerow([',CAv_std,', str(np.std(self.ca_avg))])
            self.stats_cag = self.ca_tests_avg    
            writer.writerow([',CAg_avg,', str(np.mean(self.ca_tests_avg))])
            writer.writerow([',CAg_std,', str(np.std(self.ca_tests_avg))])
            if self.nb_dupl_criteria > 0:
                self.stats_pdca2 = [sum(self.nb_under_lim_prof_val[i].values())/self.nb_dupl_criteria for i in range(self.nb_models)]
                if self.nb_dupl_criteria == 1:
                    self.stats_pdca1 = self.stats_pdca2
                    writer.writerow([',PDCA1_avg,', str(np.mean(self.stats_pdca1))])
                    writer.writerow([',PDCA1_std,', str(np.std(self.stats_pdca1))])
                    writer.writerow([',PDCA2_avg,', str(np.mean(self.stats_pdca2))])
                    writer.writerow([',PDCA2_std,', str(np.std(self.stats_pdca2))])
                elif self.nb_dupl_criteria > 1:
                    self.stats_pdca1 = [sum(self.nb_under_lim_prof_val[i].values())//self.nb_dupl_criteria for i in range(self.nb_models)]
                    writer.writerow([',PDCA1_avg,', str(np.mean(self.stats_pdca1))])
                    writer.writerow([',PDCA1_std,', str(np.std(self.stats_pdca1))])
                    writer.writerow([',PDCA2_avg,', str(np.mean(self.stats_pdca2))])
                    writer.writerow([',PDCA2_std,', str(np.std(self.stats_pdca2))])


    def report_plot_results_csv(self):
        """ Reports the summarized metrics 
        of the 'nb_models' trials into a csv file """

        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "%s/summary_results_meta-%d-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, len(self.l_dupl_criteria), dt)
        with open(filename, "w") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")
            writer.writerow(['SUMMARY DETAILS LIST'])
            writer.writerow([',exec_time,' , [round(i,2) for i in self.exec_time]])
            writer.writerow([',CAv,', [round(i,2) for i in self.ca_avg]])
            writer.writerow([',CAg,', [round(i,2) for i in self.ca_tests_avg]])
            if self.nb_dupl_criteria > 0:
                writer.writerow([',PDCA1_avg,', [round(i,2) for i in self.stats_pdca1]])
                writer.writerow([',PDCA2_avg,', [round(i,2) for i in self.stats_pdca2]])



    def build_osomcda_instance_random(self):
        """ Builds an instance composed of the list of criteria, 
        a performance table and assigned categories (assignment examples)
        as a csv file"""
        
        criteria = [f.id for f in self.dupl_model_criteria]
        nb_crits = len(criteria) 
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "%s/instance-%d-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, len(self.l_dupl_criteria), dt)
        with open(filename, "w") as tmp_newfile:
            out = csv.writer(tmp_newfile, delimiter=" ")
            out.writerow(["criterion,direction," + ("," * (nb_crits - 2))])
            for crit in self.model.criteria:
                out.writerow([crit.id + "," + str(1) + "," + ("," * (nb_crits - 2))])
            for i in range(self.nb_criteria):
                if i in self.l_dupl_criteria:
                    out.writerow([list(self.model.criteria.values())[i].id + "d," + str(-1) + "," + ("," * (nb_crits - 2))])
            out.writerow(["," * (nb_crits)])
            out.writerow(["category,rank," + ("," * (nb_crits - 2))])
            for i in self.model.categories_profiles.to_categories().values():
                out.writerow([i.id + "," + str(i.rank) + "," + ("," * (nb_crits - 2))])
            out.writerow(["," * (nb_crits)])
            out.writerow(["pt," + ",".join(criteria) + ",assignment"])
            for pt_values in self.pt.values():
                nrow = [str(pt_values.performances[el.id]) for el in self.model.criteria]
                dupl_nrow = [str(pt_values.performances[list(self.model.criteria.values())[i].id]) for i in range(self.nb_criteria) if i in self.l_dupl_criteria]
                if self.l_dupl_criteria:
                    out.writerow(["pt" + pt_values.id + "," + ",".join(nrow) + "," + ",".join(dupl_nrow) + "," + self.aa[pt_values.id].category_id])
                else :
                    out.writerow(["pt" + pt_values.id + "," + ",".join(nrow)  + "," + self.aa[pt_values.id].category_id])
        return filename


    def learning_process(self):
        """ Unfold the learning process according to the settings 
        defined with the __init__ function : 
            - inference of MR-Sort parameters
            - export the results details into csv file"""
        
        if not os.path.exists(DATADIR):
            os.mkdir(DATADIR)
        self.run_mrsort_all_models()
        self.report_plot_results_csv()
        if self.nb_models==1:
            self.build_osomcda_instance_random()



if __name__ == "__main__":
        
    # PROBLEM PARAMETERS
    
    # The number of categories
    nb_categories = 2
    
    # The number of criteria of the problem
    nb_criteria = 5
    
    # The number of alternatives of the learning set
    nb_alternatives = 50
    
    # Percentage of noise to be introduced in the learning set 
    # (0 by default) : 0.05 for 5% of noise
    noise = 0
    
    # The list of preference direction of criteria 
    # (All criteria fixed to gain criteria)
    dir_criteria = [1,1,1,1,1]

    # The list of indices of criteria with unknown preference directions : 
    # if empty, then criteria are all gain criteria (i.e. all are known)
    # else, insert index 0 for c1, 1 for c2, 2 for c3, etc ...
    l_dupl_criteria = [0,1,2]
    
    # Number of alternatives in the tests set
    nb_tests = 10000
    
    # Number of random trials 
    nb_models = 1


    # ALGORITHM PARAMETERS
    
    # Number of iteration of the metaheuristic algorithm (outer loop)
    meta_l = 30
    
    # The number of iteration of the metaheuristic algorithm (inner loop)
    meta_ll = 20
    
    # The number of models (population)
    meta_nb_models = 10
    

    # Creation of an instance of test for inferring an MR-Sort model with META algorithm
    inst = RandMRSortMetaLearning(nb_alternatives, nb_categories, nb_criteria, dir_criteria, l_dupl_criteria, nb_tests,  nb_models, meta_l, meta_ll, 
            meta_nb_models, noise = noise)
    
    # Running the META algorithm to learn model parameters 
    # Export tests results in csv files
    inst.learning_process()
    




