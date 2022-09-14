# -*- coding: utf-8 -*-

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
import csv
import datetime
import random
import time
import numpy as np
from pymcda.learning.meta_mrsort_sp import MetaMRSortVCPop4MSJP_SP
from pymcda.types import CriterionValue, CriteriaValues
from pymcda.types import AlternativeAssignment, AlternativesAssignments
from pymcda.generate import generate_alternatives
from pymcda.pt_sorted import SortedPerformanceTable
from pymcda.generate import generate_random_mrsort_model
from pymcda.generate import generate_random_performance_table_msjp
from pymcda.utils import add_errors_in_assignments
from copy import deepcopy


# Global variable DATADIR that contains the absolute path to the results directory named 'results-meta-sp' located inside the project
DATADIR = os.getenv('DATADIR', os.path.abspath('.') + 'results-meta-sp')


class RandMRSortMetaSPLearning():
    
    def __init__(self, nb_alternatives, nb_categories, nb_criteria, dir_criteria, l_dupl_criteria, nb_tests, \
                nb_models, meta_l, meta_ll, meta_nb_models, noise = None, mu = 0.95, gamma = 0.5, fixed_profc1 = None, \
                version_meta = 4, pretreatment = False, fixed_w1 = None, model_with_right_nb_crit = 0, \
                model_heuristic = False, duplication = False, strat_heur = [], file_datasets = "", testset_perc=0, current_instance = ""):
        """ Create a new instance of test including a learning set,
        the parameters of the problem for the use algorithm META-SP 
        in order to learn an MR-Sort model """
        
        self.nb_alternatives = nb_alternatives
        self.nb_categories = nb_categories
        self.nb_criteria = nb_criteria
        self.dir_criteria = dir_criteria
        self.meta_l = meta_l
        self.meta_ll = meta_ll
        self.meta_nb_models = meta_nb_models
        self.nb_tests = nb_tests
        self.nb_models = nb_models
        self.model = None
        self.pt = None
        self.noise = noise
        self.learned_models = []
        self.ca_avg = [0]*(self.nb_models)
        self.ca_tests_avg = [0]*(self.nb_models)
        self.ca_good_avg = 0
        self.ca_good_tests_avg = 0
        self.ca = 0
        self.mu = mu
        self.gamma = gamma
        self.exec_time = [0]*(self.nb_models)
        self.stats_cav = 0
        self.stats_cag = 0
        self.stats_time = 0
        self.strat_heur = [1,1,1]

        

    def run_mrsort(self):
        """ Runs one trial of the META-SP algorithm """

        categories = self.model.categories_profiles.to_categories()
        self.dupl_model_criteria = self.model.criteria
        if self.noise is None:
            meta = MetaMRSortVCPop4MSJP_SP(self.meta_nb_models, self.dupl_model_criteria,\
                                        categories, self.pt_dupl_sorted, self.aa, gamma = self.gamma,\
                                        known_lbda=self.model.lbda, known_criteria=self.model.criteria, known_cv=self.model.cv, strat_heur=self.strat_heur)
        else:
            meta = MetaMRSortVCPop4MSJP_SP(self.meta_nb_models, self.dupl_model_criteria,\
                                        categories, self.pt_dupl_sorted, self.aa_noisy, gamma = self.gamma,\
                                        known_lbda=self.model.lbda, known_criteria=self.model.criteria, known_cv=self.model.cv, strat_heur=self.strat_heur)
        t1 = time.time()
        for i in range(self.meta_l):
            self.model2, ca, self.all_models,nbit = meta.optimize(self.meta_ll,0)
            if ca == 1:
                break

        # Choice of a better representative model
        for m in self.all_models:
            if m.ca == ca :
                if [ff.value for ff in m.model.cv.values()].count(0) < [ff.value for ff in self.model2.cv.values()].count(0):
                    self.mdoel2 = m.model

        self.exec_time[self.num_model] = (time.time() - t1)
        return self.exec_time[self.num_model]

    

    def get_assignment_sp(self, tmp_model, ap):
        """ Assigns an example following the MR-Sort-SP rule """
        
        categories = list(reversed(tmp_model.categories))
        cat = categories[0]
        cw = 0
        i=0
        for cat_i in tmp_model.bpt.keys():
            cw = 0
            for j in tmp_model.cv.keys():
                if tmp_model.criteria[j].direction==2:
                    if ap.performances[j] >= tmp_model.bpt_sp[cat_i].performances[j][0] and ap.performances[j] <= tmp_model.bpt_sp[cat_i].performances[j][1]:
                        cw+=tmp_model.cv[j].value
                elif tmp_model.criteria[j].direction==-2:
                    if ap.performances[j] <= tmp_model.bpt_sp[cat_i].performances[j][0] or ap.performances[j] >= tmp_model.bpt_sp[cat_i].performances[j][1]:
                        cw+=tmp_model.cv[j].value
                else:
                    if tmp_model.criteria[j].direction==1 and ap.performances[j] >= tmp_model.bpt[cat_i].performances[j]:
                        cw+=tmp_model.cv[j].value
                    if tmp_model.criteria[j].direction==-1 and ap.performances[j] <= tmp_model.bpt[cat_i].performances[j]:
                        cw+=tmp_model.cv[j].value
            if round(cw,10) >= round(tmp_model.lbda,10):
                break
            cat = categories[i + 1]
            i+=1
        return AlternativeAssignment(ap.id, cat)

    
    
    def get_assignments_sp(self, tmp_model, tmp_pt):
        """ Assigns examples following the MR-Sort-SP rule """
        
        aa = AlternativesAssignments()
        for ap in tmp_pt:
            a = self.get_assignment_sp(tmp_model, ap)
            aa.append(a)
        return aa



    def eval_model_validation(self):
        """ Evaluates metrics regarding the learning process 
        such as the learning classification accuracy """
        
        self.aa_learned = self.get_assignments_sp(self.model2, self.pt_dupl)
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
        totalg = 1 if totalg == 0 else totalg
        self.ca_avg[self.num_model] = float(total-nok)/total
        self.ca_good_avg += (float(okg)/totalg)

        return (float(total-nok)/total),(float(okg)/totalg)


    def eval_model_test(self):
        """ Evaluates metrics regarding the test process 
        such as the classification accuracy in generalization """

        a_tests = generate_alternatives(self.nb_tests)
        pt_tests,pt_tests_dupl = generate_random_performance_table_msjp(a_tests, self.model.criteria, dupl_crits = self.dupl_model_criteria)
        ao_tests = self.get_assignments_sp(self.model, pt_tests)
        al_tests = self.get_assignments_sp(self.model2, pt_tests_dupl)
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



    def random_criteria_weights(self,n,k=3):
        """ Generates random values as weights of an MR-Sort-SP model """
        
        cval = [0]*n
        no_min_w =  True
        while no_min_w:
            random.seed()
            weights = [round(random.random(),k) for i in range(n - 1) ]
            weights += [0,1]
            weights.sort()
            if [i for i in range(n) if abs(weights[i]-weights[i+1]) < 0.05] == [] :
                no_min_w = False
        for i in range(n):
            cval[i] = round(weights[i+1] - weights[i], k)
        return cval



    def run_mrsort_all_models(self):
        """ Repetes 'nb_models' times the MR-Sort-SP learning process 
        with similar problem parameters 
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
                self.model.lbda = round(random.uniform(0, 1), 2)
                for i in range(self.nb_criteria):
                    if abs(self.dir_criteria[i])==2:
                        ttmp = tuple(sorted([round(random.uniform(0.1,0.85),2),round(random.uniform(0.1,0.85),2)]))
                        self.model.bpt_sp['b1'].performances["c"+str(i+1)] = (ttmp[0],round(ttmp[1]+0.05,2))
                    else:
                        ttmp = round(random.uniform(0.1,0.9),2)
                        self.model.bpt['b1'].performances["c"+str(i+1)] = ttmp
                        self.model.bpt_sp['b1'].performances["c"+str(i+1)] = ttmp
                cvals = CriteriaValues()
                tmp = self.random_criteria_weights(self.nb_criteria,k=2)
                for i in range(self.nb_criteria):
                    cval = CriterionValue()
                    cval.id = "c"+str(i+1)
                    cval.value = tmp[i]
                    cvals.append(cval)
                self.model.cv = cvals
                self.pt,self.pt_dupl = generate_random_performance_table_msjp(self.a, self.model.criteria, dupl_crits = [], k=1)
                self.aa = self.get_assignments_sp(self.model, self.pt)
                self.pt_dupl_sorted = SortedPerformanceTable(self.pt_dupl)
                i = 1
                size = len(self.aa.get_alternatives_in_category('cat'+str(i)))
                while (size >= b_inf) and (size <= b_sup):
                    if i == self.nb_categories:
                        notfound = False
                        break
                    i += 1
                    size = len(self.aa.get_alternatives_in_category('cat'+str(i)))                        
            self.report_original_model_param_csv()
            if self.noise != None:
                self.aa_noisy = deepcopy(self.aa)
                self.aa_err_only = add_errors_in_assignments(self.aa_noisy, self.model.categories, self.noise)

            # Execution of the META-SP algorithm 
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
            self.output_dir = "/random_tests_meta_sp_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.nb_categories)) + str_noise + "/"
        else:
            self.output_dir = "/random_tests_meta_sp_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.nb_criteria)) + "-" + str(int(self.dir_criteria.count(1))) + "-" + str(int(self.dir_criteria.count(-1))) + "-" + str(int(self.dir_criteria.count(2))) + "-" + str(int(self.dir_criteria.count(-2))) + str_noise + "/"
        if not os.path.exists(DATADIR + self.output_dir):
            os.mkdir(DATADIR + self.output_dir)
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.filename_valid = "%s/random_tests_details_meta_sp-rand-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, dt)
        
        with open(self.filename_valid, "a") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")
            writer.writerow(['PARAMETERS'])
            writer.writerow([',algorithm,', 'META-SP'])
            writer.writerow([',nb_alternatives,', self.nb_alternatives])
            writer.writerow([',nb_categories,', self.nb_categories])
            writer.writerow([',nb_criteria,', self.nb_criteria])
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
                writer.writerow([',profiles_values_'+cat_i+',', ",".join([str(self.model.bpt_sp[cat_i].performances[i][0]) if abs(j.direction)==2 else str(self.model.bpt[cat_i].performances[i]) for i,j in self.model.criteria.items()])])
                writer.writerow([',profiles2_values_'+cat_i+',', ",".join([str(self.model.bpt_sp[cat_i].performances[i][1]) if abs(j.direction)==2 else "" for i,j in self.model.criteria.items()])])
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
                writer.writerow([',profiles_values_'+cat_i+',', ",".join([str(self.model2.bpt_sp[cat_i].performances[i][0]) if abs(j.direction)==2 else str(self.model2.bpt[cat_i].performances[i]) for i,j in self.model2.criteria.items()])])
                writer.writerow([',profiles2_values_'+cat_i+',', ",".join([str(self.model2.bpt_sp[cat_i].performances[i][1]) if abs(j.direction)==2 else "" for i,j in self.model2.criteria.items()])])
            writer.writerow([',criteria_weights,', ",".join([str(i.value) for i in self.model2.cv.values()])])
            writer.writerow([',lambda,', self.model2.lbda])            
            if self.aa_learned:
                writer.writerow([',assignments_id,', ",".join([str(i.id) for i in self.aa_learned])])
                writer.writerow([',assignments_cat,',",".join([str(i.category_id) for i in self.aa_learned])])
            writer.writerow([',execution_time,', str(self.exec_time[self.num_model])])
            writer.writerow([',CA,', str(self.ca_avg[self.num_model])])
            writer.writerow([',CA_tests,', str(self.ca_tests_avg[self.num_model])])
                        


    def report_summary_results_csv(self):
        """ Reports the average/std metrics 
        over the 'nb_models' trials into a csv file """

        with open(self.filename_valid, "a") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")
            writer.writerow(['SUMMARY'])
            self.stats_time = self.exec_time
            writer.writerow([',exec_time_avg,' , str(np.mean([x for x in self.stats_time if x is not None]))])
            writer.writerow([',exec_time_std,' , str(np.std([x for x in self.stats_time if x is not None]))])
            self.stats_cav = self.ca_avg
            writer.writerow([',CAv_avg,', str(np.mean([x for x in self.ca_avg if x is not None]))])
            writer.writerow([',CAv_std,', str(np.std([x for x in self.ca_avg if x is not None]))])
            self.stats_cag = self.ca_tests_avg    
            writer.writerow([',CAg_avg,', str(np.mean([x for x in self.ca_tests_avg if x is not None]))])
            writer.writerow([',CAg_std,', str(np.std([x for x in self.ca_tests_avg if x is not None]))])




    def report_plot_results_csv(self):
        """ Reports the summarized metrics 
        of the 'nb_models' trials into a csv file """

        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "%s/summary_results_meta_sp-rand-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, dt)
        with open(filename, "w") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")
            writer.writerow(['SUMMARY DETAILS LIST'])
            writer.writerow([',exec_time,' , [round(i,2) for i in self.exec_time if i is not None]])
            writer.writerow([',CAv,', [round(i,2) for i in self.ca_avg if i is not None]])
            writer.writerow([',CAg,', [round(i,2) for i in self.ca_tests_avg if i is not None]])



    def build_osomcda_instance_random(self):
        """ Builds an instance composed of the list of criteria, 
        a performance table and assigned categories (assignment examples)
        as a csv file"""
        
        criteria = [f.id for f in self.dupl_model_criteria]
        nb_crits = len(criteria) 
        
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "%s/instance_meta_sp-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, dt)

        with open(filename, "w") as tmp_newfile:
            out = csv.writer(tmp_newfile, delimiter=" ")
            out.writerow(["criterion,direction," + ("," * (nb_crits - 2))])
            for crit in self.model.criteria:
                out.writerow([crit.id + "," + str(1) + "," + ("," * (nb_crits - 2))])
            out.writerow(["," * (nb_crits)])
            out.writerow(["category,rank," + ("," * (nb_crits - 2))])
            for i in self.model.categories_profiles.to_categories().values():
                out.writerow([i.id + "," + str(i.rank) + "," + ("," * (nb_crits - 2))])
            out.writerow(["," * (nb_crits)])
            out.writerow(["pt," + ",".join(criteria) + ",assignment"])
            for pt_values in self.pt.values():
                nrow = [str(pt_values.performances[el.id]) for el in self.model.criteria]
                out.writerow(["pt" + pt_values.id + "," + ",".join(nrow)  + "," + self.aa[pt_values.id].category_id])
        return filename



    def learning_process(self):
        """ Unfold the learning process according to the settings 
        defined with the __init__ function : 
            - inference of MR-Sort parameters
            - export the results details into csv file """
        
        if not os.path.exists(DATADIR):
            os.mkdir(DATADIR)
        self.run_mrsort_all_models()
        self.report_plot_results_csv()
        if self.nb_models==1:
            self.build_osomcda_instance_random()




if __name__ == "__main__":


    # PROBLEM PARAMETERS
    
    # The number of categories (fixed to 2)
    nb_categories = 2
    
    # The number of criteria of the problem
    nb_criteria = 5
    
    # The number of alternatives of the learning set
    nb_alternatives = 50
    
    # Percentage of noise to be introduced in the learning set 
    # (0 by default) : 0.05 for 5% of noise
    noise = 0
    
    # The list of preference direction of criteria 
    # 1: increasing, -1:decreasing, 2:single-peaked, -2:single-valley 
    dir_criteria = [2,1,1,1,1]

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

    # Creation of an instance of test for inferring an MR-Sort model with META-SP algorithm
    inst = RandMRSortMetaSPLearning(nb_alternatives, nb_categories, nb_criteria, dir_criteria, [], \
                    nb_tests, nb_models, meta_l, meta_ll, meta_nb_models, noise = noise)
    
    # Running the META-SP algorithm to learn model parameters 
    # Export tests results in csv files
    inst.learning_process()
    
    




