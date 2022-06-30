# -*- coding: utf-8 -*-

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
import csv
import datetime
import random
import numpy as np
from pymcda.electre_tri import MRSort
from pymcda.learning.mip_mrsort_sp import MipMRSort
from pymcda.types import CriterionValue, CriteriaValues
from pymcda.types import Criteria, Criterion
from pymcda.types import AlternativeAssignment, AlternativesAssignments
from pymcda.generate import generate_alternatives
from pymcda.pt_sorted import SortedPerformanceTable
from pymcda.generate import generate_random_mrsort_model
from pymcda.generate import generate_random_performance_table_msjp
from pymcda.utils import add_errors_in_assignments
from copy import deepcopy


# Global variable DATADIR that contains the absolute path to the results directory named 'results-mip-sp' located inside the project
DATADIR = os.getenv('DATADIR', '/Users/pegdwendestephaneminoungou/python_workspace/Spyder/pm-pymcda/results-mip-sp')


class RandMRSortMIPSPLearning():
    
    def __init__(self, nb_alternatives, nb_categories, nb_criteria, dir_criteria, l_unk_pref_dirs, nb_tests, nb_models, noise = None, current_instance = ""):
        """ Create a new instance of test including a learning set,
        the parameters of the problem for the use algorithm MIP-SP 
        in order to learn an MR-Sort model """

        self.nb_alternatives = nb_alternatives
        self.nb_categories = nb_categories
        self.nb_criteria = nb_criteria
        self.dir_criteria = dir_criteria
        self.nb_unk_pref_dirs = len(l_unk_pref_dirs)
        self.l_unk_pref_dirs = l_unk_pref_dirs
        self.nb_tests = nb_tests
        self.nb_models = nb_models
        self.model = None
        self.pt = None
        self.noise = noise
        self.ca_avg = [0]*(self.nb_models)
        self.ca_tests_avg = [0]*(self.nb_models)
        self.ca_good_avg = 0
        self.ca_good_tests_avg = 0
        self.ca = 0
        self.exec_time = [0]*(self.nb_models)
        self.pdca1 = [None]*(self.nb_models)
        self.pdca2 = [None]*(self.nb_models)
        self.stats_cav = 0
        self.stats_cag = 0
        self.stats_capd = 0
        self.stats_time = 0
        self.cplex_time =0
        self.dupl_model_criteria = []
        

    def run_mrsort(self):
        """ Runs one trial of the MIP-SP algorithm """
        
        # Prepares criteria with 'known' and 'unknown' preference directions :
        lcriteria = []
        for j in range(self.nb_criteria):
            if j in self.l_unk_pref_dirs:
                lcriteria += [Criterion(id=list(self.model.criteria.values())[j].id , direction = 0)] #if unknown by default 0 but will be set to 2 just before MIP execution
            else :
                lcriteria += [Criterion(id=list(self.model.criteria.values())[j].id , direction = self.dir_criteria[j])]
        
        # Runs the MIP for MR-Sort-SP models
        if self.noise is None:
            self.model2 = MRSort(Criteria(lcriteria), None, None, None, self.model.categories_profiles, None, None, None)
            mip = MipMRSort(self.model2, self.pt, self.aa)
            obj = mip.solve()
        else:
            self.model2 = MRSort(Criteria(lcriteria), None, None, None, self.model.categories_profiles, None, None, None)
            mip = MipMRSort(self.model2, self.pt, self.aa_noisy)
            obj = mip.solve()
            self.curr_status = obj[8]
            self.cplex_time = obj[9]
            
            # Assigning preferences directions to criteria with unknown preference directions
            self.pdca1[self.num_model] = dict()
            self.pdca2[self.num_model] = dict()
            if self.model2.bpt is not None:
                for cat_i in  self.model2.bpt.keys():
                    for j in range(self.nb_criteria):
                        if j in self.l_unk_pref_dirs:
                            sigma = [(i,round(k,5)) for i,k in obj[4] if list(self.model.criteria.values())[j].id==i][0]
                            interval = [round(h,5) for h in self.model2.bpt_sp[cat_i].performances[sigma[0]]]
                            if sigma[1]==1 and interval[0]==0:
                                self.model2.criteria[sigma[0]].direction = -1
                                self.model2.bpt[cat_i].performances[sigma[0]] = interval[1]
                            elif sigma[1]==1 and interval[1]==1:
                                self.model2.criteria[sigma[0]].direction = 1
                                self.model2.bpt[cat_i].performances[sigma[0]] = interval[0]
                            elif sigma[1]==1: #SP
                                self.model2.criteria[sigma[0]].direction = 2
                            elif sigma[1]==0: #SV
                                self.model2.criteria[sigma[0]].direction = -2
                            if self.model.criteria[sigma[0]].direction == self.model2.criteria[sigma[0]].direction:
                                self.pdca1[self.num_model][j] = 1
                                self.pdca2[self.num_model][j] = 1
                            else:
                                self.pdca1[self.num_model][j] = 0
                                self.pdca2[self.num_model][j] = 0
                        tmpi = list(self.model.criteria.values())[j].id
                        if isinstance(self.model2.bpt_sp[cat_i].performances[tmpi],tuple):
                            self.model2.bpt_sp[cat_i].performances[tmpi] = (round(self.model2.bpt_sp[cat_i].performances[tmpi][0],5),round(self.model2.bpt_sp[cat_i].performances[tmpi][1],5))
                        else:
                            self.model2.bpt_sp[cat_i].performances[tmpi] = round(self.model2.bpt_sp[cat_i].performances[tmpi],5)
        self.exec_time[self.num_model] = self.cplex_time
        
        return (self.exec_time[self.num_model], self.curr_status)

    

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
            if round(cw,7) >= round(tmp_model.lbda,7):
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
            weights = [random.random() for i in range(n - 1) ]
            weights += [0,1]
            weights.sort()
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
    
            # Execution of the MIP-SP algorithm 
            res1, status = self.run_mrsort()
    
            # Testing if the MIP program was aborted because of timeout
            if status==107 or status==108:
                self.mip_obj[self.num_model] = None
                self.mip_gap[self.num_model] = None
                self.mip_gamma[self.num_model] = None
                self.mip_sumbsp[self.num_model] = None
                self.mip_sigma[self.num_model] = None
                self.mip_bm[self.num_model] = None
                self.mip_b[self.num_model] = None
                self.pdca1[self.num_model] = None
                self.pdca2[self.num_model] = None
                self.exec_time[self.num_model] = None
                self.ca_avg[self.num_model] = None
                self.ca_tests_avg[self.num_model] = None
                self.model2 = None
            else:
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
            self.output_dir = "/random_tests_mip_sp_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.nb_categories))  + "_dupl" + str(len(self.l_dupl_criteria)) + str_noise + "/"
        else:
            self.output_dir = "/random_tests_mip_sp_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.nb_criteria)) + "-" + str(int(self.dir_criteria.count(1))) + "-" + str(int(self.dir_criteria.count(-1))) + "-" + str(int(self.dir_criteria.count(2))) + "-" + str(int(self.dir_criteria.count(-2))) + "_dupl" + str(len(self.l_unk_pref_dirs)) + str_noise + "/"
        if not os.path.exists(DATADIR + self.output_dir):
            os.mkdir(DATADIR + self.output_dir)

        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.filename_valid = "%s/random_tests_details_mip_sp-rand-%d-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, len(self.l_unk_pref_dirs), dt)
        with open(self.filename_valid, "a") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")
            writer.writerow(['PARAMETERS'])
            writer.writerow([',algorithm,', 'MIP-SP'])
            writer.writerow([',nb_alternatives,', self.nb_alternatives])
            writer.writerow([',nb_categories,', self.nb_categories])
            writer.writerow([',nb_criteria,', self.nb_criteria])
            writer.writerow([',nb_unk_pref_dirs,', str(len(self.l_unk_pref_dirs))])
            writer.writerow([',nb_repetitions,', self.nb_models])
            writer.writerow([',nb_alternatives_test,', self.nb_tests])
            writer.writerow([',noise,', str(self.noise)])
                
            
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
                #import pdb; pdb.set_trace()
                writer.writerow([',assignments_id,', ",".join([str(i.id) for i in self.aa_learned])])
                writer.writerow([',assignments_cat,',",".join([str(i.category_id) for i in self.aa_learned])])
            writer.writerow([',execution_time,', str(self.exec_time[self.num_model])])
            writer.writerow([',CA,', str(self.ca_avg[self.num_model])])
            writer.writerow([',CA_tests,', str(self.ca_tests_avg[self.num_model])])
            if self.l_unk_pref_dirs:
                writer.writerow([',PDCA1,', str((sum(self.pdca1[self.num_model].values()))//len(self.l_unk_pref_dirs))])
                writer.writerow([',PDCA2,', str((sum(self.pdca2[self.num_model].values()))/len(self.l_unk_pref_dirs))])

    
    def report_summary_results_csv(self):
        """ Reports the average/std metrics 
        over the 'nb_models' trials into a csv file.
        Only results of terminated instances are recorded """

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
            if len(self.l_unk_pref_dirs)!=0:
                writer.writerow([',PDCA1_avg,', str(np.mean([sum(self.pdca1[i].values())//len(self.l_unk_pref_dirs) for i in range(self.nb_models) if self.pdca1[i] ]))])
                writer.writerow([',PDCA1_std,', str(np.std([sum(self.pdca1[i].values())//len(self.l_unk_pref_dirs) for i in range(self.nb_models) if self.pdca1[i] ]))])
                writer.writerow([',PDCA2_avg,', str(np.mean([sum(self.pdca1[i].values())/len(self.l_unk_pref_dirs) for i in range(self.nb_models) if self.pdca1[i] ]))])
                writer.writerow([',PDCA2_std,', str(np.std([sum(self.pdca1[i].values())/len(self.l_unk_pref_dirs) for i in range(self.nb_models) if self.pdca1[i] ]))])


    def report_plot_results_csv(self):
        """ Reports the summarized metrics 
        of the 'nb_models' trials into a csv file """

        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "%s/summary_results_mip_sp-rand-%d-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, len(self.l_unk_pref_dirs), dt)
        with open(filename, "w") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")
            writer.writerow(['SUMMARY DETAILS LIST'])
            writer.writerow([',exec_time,' , [round(i,2) for i in self.exec_time if i is not None]])
            writer.writerow([',CAv,', [round(i,2) for i in self.ca_avg if i is not None]])
            writer.writerow([',CAg,', [round(i,2) for i in self.ca_tests_avg if i is not None]])
            if len(self.l_unk_pref_dirs)!=0:        
                writer.writerow([',PDCA1,', [round(sum(self.pdca1[i].values())//len(self.l_unk_pref_dirs),2)  for i in range(self.nb_models) if self.pdca1[i] ]])
                writer.writerow([',PDCA2,', [round(sum(self.pdca1[i].values())/len(self.l_unk_pref_dirs),2)  for i in range(self.nb_models) if self.pdca1[i] ]])



    def build_osomcda_instance_random(self):
        """ Builds an instance composed of the list of criteria, 
        a performance table and assigned categories (assignment examples)
        as a csv file"""
        
        criteria = [f.id for f in self.dupl_model_criteria]
        nb_crits = len(criteria) 
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "%s/instance_mip_sp-%d-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, len(self.l_unk_pref_dirs), dt)

        with open(filename, "w") as tmp_newfile:
            out = csv.writer(tmp_newfile, delimiter=" ")
            out.writerow(["criterion,direction," + ("," * (nb_crits - 2))])
            for crit in self.model.criteria:
                out.writerow([crit.id + "," + str(1) + "," + ("," * (nb_crits - 2))])
            for i in range(self.nb_criteria):
                if i in self.l_unk_pref_dirs:
                    out.writerow([list(self.model.criteria.values())[i].id + "d," + str(-1) + "," + ("," * (nb_crits - 2))])
            out.writerow(["," * (nb_crits)])
            out.writerow(["category,rank," + ("," * (nb_crits - 2))])
            for i in self.model.categories_profiles.to_categories().values():
                out.writerow([i.id + "," + str(i.rank) + "," + ("," * (nb_crits - 2))])
            out.writerow(["," * (nb_crits)])
            out.writerow(["pt," + ",".join([cri.id for cri in self.model.criteria]) + ",assignment"])
            for pt_values in self.pt.values():
                nrow = [str(pt_values.performances[el.id]) for el in self.model.criteria]
                #dupl_nrow = [str(pt_values.performances[list(self.model.criteria.values())[i].id]) for i in range(self.nb_criteria) if i in self.l_unk_pref_dirs]
                # if self.l_unk_pref_dirs:
                #     out.writerow(["pt" + pt_values.id + "," + ",".join(nrow) + "," + ",".join(dupl_nrow) + "," + self.aa[pt_values.id].category_id])
                # else :
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
    
    # The list of indices of criteria with unknown preference directions : 
    # if empty, then criteria are all gain criteria (i.e. all are known)
    # else, insert index 0 for c1, 1 for c2, 2 for c3, etc ...
    l_unk_pref_dirs = []


    # The list of preference direction of criteria 
    # 1: increasing, -1:decreasing, 2:single-peaked, -2:single-valley 
    dir_criteria = [-1,1,1,1,1]


    # The number of alternatives of the learning set
    nb_alternatives = 50
    
    # Percentage of noise to be introduced in the learning set 
    # (0 by default) : 0.05 for 5% of noise
    noise = 0
    
    # Number of alternatives in the tests set
    nb_tests = 10000
    
    # Number of random trials 
    nb_models = 1
    
    
    # Creation of an instance of test for inferring an MR-Sort model with META-SP algorithm
    inst = RandMRSortMIPSPLearning(nb_alternatives, nb_categories, nb_criteria, dir_criteria, l_unk_pref_dirs, nb_tests, nb_models, noise = noise)
    
    # Running the MIP-SP algorithm to learn model parameters 
    # Export tests results in csv files
    inst.learning_process()


