from __future__ import division
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
from pymcda.types import CriterionValue, CriteriaValues
import cplex

verbose = False

class LpMRSortWeights(object):

    def __init__(self, model, pt, aa_ori, delta=0.0001, gamma = 0.5, version_meta = 4, pretreatment_crit = None,
                 fct_w_threshold = 0, w_threshold = 0):
        """ Initializes a LP parameters for the optimization 
        of weights of an MR-Sort model """
        
        self.version_meta = version_meta
        self.model = model
        self.categories = model.categories_profiles.get_ordered_categories()
        self.profiles = model.categories_profiles.get_ordered_profiles()
        self.delta = delta
        self.gamma = gamma
        self.w_threshold = w_threshold
        self.fct_w_threshold = fct_w_threshold
        self.it_meta = 0
        self.pretreatment_crit = pretreatment_crit
        self.cat_ranks = { c: i+1 for i, c in enumerate(self.categories) }
        self.pt = { a.id: a.performances for a in pt }
        self.aa_ori = aa_ori
        self.update_linear_program()


    def update_linear_program(self,fct_w_threshold = 0):
        """ Updates/prepare a linear programming 
        for the optimization of weights of an MR-Sort model """
        
        self.fct_w_threshold = fct_w_threshold
        self.compute_constraints(self.aa_ori, self.model.bpt)

        # Use of cplex solver
        solver_max_threads = int(os.getenv('SOLVER_MAX_THREADS', 0))
        self.lp = cplex.Cplex()
        self.lp.parameters.threads.set(solver_max_threads)
        if verbose is False:
            self.lp.set_log_stream(None)
            self.lp.set_results_stream(None)
        self.add_variables_cplex()
        self.add_constraints_cplex()
        self.add_objective_cplex()
        self.solve_function = self.solve_cplex



    def compute_constraints(self, aa, bpt):
        """ Compute LP constraints regarding profiles 
        and alternatives evaluations """
        
        aa = { a.id: self.cat_ranks[a.category_id] \
               for a in aa }
        bpt = { a.id: a.performances \
                for a in bpt }

        self.c_xi = dict()
        self.c_yi = dict()
        self.a_c_xi = dict()
        self.a_c_yi = dict()
        for a_id in self.pt.keys():
            ap = self.pt[a_id]
            cat_rank = aa[a_id]

            if cat_rank > 1:
                lower_profile = self.profiles[cat_rank-2]
                bp = bpt[lower_profile]

                dj = str()
                for c in self.model.criteria:
                    if ap[c.id] * c.direction >= bp[c.id] * c.direction:
                        dj += '1'
                    else:
                        dj += '0'

                # Deletes old constraints
                if a_id in self.a_c_xi:
                    old = self.a_c_xi[a_id]
                    if self.c_xi[old] == 1:
                        del self.c_xi[old]
                    else:
                        self.c_xi[old] -= 1

                # Saves constraints
                self.a_c_xi[a_id] = dj

                # Adds new constraints
                if not dj in self.c_xi:
                    self.c_xi[dj] = 1
                else:
                    self.c_xi[dj] += 1

            if cat_rank < len(self.categories):
                upper_profile = self.profiles[cat_rank-1]
                bp = bpt[upper_profile]

                dj = str()
                for c in self.model.criteria:
                    if ap[c.id] * c.direction >= bp[c.id] * c.direction:
                        dj += '1'
                    else:
                        dj += '0'

                # Deletes old constraints
                if a_id in self.a_c_yi:
                    old = self.a_c_yi[a_id]
                    if self.c_yi[old] == 1:
                        del self.c_yi[old]
                    else:
                        self.c_yi[old] -= 1

                # Saves constraints
                self.a_c_yi[a_id] = dj

                # Adds new constraints
                if not dj in self.c_yi:
                    self.c_yi[dj] = 1
                else:
                    self.c_yi[dj] += 1

    def add_variables_cplex(self):
        """ Defines evaluation scales of LP variables 
        for the optimization of weights of an MR-Sort model """
        
        self.lp.variables.add(names=['w'+c.id for c in self.model.criteria],
                              lb=[0 for c in self.model.criteria],
                              ub=[1 for c in self.model.criteria])

        self.lp.variables.add(names=['x'+dj for dj in self.c_xi],
                              lb = [0 for dj in self.c_xi],
                              ub = [1 for dj in self.c_xi])
        self.lp.variables.add(names=['y'+dj for dj in self.c_yi],
                              lb = [0 for dj in self.c_yi],
                              ub = [1 for dj in self.c_yi])
        self.lp.variables.add(names=['xp'+dj for dj in self.c_xi],
                              lb = [0 for dj in self.c_xi],
                              ub = [1 for dj in self.c_xi])
        self.lp.variables.add(names=['yp'+dj for dj in self.c_yi],
                              lb = [0 for dj in self.c_yi],
                              ub = [1 for dj in self.c_yi])
        self.lp.variables.add(names=['lambda'], lb = [0.5], ub = [1])



    def add_constraints_cplex(self):
        """ Adds constraints on LP variables 
        for the optimization of weights of an MR-Sort model """
        
        constraints = self.lp.linear_constraints
        w_vars = ['w'+c.id for c in self.model.criteria]
        for dj in self.c_xi:
            coef = list(map(float, list(dj)))

            constraints.add(names=['cinf'+dj],
                            lin_expr =
                                [
                                 [w_vars + ['x'+dj, 'xp'+dj, 'lambda'],
                                  coef + [-1.0, 1.0, -1.0]],
                                ],
                            senses = ["E"],
                            rhs = [0],
                           )

        for dj in self.c_yi:
            coef = list(map(float, list(dj)))
            constraints.add(names=['csup'+dj],
                            lin_expr =
                                [
                                 [w_vars + ['y'+dj, 'yp'+dj, 'lambda'],
                                  coef + [1.0, -1.0, -1.0]],
                                ],
                            senses = ["E"],
                            rhs = [-self.delta],
                           )
        
        # contraint : sum w_j = 1
        constraints.add(names=['wsum'],
                        lin_expr = [[w_vars,
                                    [1.0] * len(w_vars)],
                                   ],
                        senses = ["E"],
                        rhs = [1]
                        )
    
    
    def add_objective_cplex(self):
        """ Construct an objective function for the optimization 
        of weights of an MR-Sort model """
        
        self.lp.objective.set_sense(self.lp.objective.sense.minimize)
        for dj, coef in self.c_xi.items():
            self.lp.objective.set_linear('xp'+dj, coef)
        for dj, coef in self.c_yi.items():
            self.lp.objective.set_linear('yp'+dj, coef)
        


    def solve_cplex(self):
        """ Execute the CPLEX solver to solve the LP problem 
        in order to optimize the weights of an MR-Sort model """

        self.lp.solve()

        status = self.lp.solution.get_status()
        if status != self.lp.solution.status.optimal:
            raise RuntimeError("Solver status: %s" % status)

        obj = self.lp.solution.get_objective_value()

        cvs = CriteriaValues()
        for c in self.model.criteria:
            cv = CriterionValue()
            cv.id = c.id
            cv.value = self.lp.solution.get_values('w'+c.id)
            cvs.append(cv)

        self.model.cv = cvs
        self.model.lbda = self.lp.solution.get_values("lambda")

        return obj

    def solve(self):
        """ Use a solver to solve the LP problem """
        
        return self.solve_function()


class LpMRSortWeightsPositive(LpMRSortWeights):

    def add_objective_cplex(self):
        """ Construct an objective function for the optimization 
        of positive weights of an MR-Sort model """
        
        self.lp.objective.set_sense(self.lp.objective.sense.minimize)
        for dj, coef in self.c_xi.items():
            self.lp.objective.set_linear('xp'+dj, coef)
        for dj, coef in self.c_yi.items():
            self.lp.objective.set_linear('yp'+dj, coef)
        

