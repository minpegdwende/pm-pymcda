from __future__ import division
import os, sys
from itertools import product
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
from pymcda.types import CriterionValue, CriteriaValues
from pymcda.types import AlternativePerformances, PerformanceTable

verbose = False

class MipMRSort():

    def __init__(self, model, pt, aa, epsilon = 0.01, timeout=3600):
        """ Initializes the MIP setting (variables, solver parameters, etc) 
        for the learning of MR-Sort-SP parameters.
        Default value of the timeout = 1h """
        
        self.pt = pt
        self.aa = aa
        self.model = model
        self.criteria = model.criteria.get_active()
        self.cps = model.categories_profiles
        self.epsilon = epsilon
        self.bigm = 2

        self.__profiles = self.cps.get_ordered_profiles()
        self.__categories = self.cps.get_ordered_categories()

        # Transformation of cost criteria to 'minus gain' criteria
        self.pt.update_direction(model.criteria)   
        if self.model.bpt is not None:
            self.model.bpt.update_direction(model.criteria)
            tmp_pt = self.pt.copy()
            for bp in self.model.bpt:
                tmp_pt.append(bp)
            self.ap_min = tmp_pt.get_min()
            self.ap_max = tmp_pt.get_max()
            self.ap_range = tmp_pt.get_range()
        else:
            self.ap_min = self.pt.get_min()
            self.ap_max = self.pt.get_max()
            self.ap_range = self.pt.get_range()
        for c in self.criteria:
            self.ap_min.performances[c.id] -= self.epsilon
            self.ap_max.performances[c.id] += self.epsilon
            self.ap_range.performances[c.id] += 2 * self.epsilon * 100

        # Setting of the solver (CPLEX)     
        import cplex
        solver_max_threads = int(os.getenv('SOLVER_MAX_THREADS', 0))
        self.lp = cplex.Cplex()
        self.lp.parameters.threads.set(solver_max_threads)
        self.lp.parameters.simplex.tolerances.optimality.set(0.000000001)
        self.lp.parameters.simplex.tolerances.feasibility.set(0.000001)
        self.lp.parameters.mip.tolerances.mipgap.set(0.000001)
        self.lp.parameters.clocktype.set(1)
        self.lp.parameters.timelimit.set(timeout)
        
        # Adding variables and constraints to the solver instance 
        # and solves the MIP program
        self.add_variables_cplex()
        self.add_constraints_cplex()
        self.add_extra_constraints_cplex()
        self.add_objective_cplex()
        self.solve_function = self.solve_cplex
        if verbose is False:
            self.lp.set_log_stream(None)
            self.lp.set_results_stream(None)

        # Re-transforming 'minus gain' criteria to cost criteria 
        self.pt.update_direction(model.criteria)
        if self.model.bpt is not None:
            self.model.bpt.update_direction(model.criteria)


    def add_variables_cplex(self):
        """ Adds variables of the MIP instance and their scale """
        
        self.lp.variables.add(names = ["gamma_" + a for a in self.aa.keys()],
                               types = [self.lp.variables.type.binary
                                        for a in self.aa.keys()])
        self.lp.variables.add(names = ["none"], lb = [0], ub = [0])
        self.lp.variables.add(names = ["lambda"], lb = [0], ub = [1])
        self.lp.variables.add(names = ["w_" + c.id for c in self.criteria],
                              lb = [0 for c in self.criteria],
                              ub = [1 for c in self.criteria])
    
        self.lp.variables.add(names = ["b_m_"+ c.id for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              lb = [0 for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              ub = [1 for c in self.criteria if abs(c.direction)==2 or c.direction==0])
        self.lp.variables.add(names = ["g_%s_%s" % (profile, c.id)
                                       for profile in self.__profiles
                                       for c in self.criteria],
                              lb = [self.ap_min.performances[c.id] if abs(c.direction)==1 else self.epsilon
                                    for profile in self.__profiles
                                    for c in self.criteria],
                              ub = [self.ap_max.performances[c.id] + self.epsilon if abs(c.direction)==1 else 0.5
                                    for profile in self.__profiles
                                    for c in self.criteria])
        
        a1 = self.aa.get_alternatives_in_categories(self.__categories[1:])
        a2 = self.aa.get_alternatives_in_categories(self.__categories[:-1])
        self.lp.variables.add(names = ["cinf_%s_%s" % (a, c.id)
                                       for a in a1
                                       for c in self.criteria],
                              lb = [0 for a in a1 for c in self.criteria],
                              ub = [1 for a in a1 for c in self.criteria])
        self.lp.variables.add(names = ["csup_%s_%s" % (a, c.id)
                                       for a in a2
                                       for c in self.criteria],
                              lb = [0 for a in a2 for c in self.criteria],
                              ub = [1 for a in a2 for c in self.criteria])
        self.lp.variables.add(names = ["dinf_%s_%s" % (a, c.id)
                                       for a in a1
                                       for c in self.criteria],
                              types = [self.lp.variables.type.binary
                                       for a in a1
                                       for c in self.criteria])
        self.lp.variables.add(names = ["dsup_%s_%s" % (a, c.id)
                                       for a in a2
                                       for c in self.criteria],
                              types = [self.lp.variables.type.binary
                                       for a in a2
                                       for c in self.criteria]) 
        self.lp.variables.add(names = ["betainf_%s_%s" % (a, c.id)
                                       for a in a1
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              types = [self.lp.variables.type.binary
                                       for a in a1
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0])
        self.lp.variables.add(names = ["betasup_%s_%s" % (a, c.id)
                                       for a in a2
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              types = [self.lp.variables.type.binary
                                       for a in a2
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0])
        self.lp.variables.add(names = ["sigma_%s" % (c.id)
                                       for c in self.criteria if c.direction==0],
                              types = [self.lp.variables.type.binary
                                       for c in self.criteria if c.direction==0])         
        self.lp.variables.add(names = ["alphapinf_%s_%s" % (a, c.id)
                                       for a in a1
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              lb = [0 for a in a1 for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              ub = [1 for a in a1 for c in self.criteria if abs(c.direction)==2 or c.direction==0])
        self.lp.variables.add(names = ["alphapsup_%s_%s" % (a, c.id)
                                       for a in a2
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              lb = [0 for a in a2 for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              ub = [1 for a in a2 for c in self.criteria if abs(c.direction)==2 or c.direction==0])
        self.lp.variables.add(names = ["alphaminf_%s_%s" % (a, c.id)
                                       for a in a1
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              lb = [0 for a in a1 for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              ub = [1 for a in a1 for c in self.criteria if abs(c.direction)==2 or c.direction==0])
        self.lp.variables.add(names = ["alphamsup_%s_%s" % (a, c.id)
                                       for a in a2
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              lb = [0 for a in a2 for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              ub = [1 for a in a2 for c in self.criteria if abs(c.direction)==2 or c.direction==0])



    def __add_lower_constraints_cplex(self, aa):
        """ Adds constraints pertaining to the assignment of alternatives 
        in lower categories """
        
        constraints = self.lp.linear_constraints
        i = self.__categories.index(aa.category_id)
        b = self.__profiles[i - 1]
        bigm = self.bigm
        
        # sum cinf_j(a_i, b_{h-1}) >= lambda + M (alpha_i - 1)
        constraints.add(names = ["gamma_inf_%s" % (aa.id)],
                        lin_expr =
                            [
                              [["cinf_%s_%s" % (aa.id, c.id) for c in self.criteria] + ["lambda"] + ["gamma_%s" % aa.id],
                              [1 for c in self.criteria] + [-1] + [-bigm]],
                            ],
                        senses = ["G"],
                        rhs = [-bigm]
                        )       
        

        for c in self.criteria:
            bigm = self.ap_range.performances[c.id]
            bigm = self.bigm

            # cinf_j(a_i, b_{h-1}) <= w_j
            constraints.add(names = ["c_cinf_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                  [["cinf_%s_%s" % (aa.id, c.id), "w_" + c.id],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                            )
            
            # cinf_j(a_i, b_{h-1}) <= dinf_{i,j}
            constraints.add(names = ["c_cinf2_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                  [["cinf_%s_%s" % (aa.id, c.id), "dinf_%s_%s" % (aa.id, c.id)],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                            )
            
            # cinf_j(a_i, b_{h-1}) >= dinf_{i,j} - 1 + w_j
            constraints.add(names = ["c_cinf3_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                  [["cinf_%s_%s" % (aa.id, c.id), "dinf_%s_%s" % (aa.id, c.id), "w_" + c.id],
                                  [1, -1, -1]],
                                ],
                            senses = ["G"],
                            rhs = [-1]
                            )

            if abs(c.direction) == 1:

                # M dinf_(i,j) >= a_{i,j} - b_{h-1,j} + epsilon
                constraints.add(names = ["d_dinf1_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["dinf_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id)],
                                      [bigm, 1]],
                                    ],
                                senses = ["G"],
                                rhs = [self.pt[aa.id].performances[c.id] + self.epsilon]
                                )
    
                
                # M (dinf_(i,j) - 1) <= a_{i,j} - b_{h-1,j}
                constraints.add(names = ["d_dinf2_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["dinf_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id)],
                                      [bigm, 1]],
                                    ],
                                senses = ["L"],
                                rhs = [self.pt[aa.id].performances[c.id] + bigm]
                                )
                
                
            # Additionnal constraints for single-peaked/single-valley criteria
            # and criteria with unknown preference directions
            if abs(c.direction) == 2 or c.direction == 0:
                
                # alphapinf_(i,j) - alphaminf_(i,j) = b_m - a_{i,j}
                constraints.add(names = ["alphapminf_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id), "b_m_%s" % (c.id)],
                                      [1, -1, -1]],
                                    ],
                                senses = ["E"],
                                rhs = [-self.pt[aa.id].performances[c.id]]
                                )
                
                constraints.add(names = ["alphapminf1_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["alphapinf_%s_%s" % (aa.id, c.id), "betainf_%s_%s" % (aa.id, c.id)],
                                      [1, -bigm]],
                                    ],
                                senses = ["L"],
                                rhs = [0]
                                )
                constraints.add(names = ["alphapminf2_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["alphaminf_%s_%s" % (aa.id, c.id), "betainf_%s_%s" % (aa.id, c.id)],
                                      [1, bigm]],
                                    ],
                                senses = ["L"],
                                rhs = [bigm]
                                )
                
                # bm + b_* <= 1
                constraints.add(names = ["bmbinf1_%s_%s" % (b, c.id)],
                                lin_expr =
                                    [
                                      [["b_m_%s" % (c.id) , "g_%s_%s" % (b, c.id)],
                                      [1, 1]],
                                    ],
                                senses = ["L"],
                                rhs = [1]
                                )
                # bm - b_* >= 0
                constraints.add(names = ["bmbinf2_%s_%s" % (b, c.id)],
                                lin_expr =
                                    [
                                      [["b_m_%s" % (c.id) , "g_%s_%s" % (b, c.id)],
                                      [1, -1]],
                                    ],
                                senses = ["G"],
                                rhs = [0]
                                )
                
                if c.direction==0:
                    ## Constraints pertaining to the absolute value transformation
                    ## M(1 - sigma_i) + M dinf_(i,j) > b_{h-1,j} - alphapinf_{i,j} - alphaminf_{i,j} => SP
                    constraints.add(names = ["d_dinf3_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, -bigm, -1, 1, 1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon - bigm]
                                    )
        
                    ## M(sigma - 1) + M (dinf_(i,j)-1) <= b_{h-1,j} - alphapinf_{i,j} - alphaminf_{i,j} => SP
                    constraints.add(names = ["d_dinf4_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, bigm, -1, 1, 1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm + bigm]
                                    )
                    
                    ## M sigma_i + M dinf_(i,j)  > alphapinf_{i,j} + alphaminf_{i,j} - b_{h-1,j} => SV
                    constraints.add(names = ["d_dinf5_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, bigm, 1, -1, -1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon]
                                    )
        
                    ## -M sigma_i + M (dinf_(i,j)-1) <= alphapinf_{i,j} + alphaminf_{i,j} - b_{h-1,j} => SV
                    constraints.add(names = ["d_dinf6_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, -bigm, 1, -1, -1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm]
                                    )
                
                if c.direction == 2:
                    # M dinf_(i,j) > b_{h-1,j} - alpha_{i,j}
                    constraints.add(names = ["d_dinf13_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, -1, 1, 1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon]
                                    )
        
                    # M (dinf_(i,j)-1) <= b_{h-1,j} - alpha_{i,j}
                    constraints.add(names = ["d_dinf14_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, -1, 1, 1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm]
                                    )
                    
                if c.direction == -2:

                    # M dinf_(i,j) >= alphinf_{i,j} - b_{h-1,j} + epsilon
                    constraints.add(names = ["d_dinf23_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, 1, -1, -1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon]
                                    )

                    # M (dinf_(i,j) - 1) <= alphinf_{i,j} - b_{h-1,j}
                    constraints.add(names = ["d_dinf24_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, 1, -1, -1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm]
                                    )
            
            
    def __add_upper_constraints_cplex(self, aa):
        """ Adds constraints pertaining to the assignment of alternatives 
        in upper categories """
        
        constraints = self.lp.linear_constraints
        i = self.__categories.index(aa.category_id)
        b = self.__profiles[i]
        bigm = self.bigm
        
        # sum csup_j(a_i, b_{h-1}) + epsilon <= lambda - M (gamma_i - 1)
        constraints.add(names = ["gamma_sup_%s" % (aa.id)],
                        lin_expr =
                            [
                              [["csup_%s_%s" % (aa.id, c.id) for c in self.criteria] + ["lambda"] + ["gamma_%s" % aa.id],
                              [1 for c in self.criteria] + [-1] + [bigm]],
                            ],
                        senses = ["L"],
                        rhs = [bigm - self.epsilon]
                        )
        
        for c in self.criteria:
            bigm = self.ap_range.performances[c.id]
            bigm = self.bigm

            # csup_j(a_i, b_h) <= w_j
            constraints.add(names = ["c_csup_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                  [["csup_%s_%s" % (aa.id, c.id), "w_" + c.id],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                            )
            
            # csup_j(a_i, b_h) <= dsup_{i,j}
            constraints.add(names = ["c_csup2_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                  [["csup_%s_%s" % (aa.id, c.id), "dsup_%s_%s" % (aa.id, c.id)],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                            )
            
            # csup_j(a_i, b_{h-1}) >= dsup_{i,j} - 1 + w_j
            constraints.add(names = ["c_csup3_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                  [["csup_%s_%s" % (aa.id, c.id), "dsup_%s_%s" % (aa.id, c.id), "w_" + c.id],
                                  [1, -1, -1]],
                                ],
                            senses = ["G"],
                            rhs = [-1]
                            )
            
            if abs(c.direction) == 1:
                
                # M dsup_(i,j) >= a_{i,j} - b_{h-1,j} + epsilon
                constraints.add(names = ["d_dsup1_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["dsup_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id)],
                                      [bigm, 1]],
                                    ],
                                senses = ["G"],
                                rhs = [self.pt[aa.id].performances[c.id] + self.epsilon]
                                )
    
                # M (dsup_(i,j) - 1) <= a_{i,j} - b_{h-1,j}
                constraints.add(names = ["d_dsup2_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["dsup_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id)],
                                      [bigm, 1]],
                                    ],
                                senses = ["L"],
                                rhs = [self.pt[aa.id].performances[c.id] + bigm]
                                )
            
            
            # Additionnal constraints regarding 
            # single-peaked/single-valley criteria and 
            # criteria with unknown preference directions
            if c.direction == 0 or abs(c.direction)==2:
                                            
                ##   alphapsup_(i,j) - alphaminf_(i,j) = b_m - a_{i,j}
                constraints.add(names = ["alphapmsup_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id), "b_m_%s" % (c.id)],
                                      [1, -1, -1]],
                                    ],
                                senses = ["E"],
                                rhs = [-self.pt[aa.id].performances[c.id]]
                                )
                
                constraints.add(names = ["alphapmsup1_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["alphapsup_%s_%s" % (aa.id, c.id), "betasup_%s_%s" % (aa.id, c.id)],
                                      [1, -bigm]],
                                    ],
                                senses = ["L"],
                                rhs = [0]
                                )
                constraints.add(names = ["alphapmsup2_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["alphamsup_%s_%s" % (aa.id, c.id), "betasup_%s_%s" % (aa.id, c.id)],
                                      [1, bigm]],
                                    ],
                                senses = ["L"],
                                rhs = [bigm]
                                )
    
                #bm + b_* <= 1
                constraints.add(names = ["bmbsup1_%s_%s" % (b, c.id)],
                                lin_expr =
                                    [
                                      [["b_m_%s" % (c.id) , "g_%s_%s" % (b, c.id)],
                                      [1, 1]],
                                    ],
                                senses = ["L"],
                                rhs = [1]
                                )
                
                # bm - b_* >= 0
                constraints.add(names = ["bmbsup2_%s_%s" % (b, c.id)],
                                lin_expr =
                                    [
                                      [["b_m_%s" % (c.id) , "g_%s_%s" % (b, c.id)],
                                      [1, -1]],
                                    ],
                                senses = ["G"],
                                rhs = [0]
                                )   
                
                if c.direction ==0:
                    ## M(1 - sigma_i) + M dsup_(i,j) > b_{h-1,j} - alphapsup_{i,j} - alphamsup_{i,j} =====> SP
                    constraints.add(names = ["d_dsup3_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, -bigm, -1, 1, 1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon - bigm]
                                    )
        
                    ## M(sigma - 1) + M (dsup_(i,j) - 1) <= b_{h-1,j} - alphapsup_{i,j} - alphamsup_{i,j} ====> SP
                    constraints.add(names = ["d_dsup4_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, bigm, -1, 1, 1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm + bigm]
                                    )
                    
                    # M sigma_i + M dsup_(i,j)  > alphapsup_{i,j} + alphamsup_{i,j} - b_{h-1,j} ====> SV
                    constraints.add(names = ["d_dsup5_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, bigm, 1, -1, -1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon]
                                    )
        
                    ## -M sigma_i + M (dsup_(i,j)-1) <= alphapsup_{i,j} + alphamsup_{i,j} - b_{h-1,j} ====> SV
                    constraints.add(names = ["d_dsup6_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, -bigm, 1, -1, -1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm]
                                    )
             
                if c.direction == 2:
                    
                    # M dsup_(i,j) > b_{h-1,j} - alpha_{i,j}
                    constraints.add(names = ["d_dsup13_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, -1, 1, 1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon]
                                    )

                    # M (dsup_(i,j)-1) <= b_{h-1,j} - alpha_{i,j}
                    constraints.add(names = ["d_dsup14_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, -1, 1, 1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm]
                                    )
                
                if c.direction == -2:
                    
                    # M dsup_(i,j) >= alphsup_{i,j} - b_{h-1,j} + epsilon 
                    constraints.add(names = ["d_dsup23_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, 1, -1, -1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon]
                                    )
                    
                    # M (dsup_(i,j) - 1) <= alphsup_{i,j} - b_{h-1,j}
                    constraints.add(names = ["d_dsup24_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, 1, -1, -1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm]
                                    )



    def add_alternatives_constraints(self):
        """ Adds constraints per alternatives """
        
        lower_cat = self.__categories[0]
        upper_cat = self.__categories[-1]
        for aa in self.aa:
            cat = aa.category_id
            if cat != lower_cat:
                self.add_lower_constraints(aa)
            if cat != upper_cat:
                self.add_upper_constraints(aa)


    def add_constraints_cplex(self):
        """ Adds usual constraints pertaining to weights, profiles, lambda """
        
        constraints = self.lp.linear_constraints

        self.add_lower_constraints = self.__add_lower_constraints_cplex
        self.add_upper_constraints = self.__add_upper_constraints_cplex
        self.add_alternatives_constraints()

        profiles = self.cps.get_ordered_profiles()
        for h, c in product(range(len(profiles) - 1), self.criteria):
            # print("dominance")
            constraints.add(names= ["dominance"],
                            lin_expr =
                                [
                                 [["g_%s_%s" % (profiles[h], c.id),
                                   "g_%s_%s" % (profiles[h + 1], c.id)],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                           )

        # sum w_j = 1
        if self.model.cv is None:
            constraints.add(names = ["wsum"],
                            lin_expr =
                                [
                                  [["w_%s" % c.id for c in self.criteria],
                                  [1 for c in self.criteria]],
                                ],
                            senses = ["E"],
                            rhs = [1]
                            )

    def add_extra_constraints_cplex(self):
        """ Adhoc constraints to be added for debugging purposes"""
        
        constraints = self.lp.linear_constraints
                
                
    def add_objective_cplex(self):
        """ Add the objective function of the MIP formulation """
        
        eps = round(0.1/len(self.criteria),3)
        self.lp.objective.set_sense(self.lp.objective.sense.maximize)
        self.lp.objective.set_linear([("gamma_%s" % aid, 1) for aid in self.aa.keys()] +
                                     [("g_%s_%s" % (profile, c.id), eps) for profile in self.__profiles for c in self.criteria if abs(c.direction)==0])


    def solve_cplex(self):
        """ Execute the MIP program """
        
        import cplex
        cplex.Cplex().set_results_stream(None)
        cplex.Cplex().set_log_stream(None)
        start = cplex.Cplex().get_time()
        try:
            self.lp.solve()
        except cplex.exceptions.errors.CplexSolverError("",None,108) :
            pass
            
        end = cplex.Cplex().get_time()
        cplex_time = end-start
        status = self.lp.solution.get_status()
        if status ==108:
            return (None,None,None, None,None,None,None,None, self.lp.solution.get_status(), None)
        
        if status != self.lp.solution.status.MIP_optimal and status != 102 and status != 107 and status != 108:
            raise RuntimeError("Solver status: %s" % status)
        
        # Extraction of variables of the MIP resolved
        obj = self.lp.solution.get_objective_value()
        cvs = CriteriaValues()
        for c in self.criteria:
            cv = CriterionValue()
            cv.id = c.id
            cv.value = self.lp.solution.get_values('w_' + c.id)
            cvs.append(cv)
        self.model.cv = cvs
        self.model.lbda = self.lp.solution.get_values("lambda")
        pt = PerformanceTable()
        pt2 = PerformanceTable()
        for p in self.__profiles:
            ap = AlternativePerformances(p)
            ap2 = AlternativePerformances(p)
            for c in self.criteria:
                perf = self.lp.solution.get_values("g_%s_%s" % (p, c.id))
                ap.performances[c.id] = round(perf, 5)
                ap2.performances[c.id] = round(perf, 5)
                if abs(c.direction)==2 or c.direction==0:
                    bm = self.lp.solution.get_values("b_m_%s" % (c.id))
                    ap.performances[c.id] = bm - round(perf, 5)
                    ap2.performances[c.id] = (bm - round(perf, 5),bm + round(perf, 5))
            pt.append(ap)
            pt2.append(ap2)
        self.model.bpt = pt
        self.model.bpt_sp = pt2
        self.model.bpt.update_direction(self.model.criteria)
        self.model.bpt_sp.update_direction(self.model.criteria)
        tmp3 = [self.lp.solution.get_values("g_%s_%s" % (profile, c.id)) for profile in self.__profiles for c in self.criteria if c.direction==0]
        tmp4 = [(c.id, self.lp.solution.get_values("sigma_%s" % (c.id))) for c in self.criteria if abs(c.direction)==0]
        tmp5 = [self.lp.solution.get_values("sigma_%s" % (c.id)) for c in self.criteria if abs(c.direction)==0]
        tmp6 = [self.lp.solution.get_values("b_m_%s" % (c.id)) for c in self.criteria if c.direction==0]
        tmp7 = [self.lp.solution.get_values("g_%s_%s" % (profile, c.id)) for profile in self.__profiles for c in self.criteria if c.direction==0]

        return (obj,self.lp.solution.MIP.get_mip_relative_gap(),sum([self.lp.solution.get_values("gamma_" + a) for a in self.aa.keys()]), sum(tmp3),tmp4,tmp5,tmp6,tmp7, status, cplex_time)


    def solve(self):
        """ Solves the MIP program """
        return self.solve_function()

