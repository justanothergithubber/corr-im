"""Contains the linear program for the correlation robust influence problem."""
# Python Standard library
from typing import Hashable, Optional, Sequence, Union
# packages
from igraph import Graph
from pyomo.environ import (AbstractModel, ConcreteModel, Constraint, Objective,
                           Set, SolverFactory, Var, minimize, value)

SOLVER = SolverFactory("gurobi")  # modify this to use alternative solvers
# An alternative is CBC, which uses the string "cbc"


def inf_abs_mod(input_graph: Graph) -> AbstractModel:
    """Create the abstract model which is shared among all concrete models."""
    # RESERVED SEED NAMES
    # if 's' in input_graph or 't' in input_graph:
    #     raise ValueError

    # Abstract Model to be shared throughout greedy algorithm
    mod = AbstractModel()

    # Common sets
    mod.V = Set(initialize=[n.index for n in input_graph.vs()])
    mod.E = Set(initialize=[e.tuple for e in input_graph.es()],
                dimen=2)

    # Common variables
    mod.pi = Var(mod.V, bounds=(0, 1))

    return mod


def inf_conc_mod(seed_set: Sequence[Hashable], abs_mod: AbstractModel,
                 input_graph: Graph,
                 solve: Optional[bool] = None) -> Union[float,
                                                        ConcreteModel]:
    """Instantiate a concrete instance of the model."""
    inst = abs_mod.create_instance()

    # Sets
    inst.S = Set(initialize=seed_set)
    inst.VmS = inst.V - inst.S

    # Objective Function
    o_expr = sum(inst.pi[i] for i in inst.VmS)
    inst.obj = Objective(expr=o_expr, sense=minimize)

    # Constraints
    inst.flow_profit = Constraint(inst.E,
                                  rule=(lambda model, i, j:
                                        (model.pi[i] - model.pi[j] <=
                                         input_graph.es[
                                            input_graph.get_eid(i, j)]["q"])
                                        )
                                  )

    inst.seed = Constraint(inst.S, rule=lambda model, i: model.pi[i] == 1)
    if solve:
        SOLVER.solve(inst)
        return value(inst.obj) + len(seed_set)
    # else
    return inst
