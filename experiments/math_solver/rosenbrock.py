"""
Parametric Mixed Integer Constrained Rosenbrock Problem
"""

import numpy as np
from pyomo import environ as pe

from experiments.math_solver.abc_solver import abcParamSolver

class rosenbrock(abcParamSolver):
    def __init__(self, steepness, num_blocks, timelimit=None):
        super().__init__(timelimit=timelimit)
        # Create model
        m = pe.ConcreteModel()
        # Mutable parameters
        m.p = pe.Param(default=1, mutable=True)
        m.a = pe.Param(pe.RangeSet(0, num_blocks-1), default=1, mutable=True)
        # Decision variables: x (continuous), y (integer)
        m.x = pe.Var(pe.RangeSet(0, num_blocks-1), domain=pe.Reals)
        m.y = pe.Var(pe.RangeSet(0, num_blocks-1), domain=pe.Integers)
        # Objective function
        obj = sum((m.a[i] - m.x[i]) ** 2 + \
                   steepness * (m.y[i] - m.x[i] ** 2) ** 2 for i in range(num_blocks))
        m.obj = pe.Objective(sense=pe.minimize, expr=obj)
        # Constraints
        m.cons = pe.ConstraintList()
        m.cons.add(sum(m.y[i] for i in range(num_blocks)) >= num_blocks * m.p / 2)
        m.cons.add(sum(m.x[i] ** 2 for i in range(num_blocks)) <= num_blocks * m.p)
        rng = np.random.RandomState(17)
        b = rng.normal(scale=1, size=(num_blocks))
        q = rng.normal(scale=1, size=(num_blocks))
        m.cons.add(sum(b[i] * m.x[i] for i in range(num_blocks)) <= 0)
        m.cons.add(sum(q[i] * m.y[i] for i in range(num_blocks)) <= 0)
        # Set attributes
        self.model = m
        self.params = {"p": m.p, "a": m.a}
        self.vars = {"x": m.x, "y": m.y}
        self.cons = m.cons
