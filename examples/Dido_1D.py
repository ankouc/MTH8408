# -*- coding: utf-8 -*-
"""1D bubble problem: minimize length of curve subject to a given area under the curve. 
In this case the area corresponds to a half-circle of diameter 1"""

from variationalproblem import VariationalProblem
from ipoptproblem import IpoptProblem
from dolfin import UnitSquareMesh, UnitIntervalMesh, FunctionSpace, grad, \
                   DirichletBC, Expression, inner, dx, Constant, plot, ds, \
                   Function, DOLFIN_EPS, TestFunction
from nlpmodel import NLPModel
from pdemodel import PDENLPModel
import numpy as np
import ipopt
import scipy.sparse as sps


#boundary conditions
bc='DirichletBC(U, self.target, lambda x, on_boundary: on_boundary)'
# Number of vertices => Number of intervals = N-1
N = 50 
mesh = UnitIntervalMesh(N-1)
#function space degree
deg=1
# declare functional: minimize length of curve (in this case perimeter of upper half-circle)
f='(-u)*dx'
#declare constraint: subject to given area under curve (area of half-circle)
h='( 1+inner(grad(u),grad(u)) )**(0.5)*dx'
target=Constant(0.0)
# functional without penalty function
penalty=False

#instantiate problem
dido = VariationalProblem(mesh, deg, f, h,target,penalty,bc)

#initial values
x0 = dido.x0

#lower boundary of x variable on mesh is 0
lb = np.ones((N), dtype=np.float_)* (0.0)
#no upper boundary
ub = np.ones((N), dtype=np.float_)* (0.5)

# upper and lower constraint set to pi/8, area of half-circle of radius 0.5
cl = [1.5708]
cu = [1.5708]


#instantiate class to call ipopt solver
dido_solver = ipopt.problem(
            n=len(x0),
            m=len(cl),
            problem_obj=IpoptProblem(dido),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
            )

dido_solver.addOption(b"mu_strategy", b"adaptive")
#dido_solver.addOption(b"tol", 1e-10)
dido_solver.addOption(b"max_iter", 50000)

#solve problem
x, info = dido_solver.solve(x0)


print("Solution of the primal variables: x=%s\n" % repr(x))

print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))

print("Objective=%s\n" % repr(info['obj_val']))

plot(dido.u, interactive=True)









