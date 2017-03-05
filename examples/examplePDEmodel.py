from variationalproblem import VariationalProblem
from dolfin import UnitSquareMesh, UnitIntervalMesh, FunctionSpace, grad, \
                   DirichletBC, Expression, inner, dx, Constant, plot, ds, \
                   Function, DOLFIN_EPS, TestFunction
from nlpmodel import NLPModel
from pdemodel import PDENLPModel
from newton import newton, newton_optim, armijo
import numpy as np

N=10
deg=2
mesh = UnitSquareMesh(N, N) 
f='0.5 * inner(u - self.target, u - self.target)**2 * dx'

target=Expression("x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])")

bc='DirichletBC(U, self.target, lambda x, on_boundary: on_boundary)'
penalty=False
h=None

ExamplePDEModel=VariationalProblem(mesh, deg, f, h,target,penalty,bc,u0=Constant(2))


newton_optim(ExamplePDEModel)
plot(ExamplePDEModel.u, interactive=True)
