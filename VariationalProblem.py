# -*- coding: utf-8 -*-

from dolfin import UnitSquareMesh, UnitIntervalMesh, FunctionSpace, grad, \
                   DirichletBC, Expression, inner, dx, Constant, plot, ds, \
                   Function, DOLFIN_EPS, TestFunction
from nlpmodel import NLPModel
from pdemodel import PDENLPModel
import numpy as np


class VariationalProblem(PDENLPModel):
	def __init__(self, mesh, deg, f, h=None,target=Constant(0.0),penalty=False,bc=None, **kwargs) : 
		
		U = FunctionSpace(mesh, "Lagrange", deg)
		self.target=target
		u = Function(U)
		if bc==None:
			bc=[]
		else:
			bc=[eval(bc)]

			
		super(VariationalProblem, self).__init__(mesh, U, bc, **kwargs)
		self.register_objective_functional(f,penalty)
		if h != None:
	       		self.register_constraint_functional(h)
       		

	

    	def register_objective_functional(self,f,penalty):
		u = self.u
		#class accepts functionals declared as strings or functions
		if type(f) is str:
			#penalty=True modifies the functional by adding a penalty function
			if penalty==True:
				beta=1e6
				del_u=self.u-self.target
				self._objective_functional=eval(f)+beta*inner(del_u,del_u)*ds 	 
			else:
    				self._objective_functional = eval(f)
		# functions must take u as input parameter
		else:
			if penalty==True:
				beta=1e6
				del_u=self.u-self.target
				self._objective_functional=f(u)+beta*inner(del_u,del_u)*ds 	 
			else:
    				self._objective_functional = f(u)
	

    	def register_constraint_functional(self,h) :
       		u = self.u
		if type(h) is str:
	       		self._constraint_functional = eval(h)
		else:
			self._constraint_functional = h(u)

	
