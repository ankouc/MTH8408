from variationalproblem import VariationalProblem
from ipoptproblem import IpoptProblem
from meshmesh import customMesh
from dolfin import UnitIntervalMesh, DOLFIN_EPS, Constant, inner, grad, Expression, dx, plot
import ipopt
from numpy import *
from newton import newton, newton_optim, armijo

# ---------- The crossing river boat -----------
# Define the function to be integrated
def f(u) : 
    v = Expression("-x[0]*x[0] + x[0]")
    c = 1.0 # boat velocity, considered to be constant
    f = ( ( (c**2.0 * (1.0 + inner(grad(u),grad(u))) - v**2.0)**(0.5) - v*inner(grad(u),grad(u))**0.5 ) / ( c**0.5 - v**0.5) )*dx
    return f


# Set boundary value
target=Constant(1.0)

# Define Dirichlet boundary (u(x) = 0 @ x = 0)
bc='DirichletBC(U, self.target, lambda x : x[0]<DOLFIN_EPS)'

# Generating a refined mesh
x_i = 0.0 # starting point
x_f = 1.0 # end point
N = 10 # number of sub-intervals
sensitivePoints = [0.0, 1.0]
nR = 2
mesh = customMesh(x_i, x_f, N, sensitivePoints, nR)
N = mesh.num_cells()

# Generate the variational problem object
boat = VariationalProblem(mesh, 1, f, None, target,False, bc)

# This section is a required tweak : without it, boat.grad(boat.x0) yields "nan" that prevent the solver to start.
# If the mesh is modified, it might not work anymore.
boat.x0 = ones((N+1), dtype=float_)
for ii in range(1,len(boat.x0),2) :
    boat.x0[ii] = 2.0
#print boat.x0
print boat.grad(boat.x0)


x0 = boat.x0

# No bounds are required on the function (y position of the boat)
lb = None
ub = None 

cl = []
cu = []

# Create solver object
solve_boat = ipopt.problem(
            n=len(x0),
            m=len(cl),
            problem_obj=IpoptProblem(boat),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
            )

solve_boat.addOption(b"mu_strategy", b"monotone")
solve_boat.addOption(b"max_iter", 25000)
solve_boat.addOption(b"tol", 1e-6)

# Call the solver
x, info = solve_boat.solve(x0)

print("Solution of the primal variables: x=%s\n" % repr(x))

print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))

print("Objective=%s\n" % repr(info['obj_val']))


plot(boat.u, interactive=True)
