#!/usr/bin/python3
from sympy import *
theta, ux, uy, uz = symbols('theta, u_x, u_y, u_z')
R  = Matrix([
    [cos(theta) + ux*ux*(1-cos(theta)),  ux*uy*(1-cos(theta))-uz*sin(theta), uz*ux*(1-cos(theta))+uy*sin(theta)],
    [ux*uy*(1-cos(theta)) + uz*sin(theta), cos(theta) + uy*uy*(1-cos(theta)),  uz*uy*(1-cos(theta))-ux*sin(theta)],
    [ux*uz*(1-cos(theta)) - uy*sin(theta), uz*uy*(1-cos(theta))+ux*sin(theta), cos(theta) + uz*uz*(1-cos(theta))]
])
inc, Ome, ome = symbols('inc, Ome, ome')
Rx = R.subs(theta, inc).subs(ux, 1).subs(uy, 0).subs(uz, 0)
Rz = R.subs(theta, Ome).subs(ux, 0).subs(uy, 0).subs(uz, 1)
Z  = Rz * Rx * Matrix([[0], [0], [1]])
RZ = R.subs(theta, ome).subs(ux, Z[0]).subs(uy, Z[1]).subs(uz, Z[2])
Ro = trigsimp(RZ*Rz*Rx)
x, y, z = symbols('x, y, z')
r = Matrix([[x], [y], [z]])
