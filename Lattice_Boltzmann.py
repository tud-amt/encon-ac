# This module simulates the Nitrogen gas flow within the autoclave and computes the steady-state temperature field
#
# The primary function in the LBM_Simulation class is simulate(). This function returns the temperature variations at the walls of the autoclave
#

import numpy as np
import matplotlib.pyplot, matplotlib.animation, timeit
from matplotlib import cm
from matplotlib import pyplot as plt
import pyximport; pyximport.install()
import subprocess
subprocess.call(["cython","-a","C_Functions.pyx"])
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

class LBM_Simulation:

    def __init__(self, Autoclave, ds, vel, viscosity, density, width, length):
        """
        This function initializes the LBM_Simulation class
        :param Autoclave: the autoclave object, containing its geometry and material properties
        :param ds: (float) - the spatial step of the simulation domain
        :param vel: (float) - the velocity of the flow near the fan(s)
        :param viscosity: (float) - the viscosity of the simulated fluid
        :param density: (float) - the density of the simulated fluid
        :param width: (float) - the width of the domain (meters)
        :param length: (float) - the length of the domain (meters)
        """

        self.Autoclave = Autoclave
        self.niu = viscosity
        self.rho = density
        self.ac = Autoclave
        self.ds = ds
        self.c = 344   # real speed of sound
        self.u0 = vel/self.c    # adimensional velocity (adimensional speed of sound = ds/ds = 1)
        self.v0 = 0  # the y-axis velocity of the fan is 0 m/s
        self.T0 = 273.15 + 20 # the ambient temperature (and initial temperature of the gas)
        self.vel_avg = np.zeros((int(width / ds), int(length / ds)))  # the average velocity field mesh
        self.Temp_avg = np.zeros((int(width / ds), int(length / ds))) # the average temperature field mesh
        self.c_step = 0  # the initial value of the number of simulation steps computed
        self.Length = length # the length of the domain
        self.Width = width  # the width of the domain
        self.setInitialConditions()  # initialize the autoclave microscopic velocity fraction domains in all 9 directions and the solid boundary matrix

    def drawLine(self, Point1, Point2, Matrix, step):

        """
        This function draws a discretized line between 2 points in a boolean matrix (sets the squares crossed by the line to True)
        :param Point1: (array) - the 2D coordinates of the first point
        :param Point2: (array) - the 2D coordinates of the second point
        :param Matrix: (boolean [][]) - the 2D boolean matrix
        :param step: (float) - the discretization line step
        :return: -
        """

        if abs(Point1[0] - Point2[0]) > abs(Point1[1] - Point2[1]): # if the line is closer to a horizontal, rather than a vertical line

            x = np.arange(Point1[0], Point2[0], np.sign(Point2[0] - Point1[0]) * step / 20) # create an x-position array between the 2 points

            for i in range(len(x) - 1):
                y = Point1[1] + (Point2[1] - Point1[1]) / (Point2[0] - Point1[0]) * (x[i] - Point1[0]) # compute the y-coordinate of the line between the points for each element in x
                Matrix[int(y)][int(x[i])] = True  # set the corresponding grid element to true
        else:  # if the line is closer to a vertical, rather than a horizontal line

            y = np.arange(Point1[1], Point2[1], np.sign(Point2[1] - Point1[1]) * step / 20)  # create an y-position array between the 2 points
            for i in range(len(y) - 1):
                x = Point1[0] + (Point2[0] - Point1[0]) / (Point2[1] - Point1[1]) * (y[i] - Point1[1]) # compute the x-coordinate of the line between the points for each element in y
                Matrix[int(y[i])][int(x)] = True  # set the corresponding grid element to true

    def simulate_fan(self, u0):

        """
        This function simulates the fan-induced velocity by imposing Dirichlet boundary conditions on the velocity field at its position
        :param u0: (float) - the imposed x-axis fan velocity
        :return: -
        """

        v0 = 0  # the y-axis fan velocity is set to 0 m/s

        # import the autoclave geometrical parameters required to determine the fan position

        D_Out = self.Autoclave.D_Out
        D_In = self.Autoclave.D_In
        L_Out = self.Autoclave.L_Out
        L_In = self.Autoclave.L_In
        C_Rad = self.Autoclave.C_Rad
        M_dist = self.Autoclave.M_Dist
        M_height = self.Autoclave.M_Height
        M_Dim = self.Autoclave.Mold_Dim
        C_Rad_Length = C_Rad * np.cos(np.arcsin(D_Out / C_Rad / 2))
        theta_0 = np.pi / 2 - np.arcsin(D_Out / 2 / C_Rad)
        X_center = self.Length / 2
        Y_center = self.Width / 2

        ds = self.ds  # import the spatial step of the domain

        x_fan = int((X_center + L_In / 2) / ds)  # the x-position of the vertical fan
        y_fan_st = int((Y_center - D_In / 2) / ds)  # the starting y-position of the fan
        y_fan_end = int((Y_center + D_In / 2) / ds) # the end y-position of the fan

        # setting the 9 microscopic velocity fractions based on the fan macroscopic velocity (u0, v0) using the discretized Boltzmann distribution

        self.fE[y_fan_st:y_fan_end, x_fan] = 1 / 9 * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * (u0 ** 2 + v0 ** 2))
        self.fW[y_fan_st:y_fan_end, x_fan] = 1 / 9 * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * (u0 ** 2 + v0 ** 2))
        self.fNE[y_fan_st:y_fan_end, x_fan] = 1 / 36 * (1 + 3 * (u0 + v0) + 4.5 * (u0 + v0) ** 2 - 1.5 * (u0 ** 2 + v0 ** 2))
        self.fSE[y_fan_st:y_fan_end, x_fan] = 1 / 36 * (1 + 3 * (u0 - v0) + 4.5 * (u0 - v0) ** 2 - 1.5 * (u0 ** 2 + v0 ** 2))
        self.fNW[y_fan_st:y_fan_end, x_fan] = 1 / 36 * (1 + 3 * (-u0 + v0) + 4.5 * (-u0 + v0) ** 2 - 1.5 * (u0 ** 2 + v0 ** 2))
        self.fSW[y_fan_st:y_fan_end, x_fan] = 1 / 36 * (1 + 3 * (-u0 - v0) + 4.5 * (u0 + v0) ** 2 - 1.5 * (u0 ** 2 + v0 ** 2))

    def drawAutoclave(self, Matrix):

        """
        This function draws the autoclave simplified cross-section as wall boundaries for the fluid using the boolean solid matrix
        :param Matrix: ( boolean [][] ) - the solid boundary matrix
        :return: Matrix (boolean [][]) - the updated boundary matrix
        """

        # initialize the solid boolean matrices to store the discretized autoclave walls

        self.Walls_Solid = np.zeros((len(Matrix), len(Matrix[0]))).astype(bool) # the boolean matrix for the bulkheads
        self.Wall_N = np.zeros((len(Matrix), len(Matrix[0]))).astype(bool) # the solid matrix for the northern wall
        self.Wall_S = np.zeros((len(Matrix), len(Matrix[0]))).astype(bool) # the solid amtrix for the southern wall

        # import the autoclave geometrical features into local variables

        D_Out = self.Autoclave.D_Out
        D_In = self.Autoclave.D_In
        L_Out = self.Autoclave.L_Out
        L_In = self.Autoclave.L_In
        C_Rad = self.Autoclave.C_Rad
        M_dist = self.Autoclave.M_Dist
        M_height = self.Autoclave.M_Height
        M_Dim = self.Autoclave.Mold_Dim
        C_Rad_Length = C_Rad * np.cos(np.arcsin(D_Out / C_Rad / 2))
        theta_0 = np.pi / 2 - np.arcsin(D_Out / 2 / C_Rad)

        ds = self.ds  # import the domain spatial step

        X_center = self.Length / 2 # compute the center of the autoclave x-position
        Y_center = self.Width / 2 # compute the center of the autoclave y-position

        # draw the autoclave northern and southern inner and outer walls

        self.drawLine([int((X_center - L_In / 2) / ds), int((Y_center - D_In / 2) / ds)], [int((X_center + L_In / 2) / ds), int((Y_center - D_In / 2) / ds)], Matrix, self.ds)
        self.drawLine([int((X_center - L_In / 2) / ds), int((Y_center + D_In / 2) / ds)], [int((X_center + L_In / 2) / ds), int((Y_center + D_In / 2) / ds)], Matrix, self.ds)
        self.drawLine([int((X_center - L_Out / 2) / ds), int((Y_center - D_Out / 2) / ds)], [int((X_center + L_Out / 2) / ds), int((Y_center - D_Out / 2) / ds)], Matrix, self.ds)
        self.drawLine([int((X_center - L_Out / 2) / ds), int((Y_center + D_Out / 2) / ds)], [int((X_center + L_Out / 2) / ds), int((Y_center + D_Out / 2) / ds)], Matrix, self.ds)

        # draw the autoclave northern and southern outer walls in the horizontal wall boolean matrices

        self.drawLine([int((X_center - L_Out / 2) / ds), int((Y_center - D_Out / 2) / ds)], [int((X_center + L_Out / 2) / ds), int((Y_center - D_Out / 2) / ds)], self.Walls_Solid, self.ds)
        self.drawLine([int((X_center - L_Out / 2) / ds), int((Y_center + D_Out / 2) / ds)], [int((X_center + L_Out / 2) / ds), int((Y_center + D_Out / 2) / ds)], self.Walls_Solid, self.ds)
        self.drawLine([int((X_center - L_Out / 2) / ds), int((Y_center - D_Out / 2) / ds)], [int((X_center + L_Out / 2) / ds), int((Y_center - D_Out / 2) / ds)], self.Wall_S,
                      self.ds)
        self.drawLine([int((X_center - L_Out / 2) / ds), int((Y_center + D_Out / 2) / ds)], [int((X_center + L_Out / 2) / ds), int((Y_center + D_Out / 2) / ds)], self.Wall_N,
                      self.ds)

        dtheta = 0.01 # the bulkhead angular drawing step

        # draw the bulkheads in the domain solid matrix and the wall outer solid matrix

        for i in range(0, int((np.pi - 2 * theta_0) / dtheta) + 1): # loop through one bulkhead

            theta = i * dtheta + theta_0  # compute the current angle
            x1 = X_center + L_Out / 2 - C_Rad_Length + C_Rad * np.sin(theta) # compute the x-position of the current point
            y1 = Y_center + C_Rad * np.cos(theta)  # compute the y-position of the current point
            x2 = X_center + L_Out / 2 - C_Rad_Length + C_Rad * np.sin(theta + dtheta)  # compute the x-position of the next point
            y2 = Y_center + C_Rad * np.cos(theta - dtheta) # compute the y-position of the next point
            self.drawLine([x1 / ds, y1 / ds], [x2 / ds, y2 / ds], Matrix, ds) # draw a line between the points in the simulation domain solid matrix
            self.drawLine([x1 / ds, y1 / ds], [x2 / ds, y2 / ds], self.Walls_Solid, ds) # draw a line between the points in the wall solid domain

        for i in range(0, int((np.pi - 2 * theta_0) / dtheta) + 1): # loop through the other bulkhead

            theta = i * dtheta + theta_0 # compute the current angle
            x1 = X_center - L_Out / 2 + C_Rad_Length - C_Rad * np.sin(theta) # compute the x-position of the current point
            y1 = Y_center + C_Rad * np.cos(theta) # compute the y-position of the current point
            x2 = X_center - L_Out / 2 + C_Rad_Length - C_Rad * np.sin(theta + dtheta) # compute the x-position of the next point
            y2 = Y_center + C_Rad * np.cos(theta - dtheta) # compute the y-position of the next point
            self.drawLine([x1 / ds, y1 / ds], [x2 / ds, y2 / ds], Matrix, ds) # draw a line between the points in the simulation domain solid matrix
            self.drawLine([x1 / ds, y1 / ds], [x2 / ds, y2 / ds], self.Walls_Solid, ds) # draw a line between the points in the wall solid domain

        # draw the mould assembly in the simulation domain solid boolean matrix

        self.drawLine([int((X_center - M_Dim[0] / 2) / ds), int((Y_center - L_In / 2 + M_height) / ds)],
                      [int((X_center + M_Dim[0] / 2) / ds), int((Y_center - L_In / 2 + M_height) / ds)], Matrix, self.ds)
        self.drawLine([int((X_center - M_Dim[0] / 2) / ds), int((Y_center - L_In / 2 + M_height + M_Dim[1]) / ds)],
                      [int((X_center + M_Dim[0] / 2) / ds), int((Y_center - L_In / 2 + M_height + M_Dim[1]) / ds)], Matrix, self.ds)
        self.drawLine([int((X_center - M_Dim[0] / 2) / ds), int((Y_center - L_In / 2 + M_height) / ds)],
                      [int((X_center - M_Dim[0] / 2) / ds), int((Y_center - L_In / 2 + M_height + M_Dim[1]) / ds)], Matrix, self.ds)
        self.drawLine([int((X_center + M_Dim[0] / 2) / ds), int((Y_center - L_In / 2 + M_height) / ds)],
                      [int((X_center + M_Dim[0] / 2) / ds), int((Y_center - L_In / 2 + M_height + M_Dim[1]) / ds)], Matrix, self.ds)

        return Matrix

    def setInitialConditions(self):

        """
        This function initializes all the 9 microscopic velcity fractions, the macrscopic x and y velocities, the temperature field matrix and the solid boolean matrix
        :return: -
        """

        # import the initial flow macroscopic velocities, the spatial step and the domain length and width

        u0 = self.u0
        v0 = 0
        ds = self.ds
        width = self.Width
        length = self.Length

        # initialize the 9 microscopic velocity fractions based on the Boltzmann discretized distribution

        self.omega = 1 / (3 * self.niu + 0.5)  # the relaxation parameter

        # the microscopic velocities

        self.f0 = 4 / 9 * (np.ones((int(width / ds), int(length / ds))) - 1.5 * (
                u0 ** 2 + v0 ** 2))  # particle denSities along 9 directiofS
        self.fN = 1 / 9 * (np.ones((int(width / ds), int(length / ds))) + 3 * v0 + 4.5 * v0 ** 2 - 1.5 * (
                u0 ** 2 + v0 ** 2))
        self.fS = 1 / 9 * (np.ones((int(width / ds), int(length / ds))) + 3 * v0 + 4.5 * v0 ** 2 - 1.5 * (
                u0 ** 2 + v0 ** 2))
        self.fE = 1 / 9 * (np.ones((int(width / ds), int(length / ds))) + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * (
                u0 ** 2 + v0 ** 2))
        self.fW = 1 / 9 * (np.ones((int(width / ds), int(length / ds))) - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * (
                u0 ** 2 + v0 ** 2))
        self.fNE = 1 / 36 * (
                np.ones((int(width / ds), int(length / ds))) + 3 * (u0 + v0) + 4.5 * (u0 + v0) ** 2 - 1.5 * (
                u0 ** 2 + v0 ** 2))
        self.fSE = 1 / 36 * (
                np.ones((int(width / ds), int(length / ds))) + 3 * (u0 - v0) + 4.5 * (u0 - v0) ** 2 - 1.5 * (
                u0 ** 2 + v0 ** 2))
        self.fNW = 1 / 36 * (
                np.ones((int(width / ds), int(length / ds))) + 3 * (-u0 + v0) + 4.5 * (-u0 + v0) ** 2 - 1.5 * (
                u0 ** 2 + v0 ** 2))
        self.fSW = 1 / 36 * (
                np.ones((int(width / ds), int(length / ds))) + 3 * (-u0 - v0) + 4.5 * (u0 + v0) ** 2 - 1.5 * (
                u0 ** 2 + v0 ** 2))
        self.F = np.ones((int(width / ds), int(length / ds), 9))
        for i in range(len(self.F)):
            for j in range(len(self.F[0])):
                self.F[i][j] = np.array([self.f0[i][j], self.fE[i][j], self.fW[i][j], self.fN[i][j], self.fS[i][j], self.fNE[i][j], self.fSW[i][j], self.fNW[i][j], self.fSE[i][j]])



        # initializing the macroscopic fields using the microscopic velocities

        self.rho = self.f0 + self.fN + self.fS + self.fE + self.fW + self.fNE + self.fSE + self.fNW + self.fSW  # the macroscopic density
        self.ux = (self.fE + self.fNE + self.fSE - self.fW - self.fNW - self.fSW) / self.rho  # the macroscopic x velocity
        self.uy = (self.fN + self.fNE + self.fNW - self.fS - self.fSE - self.fSW) / self.rho  # the macroscopic y velocity
        self.ux_avg = self.ux
        self.uy_avg = self.uy
        self.Temp = np.ones((int(width / ds), int(length / ds))) * self.T0 # the initial temperature field
        self.Solid = np.zeros((int(width / ds), int(length / ds)), bool) # the initial solid boolean matrix

        self.Solid = self.drawAutoclave(self.Solid) # draw the autoclave shape in the solid matrix

        self.SolidN = np.roll(self.Solid, 1, axis=0)  # sites just north of Solid
        self.SolidS = np.roll(self.Solid, -1, axis=0)  # sites just south of Solid
        self.SolidE = np.roll(self.Solid, 1, axis=1)  # etc.
        self.SolidW = np.roll(self.Solid, -1, axis=1)
        self.SolidNE = np.roll(self.SolidN, 1, axis=1)
        self.SolidNW = np.roll(self.SolidN, -1, axis=1)
        self.SolidSE = np.roll(self.SolidS, 1, axis=1)
        self.SolidSW = np.roll(self.SolidS, -1, axis=1)

    def stream(self):

        """
        This function performs the streaming step of the LBM method.
        :return:
        """

        # import the solid matrix and its shifted counterparts into local variables

        Solid = self.Solid
        SolidN = self.SolidN
        SolidNW = self.SolidNW
        SolidW = self.SolidW
        SolidSW = self.SolidSW
        SolidS = self.SolidS
        SolidSE = self.SolidSE
        SolidE = self.SolidE
        SolidNE = self.SolidNE

        # stream each microscopic velocity fraction one cell forward according to its orientation

        self.fN = np.roll(self.fN, 1, axis=0)  # axis 0 is north-south; + direction is north
        self.fNE = np.roll(self.fNE, 1, axis=0)
        self.fNW = np.roll(self.fNW, 1, axis=0)
        self.fS = np.roll(self.fS, -1, axis=0)
        self.fSE = np.roll(self.fSE, -1, axis=0)
        self.fSW = np.roll(self.fSW, -1, axis=0)
        self.fE = np.roll(self.fE, 1, axis=1)  # axis 1 is east-west; + direction is east
        self.fNE = np.roll(self.fNE, 1, axis=1)
        self.fSE = np.roll(self.fSE, 1, axis=1)
        self.fW = np.roll(self.fW, -1, axis=1)
        self.fNW = np.roll(self.fNW, -1, axis=1)
        self.fSW = np.roll(self.fSW, -1, axis=1)

        # Use tricky boolean arrays to handle solid collisions of flow particles (bounce-back):

        self.fN[SolidN] = self.fS[Solid]
        self.fS[SolidS] = self.fN[Solid]
        self.fE[SolidE] = self.fW[Solid]
        self.fW[SolidW] = self.fE[Solid]
        self.fNE[SolidNE] = self.fSW[Solid]
        self.fNW[SolidNW] = self.fSE[Solid]
        self.fSE[SolidSE] = self.fNW[Solid]
        self.fSW[SolidSW] = self.fNE[Solid]

    def collide(self):

        """
        This function performs the collision step in the MRT Lattice Boltzmann method, using equal relaxation times.
        :return:
        """
        # compute the macroscopic properties

        omega = self.omega # the relaxation constant
        self.rho = self.f0 + self.fN + self.fS + self.fE + self.fW + self.fNE + self.fSE + self.fNW + self.fSW # the macroscopic density
        self.ux = (self.fE + self.fNE + self.fSE - self.fW - self.fNW - self.fSW) / self.rho # the macroscopic x-velocity
        self.uy = (self.fN + self.fNE + self.fNW - self.fS - self.fSE - self.fSW) / self.rho # the macroscopic y-velocity
        ux = self.ux
        uy = self.uy
        u0 = self.u0
        v0 = self.v0
        rho = self.rho
        ux2 = ux * ux  # pre-compute terms used repeatedly...
        uy2 = uy * uy
        u2 = ux2 + uy2
        omu215 = 1 - 1.5 * u2  # "one minus u2 times 1.5"
        uxuy = ux * uy

        # update the microscopic velocitiy fractions using the collision equation

        self.f0 = (1 - omega) * self.f0 + omega * 4 / 9 * rho * omu215
        self.fN = (1 - omega) * self.fN + omega * 1 / 9 * rho * (omu215 + 3 * uy + 4.5 * uy2)
        self.fS = (1 - omega) * self.fS + omega * 1 / 9 * rho * (omu215 - 3 * uy + 4.5 * uy2)
        self.fE = (1 - omega) * self.fE + omega * 1 / 9 * rho * (omu215 + 3 * ux + 4.5 * ux2)
        self.fW = (1 - omega) * self.fW + omega * 1 / 9 * rho * (omu215 - 3 * ux + 4.5 * ux2)
        self.fNE = (1 - omega) * self.fNE + omega * 1 / 36 * rho * (omu215 + 3 * (ux + uy) + 4.5 * (u2 + 2 * uxuy))
        self.fNW = (1 - omega) * self.fNW + omega * 1 / 36 * rho * (omu215 + 3 * (-ux + uy) + 4.5 * (u2 - 2 * uxuy))
        self.fSE = (1 - omega) * self.fSE + omega * 1 / 36 * rho * (omu215 + 3 * (ux - uy) + 4.5 * (u2 - 2 * uxuy))
        self.fSW = (1 - omega) * self.fSW + omega * 1 / 36 * rho * (omu215 + 3 * (-ux - uy) + 4.5 * (u2 + 2 * uxuy))

        # Force steady rightward flow at the fan location
        self.simulate_fan(u0)

    def collideMRT(self):

        """
        This function performs the collision step of the MRT Lattice Boltzmann method using different relaxation times
        :return: -
        """

        # import the fan velocities and microscopic velocity fractions into local variables

        u0 = self.u0
        v0 = self.v0
        F0 = self.f0
        F1 = self.fE
        F2 = self.fW
        F3 = self.fN
        F4 = self.fS
        F5 = self.fNE
        F6 = self.fSW
        F7 = self.fNW
        F8 = self.fSE
        omega = self.omega  # the relaxation parameter
        rho = F0 + F3 + F4 + F1 + F2 + F5 + F8 + F7 + F6 # compute the macroscopic flow density
        ux = (F1 + F5 + F8 - F2 - F7 - F6) / rho # compute the macroscopic x-velocity
        uy = (F3 + F5 + F7 - F4 - F8 - F6) / rho # compute the macroscopic y-velocity
        self.ux = ux # update the macroscopic x-velocity
        self.uy = uy # update the macroscopic y-velocity

        # initialize the M matrix and its inverse

        M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                      [4, -2, -2, -2, -2, 1, 1, 1, 1],
                      [0, 1, -1, 0, 0, 1, -1, -1, 1],
                      [0, -2, 2, 0, 0, 1, -1, -1, 1],
                      [0, 0, 0, 1, -1, 1, -1, 1, -1],
                      [0, 0, 0, -2, 2, 1, -1, 1, -1],
                      [0, 1, 1, -1, -1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 1, -1, -1]])
        M_inv = np.linalg.inv(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                                        [4, -2, -2, -2, -2, 1, 1, 1, 1],
                                        [0, 1, -1, 0, 0, 1, -1, -1, 1],
                                        [0, -2, 2, 0, 0, 1, -1, -1, 1],
                                        [0, 0, 0, 1, -1, 1, -1, 1, -1],
                                        [0, 0, 0, -2, 2, 1, -1, 1, -1],
                                        [0, 1, 1, -1, -1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 1, 1, -1, -1]]))

        R = np.dot(M, self.F)
        e_eq = -2 * rho + 3 * (ux ** 2 + uy ** 2)
        eps_eq = rho - 3 * (ux ** 2 + uy ** 2)
        p_xx_eq = ux ** 2 - uy ** 2
        p_xy_eq = ux * uy
        q_x_eq = -ux
        q_y_eq = -uy
        S = np.diag([0, 1.1, 1.3, 0, 1.8, 0, 1.8, omega, omega])
        # construct the R_eq matrix

        R_eq = np.array([rho, e_eq, eps_eq, 0, q_x_eq, 0, q_y_eq, p_xx_eq, p_xy_eq])

        # compute the new microscopic velocity fraction array

        self.F = self.F - np.dot(np.dot(M_inv, S), (R - R_eq))
        self.f0 = self.F[:, :, 0]
        self.fE = self.F[:, :, 1]
        self.fW = self.F[:, :, 2]
        self.fN = self.F[:, :, 3]
        self.fS = self.F[:, :, 4]
        self.fNE = self.F[:, :, 5]
        self.fSW = self.F[:, :, 6]
        self.fNW = self.F[:, :, 7]
        self.fSE = self.F[:, :, 8]
        self.rho = rho

        # for i in range(len(F0)):
        #     for j in range(len(F0[0])):  # loop through each cell in the domain
        #
        #         # take the microscopic velocity fractions of every cell and add them to an array
        #
        #         f0 = F0[i][j]
        #         f1 = F1[i][j]
        #         f2 = F2[i][j]
        #         f3 = F3[i][j]
        #         f4 = F4[i][j]
        #         f5 = F5[i][j]
        #         f6 = F6[i][j]
        #         f7 = F7[i][j]
        #         f8 = F8[i][j]
        #         F = np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8])
        #
        #         R = np.dot(M, F) # compute the R matrix
        #         rho[i][j] = sum(F) # compute and append the density of the cell
        #         u = ux[i][j] # get the x-axis velocity
        #         v = uy[i][j] # get the y-axis velocity
        #
        #         # compute the elements of the R_eq matrix
        #
        #         e_eq = -2 * rho[i][j] + 3 * (u ** 2 + v ** 2)
        #         eps_eq = rho[i][j] - 3 * (u ** 2 + v ** 2)
        #         p_xx_eq = u ** 2 - v ** 2
        #         p_xy_eq = u * v
        #         q_x_eq = -u
        #         q_y_eq = -v
        #
        #         # initialize the S matrix, containing the relaxation parameters
        #
        #         S = np.diag([0, 1.1, 1.3, 0, 1.8, 0, 1.8, omega, omega])
        #
        #         # construct the R_eq matrix
        #
        #         R_eq = np.array([rho[i][j], e_eq, eps_eq, 0, q_x_eq, 0, q_y_eq, p_xx_eq, p_xy_eq])
        #
        #         # compute the new microscopic velocity fraction array
        #
        #         F_1 = F - np.dot(np.dot(M_inv, S), (R - R_eq))
        #
        #         # the cell microscopic velocities are updated
        #
        #         F0[i][j] = F_1[0]
        #         F1[i][j] = F_1[1]
        #         F2[i][j] = F_1[2]
        #         F3[i][j] = F_1[3]
        #         F4[i][j] = F_1[4]
        #         F5[i][j] = F_1[5]
        #         F6[i][j] = F_1[6]
        #         F7[i][j] = F_1[7]
        #         F8[i][j] = F_1[8]
        #
        # # the class objects are updated with the computed velocity fields and density
        # self.f0 = F0
        # self.fE = F1
        # self.fW = F2
        # self.fN = F3
        # self.fS = F4
        # self.fNE = F5
        # self.fSW = F6
        # self.fNW = F7
        # self.fSE = F8
        # self.rho = rho

        # force boundary conditions at the external domain boundaries

        self.fE[:, 0] = 1 / 9 * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * (u0 ** 2 + v0 ** 2))
        self.fW[:, 0] = 1 / 9 * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * (u0 ** 2 + v0 ** 2))
        self.fNE[:, 0] = 1 / 36 * (1 + 3 * (u0 + v0) + 4.5 * (u0 + v0) ** 2 - 1.5 * (u0 ** 2 + v0 ** 2))
        self.fSE[:, 0] = 1 / 36 * (1 + 3 * (u0 - v0) + 4.5 * (u0 - v0) ** 2 - 1.5 * (u0 ** 2 + v0 ** 2))
        self.fNW[:, 0] = 1 / 36 * (1 + 3 * (-u0 + v0) + 4.5 * (-u0 + v0) ** 2 - 1.5 * (u0 ** 2 + v0 ** 2))
        self.fSW[:, 0] = 1 / 36 * (1 + 3 * (-u0 - v0) + 4.5 * (u0 + v0) ** 2 - 1.5 * (u0 ** 2 + v0 ** 2))

        # Force steady rightward flow at the fan location (no need to set 0, N, and S components):
        self.simulate_fan(u0)

    def simulate_heater(self, T):

        """
        This function simulates the heater by enforcing the heater temperature through Dirichlet condition at the heating elements locations
        :param T: (float) - the heater temperature
        :return: -
        """
        # import the required autoclave geometrical features to define the heating elements
        D_In = self.Autoclave.D_In
        L_In = self.Autoclave.L_In
        X_center = self.Length / 2
        Y_center = self.Width / 2

        ds = self.ds # import the flow domain spatial step

        # define the heating element start and end points

        x_heater_st = int((X_center - L_In * 0.8 / 2) / ds)
        x_heater_end = int((X_center + L_In * 0.8 / 2) / ds)
        y_heater_1 = int((Y_center + D_In / 2) / ds + 2)
        y_heater_2 = int((Y_center - D_In / 2) / ds - 2)

        # apply the heater temperature in the temperature field matrix at the heating element locations

        self.Temp[y_heater_1:y_heater_1 + 2, x_heater_st:x_heater_end] = T
        self.Temp[y_heater_2-2:y_heater_2, x_heater_st:x_heater_end] = T

    def simulate_wall_loss(self, T0, Cp, rho, k, thickness, dt):

        """
        This function simulates the heat loss through the autoclave loss based on the wall properties
        :param T0: (float) - the ambient temperature
        :param Cp: (float) - the specific heat of the wall
        :param rho: (float) - the density of the wall
        :param k: (float) - the conductivity of the wall
        :param thickness: (float) - the thickness of the wall
        :param dt: (float) - the time step of the simulation
        :return: -
        """

        # import the wall matrices into local arrays and shift them to compute the neighbouring cells

        Wall = np.array(self.Walls_Solid).astype(int)
        WallN = np.roll(Wall, 1, axis=0)
        WallS = np.roll(Wall, -1, axis=0)
        WallE = np.roll(Wall, 1, axis=1)
        WallW = np.roll(Wall, -1, axis=1)

        Boundary = WallN + WallS + WallE + WallW - Wall # create the boundary matrix from the boolean wall matrices
        Boundary = np.array(Boundary).astype(bool)  # set the type to boolean
        self.Temp[Boundary] = self.Temp[Boundary] + dt*k/rho/Cp/thickness*(T0 - self.Temp[Boundary]) # change the temperature of the gas near the wall accordingly

    def conduction(self, Temp, k, Cp, rho, ds, dt):

        """
        This function computes the conduction of the gas used in the NV energy equation
        :param Temp: (float [][]) - the temperature field matrix
        :param k: (float) - the conductivity of the gas flow
        :param Cp: (float) - the specific heat of the gas flow
        :param rho: (float) - the density of the gas flow
        :param ds: (float) - the spatial step of the flow domain
        :param dt: (float) - the time step of the flow domain
        :return: conduction (float [][]) - the matrix containing the conduction contribution field
        """

        # compute the neighbouring cells at each temperature field element

        Temp_N = np.roll(Temp, 1, axis=0)
        Temp_S = np.roll(Temp, -1, axis=0)
        Temp_E = np.roll(Temp, 1, axis=1)
        Temp_W = np.roll(Temp, -1, axis=1)

        # compute the neighbouring cells of each solid boundary element

        Solid = self.Solid
        SolidN = self.SolidN
        SolidW = self.SolidW
        SolidS = self.SolidS
        SolidE = self.SolidE

        # handle boundary conditions at solid surfaces

        Temp_N[SolidN] = Temp_S[Solid]
        Temp_S[SolidS] = Temp_N[Solid]
        Temp_E[SolidE] = Temp_W[Solid]
        Temp_W[SolidW] = Temp_E[Solid]

        # compute the conduction contribution to the gas flow for the entire domain

        Conduction = k / rho / Cp * ((Temp_W - 2 * Temp + Temp_E) / ds ** 2 + (Temp_N - 2 * Temp + Temp_S) / ds ** 2)


        return Conduction

    def convection(self, Temp, u, v, ds):

        """
        This function computes the convection contribution to the gas flow domain
        :param Temp: (float [][]) - the gas flow temperature field matrix
        :param u: (float [][]) - the x-axis velocity field matrix
        :param v: (float [][]) - the y-axis velocity field matrix
        :param ds: (flaot) - the spatial step
        :return: Convection (flaot [][]) - the convection contribution to the NV energy equation matrix
        """

        # compute the neighbouring temperatures for each element in the temp. field

        Temp_N = np.roll(Temp, 1, axis=0)
        Temp_S = np.roll(Temp, -1, axis=0)
        Temp_E = np.roll(Temp, 1, axis=1)
        Temp_W = np.roll(Temp, -1, axis=1)

        # compute the convection matrix of the gas flow field

        Convection = (u * (Temp_W - Temp_E) / 2 / ds - v * (Temp_N - Temp_S) / 2 / ds)

        return Convection

    def convect(self):

        """
        This function time-marches the NV energy equation and with it, the temperature field of the autoclave gas
        :return: -
        """

        # initialize the Nitrogen gas properties and simulation spatial and time steps

        rho = self.rho
        k = 0.2
        Cp = 1004
        ds = self.ds
        dt = ds

        def f(T): # add the conduction and convection contributions to the equation
            return self.conduction(T, k, Cp, rho, ds, dt) - self.convection(T, self.ux, self.uy, ds)

        # perform a 4th order Runge Kutta time marching

        k1 = dt * f(self.Temp)
        k2 = dt * f(self.Temp + k1 / 2)
        k3 = dt * f(self.Temp + k2 / 2)
        k4 = dt * f(self.Temp + k3)

        # update the temperature field accordingly

        self.Temp = self.Temp + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # simulate the loss of heat at the walls of the autoclave

        self.simulate_wall_loss(T0 = 288.15, Cp = Cp, rho = 1.225, dt = dt, thickness=0.4, k = self.Autoclave.k)

        # simulate the heating elements boundary conditions

        self.simulate_heater(200 + 273.15)

    def compute_wall_temp(self):

        """
        This function computes, plots amd saves the temperature distributions at each outer wall and mould assembly boundary
        :return: Temp_N, Temp_S, Temp_E, Temp_W  - the outer walls temperature variations
                 Temp_MN, Temp_MS, Temp_ME, Temp_MW - the temperature variations near the mould assembly
        """

        # import the main autoclave geometrical properties into local variables

        D_Out = self.Autoclave.D_Out
        D_In = self.Autoclave.D_In
        L_Out = self.Autoclave.L_Out
        L_In = self.Autoclave.L_In
        C_Rad = self.Autoclave.C_Rad
        M_dist = self.Autoclave.M_Dist
        M_height = self.Autoclave.M_Height
        M_Dim = self.Autoclave.Mold_Dim
        C_Rad_Length = C_Rad * np.cos(np.arcsin(D_Out / C_Rad / 2))
        theta_0 = np.pi / 2 - np.arcsin(D_Out / 2 / C_Rad)

        # import the spatial domain step and set an angular step for the bulckheads sampling

        ds = self.ds
        dtheta = 0.001

        # compute the center point of the autoclave

        X_center = self.Length / 2
        Y_center = self.Width / 2

        # compute the reference points of each horizontal wall

        X1_N = int((X_center - L_Out/2)/ds)
        X2_N = int((X_center + L_Out/2)/ds)
        y_N = int((Y_center + D_Out/2)/ds)
        y_S = int((Y_center - D_Out / 2) / ds)
        X_E = int((X_center + L_In/2)/ds)

        # initialize and compute the bulkhead temperature arrays and the horizontal wall temperature arrays

        Temp_N = self.Temp[ y_N - 1, X1_N:X2_N]
        Temp_S = self.Temp[ y_S + 1, X1_N:X2_N]
        Temp_E = []
        Temp_W = []

        for i in range(0, int((np.pi - 2 * theta_0) / dtheta) + 1): # loop through the first bulkhead

            theta = i * dtheta + theta_0  # compute the current angle
            x1 = int((X_center - L_Out / 2 + C_Rad_Length - (C_Rad -2*ds)* np.sin(theta))/ds) # compute the x- position
            y1 = int((Y_center + (C_Rad -2*ds) * np.cos(theta))/ds)  # compute the y-position
            Temp_W.append(self.Temp[y1, x1])  # append the temperature at this position to the array

        for i in range(0, int((np.pi - 2 * theta_0) / dtheta) + 1): # loop through the second bulkhead

            theta = i * dtheta + theta_0  # compute the current angle
            x1 = int((X_center + L_Out / 2 - C_Rad_Length + (C_Rad -2*ds)* np.sin(theta))/ds) # compute the x- position
            y1 = int((Y_center + (C_Rad -2*ds) * np.cos(theta))/ds) # compute the y-position
            Temp_E.append(self.Temp[y1, x1]) # append the temperature at this position to the array

        # compute the reference points of the mould assembly in the autoclave

        XW = int((X_center - M_Dim[0] / 2) / ds)
        YS = int((Y_center - L_In / 2 + M_height) / ds)
        XE = int((X_center + M_Dim[0] / 2) / ds)
        YN = int((Y_center - L_In / 2 + M_height + M_Dim[1]) / ds)

        # compute the temperature variations in the vicinity of the mould assembly

        Temp_MN = self.Temp[YN + 1, XW:XE]
        Temp_MS = self.Temp[YS - 1, XW:XE]
        Temp_ME = self.Temp[YS:YN, XE + 1]
        Temp_MW = self.Temp[YS:YN, XW - 1]

        return Temp_N, Temp_S, np.array(Temp_E), np.array(Temp_W), Temp_MN, Temp_MS, Temp_ME, Temp_MN, np.average(Temp_MW)

    def compute_average_temp(self):

        """
        This function computes the average temperature of the gas flow inside the autoclave
        :return: (float) - the average temperature
        """

        D_In = self.Autoclave.D_In  # import the inner diameter of the autoclave
        L_In = self.Autoclave.L_In  # import the inner length of the autoclave

        # compute the center point of the autoclave

        X_center = self.Length / 2
        Y_center = self.Width / 2

        # compute and return the average temperature

        return np.average(self.Temp[int((Y_center - D_In/2)/self.ds):int((Y_center + D_In/2)/self.ds), int((X_center - L_In/2)/self.ds):int((X_center + L_In/2)/self.ds)])

    def curl(self):

        """
        This function computes the vorticity of the velocity field of the gas flow
        :return: (float [][]) - the vorticity field
        """

        ux = self.ux  # the x-velocity field
        uy = self.uy  # the y-velocity field

        # compute and return the vorticity with a central difference scheme

        return np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1) - np.roll(ux, -1, axis=0) + np.roll(ux, 1, axis=0)

    def vel(self):

        """
        This function returns the absolute value of the velocity field of the gas flow
        :return: (float [][]) - the absolute velocity matrix
        """

        ux = self.ux  # the x-velocity field
        uy = self.uy  # the y-velocity field
        return (ux ** 2 + uy ** 2) ** 0.5 * self.c  # the absolute value of the velocity

    def velx(self):

        """
        This function returns the x-velocity field
        :return:
        """

        return self.ux

    def nextFrame(self):

        """
        This function time-marches each frame (and subframes)
        :return: -
        """

        for step in range(int(100)):  # for every subframe in each frame

            self.stream()  # perform the streaming step of the MRT LBM method
            self.collide() # perform the simplified collision step of the MRT LBM method (much faster but less stable)
            # self.collideMRT() # perform the advanced collision step of the MRT LBM method (more stable but extremely slow)
            self.convect() # time-march the temperature field of the Nitrogen gas based on the new velocity fields
            self.vel_avg = (self.vel_avg * self.c_step + self.vel()) / (self.c_step + 1)  # compute the average velocity of the flow field
            self.ux_avg = (self.ux_avg * self.c_step + self.ux) / (self.c_step + 1)  # compute the average x-velocity of the flow field
            self.uy_avg = (self.uy_avg * self.c_step + self.uy) / (self.c_step + 1)  # compute the average y-velocity of the flow field
            self.Temp_avg = (self.Temp_avg * self.c_step + self.Temp) / (self.c_step + 1)  # compute the average temperature matrix
            self.c_step += 1  # increment the subframe step

        self.fluidImage = self.ax.imshow(self.Temp, origin='lower',
                                                   #norm=matplotlib.pyplot.Normalize(0, 400),
                                                   cmap=matplotlib.pyplot.get_cmap('viridis'), interpolation='none', extent=[0, len(self.Temp[0])*self.ds, 0, len(self.Temp)*self.ds])
        bImageArray = np.zeros((int(self.Width / self.ds), int(self.Length / self.ds), 4), np.uint8)
        bImageArray[self.Solid, 3] = 255
        self.ax.imshow(bImageArray, origin='lower', interpolation='none', extent=[0, len(self.Temp[0]) * self.ds, 0, len(self.Temp) * self.ds])

        x_c, y_c = np.meshgrid(np.arange(0, len(self.ux[0]) * self.ds, self.ds),
                               np.arange(0, len(self.ux) * self.ds, self.ds))
        #plt.quiver(x_c[::6, ::6], y_c[::6, ::6], self.ux[::6, ::6]/self.vel_avg[::6, ::6], self.uy[::6, ::6]/self.vel_avg[::6, ::6], color='white', linewidths=0.5)
        self.ax.streamplot(x_c[::6, ::6], y_c[::6, ::6], self.ux_avg[::6, ::6]/self.vel_avg[::6, ::6], self.uy_avg[::6, ::6]/self.vel_avg[::6, ::6], color='black', linewidth=0.5, arrowsize=0.5, density=1.5)

    def simulate(self):

        """
        This function performs the simulation for 400 frames and plots and saves each frame
        :return: (float []) - the wall temperature variations of the last frame in the simulation
        """

        self.startTime = timeit.default_timer() # remember the time at the start of the simulation

        # print and save the first frame of the simulation with the initial temperature field of the autoclave

        self.frameList = open('LBM/frameList.txt', 'w')
        self.p_avg = np.average(self.rho) * 287 * 288.15
        self.theFig, self.ax = matplotlib.pyplot.subplots(figsize=(8, 3))

        self.fluidImage = self.ax.imshow(self.Temp, origin='lower',
                                                   norm=matplotlib.pyplot.Normalize(0, 20),
                                                   cmap=matplotlib.pyplot.get_cmap('viridis'), interpolation='none', extent=[0, len(self.Temp[0])*self.ds, 0, len(self.Temp)*self.ds])
        self.theFig.colorbar(cm.ScalarMappable(norm=plt.Normalize(0, np.max(self.Temp)), cmap=plt.get_cmap('viridis')), label=r"Temperature [$^{\circ}$ C]")
        self.theFig.suptitle('Autoclave Steady-state Temperature Field')
        plt.xlabel('X-axis [m]')
        plt.ylabel('Y-axis [m]')
        bImageArray = np.zeros((int(self.Width / self.ds), int(self.Length / self.ds), 4), np.uint8)
        bImageArray[self.Solid, 3] = 255
        self.barrierImage = self.ax.imshow(bImageArray, origin='lower', interpolation='none', extent=[0, len(self.Temp[0])*self.ds, 0, len(self.Temp)*self.ds])

        # set the initial simulation parameters and initialize the average temperature log array

        difference = 500 # the initial difference between the current frame and the last
        frame = 0 # the initial frame
        Temp_avg = []
        time = []

        while frame < 400:  # loop through each frame

            print("Frame:", frame, "Error:", difference)  # print the current frame and the error difference

            # compute the average temperature and append it to the temperature log

            T_array = self.Temp
            T_avg = self.compute_average_temp()
            time.append(frame*self.ds)
            Temp_avg.append(T_avg)

            # time march the flow fields to the next frame

            self.nextFrame()


            difference = np.average(np.abs(self.Temp - T_array)) # compute the new difference between frames

            plt.savefig("LBM/" + str(frame), dpi=1500) # save the new autoclave colormap frame
            self.ax.clear()


            # plot and save the updated average temperature log

            fig = plt.figure()
            ax = fig.add_subplot()
            plt.suptitle("Temperature Log")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(r"Temperature [K]")
            ax.plot(np.array(time)*300 / self.c, np.array(Temp_avg) - 273.15, linestyle='-', color='black',
                    label="Temperature of Autoclave Gas")
            ax.legend(loc=2, prop={'size': 7})
            ax.grid()
            plt.savefig("LBM_Temp_Log/" + str(frame) + ".png", dpi=1500)
            plt.close(fig)


            frame += 1 # increment the frame count

        plt.savefig("Results/" + str(frame), dpi=1500) # save a copy of the last colomap frame to the Results folder

        plt.close(self.theFig) # end the plotting

        Temp_N, Temp_S, Temp_E, Temp_W, Temp_MN, Temp_MS, Temp_ME, Temp_MW, Mold_Temp = self.compute_wall_temp() # compute the wall temperatures

        # quickly visualise the wall temperatures

        plt.plot(range(len(Temp_N)), Temp_N)
        plt.plot(range(len(Temp_S)), Temp_S)
        plt.show()
        plt.ylim((0, 400))
        plt.plot(range(len(Temp_MN)), Temp_MN)
        plt.plot(range(len(Temp_MS)), Temp_MS)
        #plt.plot(range(len(Temp_W)), Temp_W)
        plt.show()
        plt.plot(range(len(Temp_W)), Temp_W)
        plt.show()

        # save the gas flow temperature field of the last frame to a .csv file

        Matrix = self.Temp
        self.saveCSV('Temperature_Gradient', Matrix[0], Matrix, False)

        # save the average temperature log to a .csv file

        Matrix = np.transpose(np.array([np.array(time)*300/60, np.array(Temp_avg) - 273.15]))
        Titles = np.array(['time [min]', 'Average Temperature [C]'])
        self.saveCSV('Simulated_Temperature_Log', Titles, Matrix, True)

        # return the temperature field variations at the walls

        return Temp_N, Temp_S, Temp_E, Temp_W, Temp_MN, Temp_MS, Temp_ME, Temp_MW, Mold_Temp

    def load_Temp_Gradient(self, file):

        """
        This function loads the temperature field in the autoclave, computes the wall temperatures and plots them into figures
        :param file: the .csv file containing the autoclave gas temperature field
        :return: the wall temperature variations
        """
        # import the temperature field data

        data = np.genfromtxt(file, delimiter=",")
        self.Temp = data

        # compute the temperature variations at the walls of the autoclave and mould assembly

        Temp_N, Temp_S, Temp_E, Temp_W, Temp_MN, Temp_MS, Temp_ME, Temp_MW, Mold_Temp = self.compute_wall_temp()

        # create length vectors for each external wall in the autoclave

        length_N = np.arange(0, len(Temp_N), 1)*self.ds
        length_S = np.arange(0, len(Temp_S) , 1) * self.ds
        length_E = np.arange(0, len(Temp_E) , 1) * 2*np.arccos(1 - (self.Autoclave.L_Out - self.Autoclave.L_In)/self.Autoclave.C_Rad)/len(Temp_E)*self.Autoclave.C_Rad
        length_W = np.arange(0, len(Temp_W) , 1) * 2*np.arccos(1 - (self.Autoclave.L_Out - self.Autoclave.L_In)/self.Autoclave.C_Rad)/len(Temp_E)*self.Autoclave.C_Rad

        # plot and save the northern and southern temperature variations of the walls (and their averages)

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.suptitle("Autoclave Temperature Variation on the Horizontal Walls")
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel(r"Temperature [$^{\circ}$ C]")
        ax.set_ylim(0, 100)
        ax.plot(length_N, Temp_N - 273.15, linestyle='-', color='red',
                label="Temperature Variation on the North Wall")
        ax.plot(length_N, np.average(Temp_N)*np.ones(len(length_N)) - 273.15, linestyle='--', color='red', linewidth=0.8,
                label="Averaged Temperature of " + str(round(np.average(Temp_N) - 273.15)) + r" $^{\circ}$ C on the North Wall")
        ax.plot(length_N, Temp_S - 273.15, linestyle='-', color='blue',
                label="Temperature Variation on the South Wall")
        ax.plot(length_S, np.average(Temp_S) * np.ones(len(length_S)) - 273.15, linestyle='--', color='blue', linewidth=0.8,
                label="Averaged Temperature of " + str(round(np.average(Temp_S) - 273.15)) + r" $^{\circ}$ C on the South Wall")
        ax.legend(loc=3, prop={'size': 7})
        ax.grid()
        plt.savefig("Results/" + "Autoclave_Horizontal_Walls_Temperature_Variation" + ".png", dpi=200)
        plt.close(fig)

        # plot and save the bulkhead temperature variations at the walls (and their averages)

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.suptitle("Autoclave Temperature Variation on the Side Walls")
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel(r"Temperature [$^{\circ}$ C]")
        ax.set_ylim(0, 150)
        ax.plot(length_E, Temp_E - 273.15, linestyle='-', color='green',
                label="Temperature Variation on the East Wall")
        ax.plot(length_E, np.average(Temp_E) * np.ones(len(length_E)) - 273.15, linestyle='--', color='green', linewidth=0.8,
                label="Averaged Temperature of " + str(round(np.average(Temp_E) - 273.15)) + r" $^{\circ}$ C on the East Wall")
        ax.plot(length_W, Temp_W - 273.15, linestyle='-', color='orange',
                label="Temperature Variation on the West Wall")
        ax.plot(length_W, np.average(Temp_W) * np.ones(len(length_W)) - 273.15, linestyle='--', color='orange', linewidth=0.8,
                label="Averaged Temperature of " + str(round(np.average(Temp_W) - 273.15)) + r" $^{\circ}$ C on the West Wall")
        ax.legend(prop={'size': 7})
        ax.grid()
        plt.savefig("Results/" + "Autoclave_Side_Walls_Temperature_Variation" + ".png", dpi=200)
        plt.close(fig)

        # plot and save the significant mould temperature variations (and their averages)

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.suptitle("Autoclave Temperature Variation on the Mould")
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel(r"Temperature [$^{\circ}$ C]")
        ax.plot(np.array(range(len(Temp_MN)))*self.ds, Temp_MN - 273.15, linestyle='-', color='red',
                label="Temperature Variation on the North Mould")
        ax.plot(np.array(range(len(Temp_MN)))*self.ds, np.average(Temp_MN) * np.ones(len(Temp_MN)) - 273.15, linestyle='--', color='red', linewidth=0.8,
                label="Averaged Temperature of " + str(round(np.average(Temp_MN) - 273.15)) + r" $^{\circ}$ C on the North Surface")
        ax.plot(np.array(range(len(Temp_MS))) * self.ds, Temp_MS - 273.15, linestyle='-', color='blue',
                label="Temperature Variation on the South Mould")
        ax.plot(np.array(range(len(Temp_MS))) * self.ds, np.average(Temp_MS) * np.ones(len(Temp_MS)) - 273.15, linestyle='--', color='blue', linewidth=0.8,
                label="Averaged Temperature of " + str(round(np.average(Temp_MS) - 273.15)) + r" $^{\circ}$ C on the South Surface")
        ax.legend(prop={'size': 7})
        ax.set_ylim(0, 150)
        ax.grid()
        plt.savefig("Results/" + "Mould_Walls_Temperature_Variation" + ".png", dpi=200)
        plt.close(fig)

        # return the variations in temperature

        return Temp_N, Temp_S, Temp_E, Temp_W, Temp_MN, Temp_MS, Temp_ME, Temp_MW, Mold_Temp

    def saveCSV(self, name, Titles, Matrix, withTitles):

        """
        This function plots data into a .csv file
        :param name: (String) - the name of the file
        :param Titles: (String[]) - list with the titles of each column
        :param Matrix: (float[][]) - the matrix of data
        :param withTitles: (boolean) - a boolean saying whether to save the column titles or not
        :return: -
        """

        if len(Titles) != len(Matrix[0]):
            print("Columns don't match with titles!!")
        else:
            f = open("CSV_Data/" + name + ".csv", 'w+')
            if withTitles:
                for i in range(len(Titles)):
                    if i < len(Titles) - 1:
                        f.write(Titles[i] + ',')
                    else:
                        f.write(Titles[i])
                f.write('\n')

            for i in range(len(Matrix)):
                for j in range(len(Matrix[i])):
                    if j < len(Matrix[i]) - 1:
                        f.write(str(Matrix[i][j]) + ',')
                    else:
                        f.write(str(Matrix[i][j]) + '\n')
            f.close()