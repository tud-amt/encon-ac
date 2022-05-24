# This module contains all the heat transfer simulation code.
#
# The Simulation class takes as input the properties of the autoclave and mould assembly. It has 2 simulation functions:
#
# sim_CFRP_Temp simulates and saves the temperature distribution and heat transfer within the mould assembly
# sim_Auto_Temp simulates and saves the temperature distribution and heat transfer within the autoclave body
#
# The Simulation_Simple class simulates and saves the heat transfer through all components and with the environment assuming constant temperature
#
#



import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy import interpolate


class Simulation:

    def __init__(self, CFRP_Panel, Bleeder, Tool_Plate, Gas, Autoclave, Fans, dx, dz, dx_a, dt):

        """
        This function initializes the Simulation class.
        :param CFRP_Panel: the CFRP panel object
        :param Bleeder: the Bleeder object
        :param Tool_Plate: the Tool Plate object
        :param Gas: The Gas object
        :param Autoclave: The Autoclave object
        :param dx: (float) the x-axis step of the mould domain
        :param dz: (float) the z-axis step of the mould domain
        :param dx_a: (float) the step of the autoclave domain
        :param dt: the time step
        """

        self.Gas = Gas
        self.Panel = CFRP_Panel
        self.Bleeder = Bleeder
        self.Tool_plate = Tool_Plate
        self.Autoclave = Autoclave
        self.Fans = Fans
        self.dx = dx
        self.dx_a = dx_a
        self.dz = dz
        self.dt = dt
        self.kx, self.kz = self.compute_Ply_Conductivities(Panel=CFRP_Panel)
        self.Panel_Width = CFRP_Panel.Width
        self.Bleeder_Width = Bleeder.Width
        self.Tool_Width = Tool_Plate.Width

    def compute_Ply_Conductivities(self, Panel):

        """
        This function computes the conductivity distribution inside the CFRP panel, for a given ply configuration
        :param Panel: the CFRP panel
        :return: k_x (float) k_z (float): the conductivities in the x and z-axis direction
        """

        k_fibre = Panel.Fibre.Therm_Cond
        k_resin = Panel.Resin.Therm_Cond

        B = 2 * (k_fibre / k_resin - 1)

        f_fractions = np.array(Panel.Fibre_fractions)
        orientations = np.array(Panel.Orientations)
        k_normal = k_resin * ((1 - 2 * (f_fractions / np.pi) ** (0.5)) + 1 / B * (
                np.pi - 4 / (B ** 2 * f_fractions / np.pi - 1) ** (0.5) * np.arctan(
            (B ** 2 * f_fractions / np.pi - 1) ** (0.5) / (1 + (B ** 2 * f_fractions / np.pi)))))
        k_parallel = f_fractions * k_fibre + (1 - f_fractions) * k_resin
        k_x = k_parallel * (np.cos(orientations)) ** 2 + k_normal * (np.sin(orientations)) ** 2
        k_z = k_normal
        return k_x, k_z

    def set_initial_CFRP_mesh(self, Temp_Log):

        """
        This function creates the matrices of the CFRP mold domain
        :param Temp_Log: (float[]) the preset temperature log
        :return: Mold_mesh (float[][]) - the discretized mold temperature mesh of the domain
                 kx_mold (float[][]) - the discretized x-axis mold conductivity mesh of the domain
                 kz_mold (float[][]) - the discretized z-axis mold conductivity mesh of the domain
                 density_mold (float[][]) - the discretized mold density mesh of the domain
                 Cp_mold (float[][]) - the discretized mold specific heat mesh of the domain
                 Tool_mesh (float[][]) - the discretized tool plate temperature mesh of the domain
                 k_tool (float[][]) - the discretized tool plate conductivity mesh of the domain
                 density_tool (float[][]) - the discretized tool density mesh of the domain
                 Cp_tool (float[][]) - the discretized tool specific heat mesh of the domain

        """

        panel = self.Panel
        bleeder = self.Bleeder
        tool_plate = self.Tool_plate
        self.T0 = Temp_Log[0] + 273.15

        x_p_st = int((bleeder.Length - panel.Length) / 2 / self.dx)
        x_p_end = int((bleeder.Length + panel.Length) / 2 / self.dx)
        z_p_st = int(0)
        z_p_end = int(panel.Height / 2 / self.dz)

        Mold_mesh = np.ones((int(bleeder.Length / self.dx), int(bleeder.Height / self.dz))) * Temp_Log[0]
        Tool_mesh = np.ones((int(tool_plate.Length / self.dx), int(tool_plate.Height / self.dz)))
        kx_mold = np.ones((int(bleeder.Length / self.dx), int(bleeder.Height / self.dz))) * bleeder.k
        kz_mold = np.ones((int(bleeder.Length / self.dx), int(bleeder.Height / self.dz))) * bleeder.k
        density_mold = np.ones((int(bleeder.Length / self.dx), int(bleeder.Height / self.dz))) * bleeder.Density
        Cp_mold = np.ones((int(bleeder.Length / self.dx), int(bleeder.Height / self.dz))) * bleeder.Cp

        sum = 0
        density_f = panel.Fibre.Mass*2/panel.Length/panel.Width/panel.Height
        density_r = panel.Resin.Mass * 2 / panel.Length / panel.Width / panel.Height
        for i in range(len(self.kx)):
            dz = int(panel.Thicknesses[i] / self.dz)
            kx_mold[x_p_st:x_p_end, (z_p_st + sum):(z_p_st + sum + dz)] = self.kx[i]
            kz_mold[x_p_st:x_p_end, (z_p_st + sum):(z_p_st + sum + dz)] = self.kz[i]
            density_mold[x_p_st:x_p_end, (z_p_st + sum):(z_p_st + sum + dz)] = panel.Fibre_fractions[
                                                                                   i] * density_f + (
                                                                                       1 - panel.Fibre_fractions[
                                                                                   i]) * density_r
            Cp_mold[x_p_st:x_p_end, (z_p_st + sum):(z_p_st + sum + dz)] = panel.Fibre_fractions[i] * panel.Fibre.Cp + (
                    1 - panel.Fibre_fractions[i]) * panel.Resin.Cp

            sum += dz
        k_tool = np.ones((int(tool_plate.Length / self.dx), int(tool_plate.Height / self.dz))) * bleeder.k
        density_tool = np.ones((int(tool_plate.Length / self.dx),
                                int(tool_plate.Height / self.dz))) * bleeder.Density
        Cp_tool = np.ones((int(tool_plate.Length / self.dx),
                           int(tool_plate.Height / self.dz))) * bleeder.Cp
        return Mold_mesh, kx_mold, kz_mold, density_mold, Cp_mold, Tool_mesh, k_tool, density_tool, Cp_tool

    def set_initial_A_mesh(self, Temp_Log, Wall_Temperatures):

        """
        This function creates the meshes for the autoclave discretized 2D domain
        :param Temp_Log: (float[]) - the preset temperature log
        :param Wall_Temperatures: (float[][]) - the recorded wall temperature variations
        :return: Auto_mesh (float[][]) - the autoclave temperature mesh for the 2D domain
                 kA_mesh (float[][]) - the autoclave conductivity mesh for the 2D domain
                 rhoA_mesh (float[][]) - the autoclave density mesh for the 2D domain
                 CpA_mesh (float[][]) - the autoclave specific heat mesh for the 2D domain
                 Temp_N (float[]) - the north wall temperature variation
                 Temp_S (float[]) - the south wall temperature variation
                 Temp_E (float[]) - the east wall temperature variation
                 Temp_W (float[]) - the west wall temperature variation
        """

        ds = self.dx_a
        self.T0 = Temp_Log[0] + 273.15
        L_Outer = int(4 / np.pi * self.Autoclave.Vol_tot / self.Autoclave.D_Out**2 /ds)
        W_Outer = int(self.Autoclave.D_Out/ds)
        L_Inner = int(4 / np.pi * self.Autoclave.V_gas / (self.Autoclave.D_Out - 2 * self.Autoclave.t_wall)**2 / ds)
        W_Inner = int((self.Autoclave.D_Out - 2*self.Autoclave.t_wall)/ds)
        thickness = int(self.Autoclave.t_wall/ds)
        #print(L_Outer, L_Inner, thickness)

        kA_mesh = np.ones((L_Outer, W_Outer))*self.Autoclave.k
        Auto_mesh = np.ones((L_Outer, W_Outer))*(self.T0)
        rhoA_mesh = np.ones((L_Outer, W_Outer))*self.Autoclave.rho
        CpA_mesh = np.ones((L_Outer, W_Outer)) * self.Autoclave.Cp

        temp_N = interpolate.interp1d(range(0, len(Wall_Temperatures[0])), Wall_Temperatures[0])
        temp_S = interpolate.interp1d(range(0, len(Wall_Temperatures[1])), Wall_Temperatures[1])
        temp_E = interpolate.interp1d(range(0, len(Wall_Temperatures[2])), Wall_Temperatures[2])
        temp_W = interpolate.interp1d(range(0, len(Wall_Temperatures[3])), Wall_Temperatures[3])

        Temp_N = temp_N(np.arange(0, L_Inner, 1)*(len(Wall_Temperatures[0])-1)/L_Inner)
        Temp_S = temp_S(np.arange(0, L_Inner, 1)*(len(Wall_Temperatures[1])-1)/L_Inner)
        Temp_E = temp_E(np.arange(0, W_Inner, 1)*(len(Wall_Temperatures[2])-1)/W_Inner/np.pi)
        Temp_W = temp_W(np.arange(0, W_Inner, 1)*(len(Wall_Temperatures[3])-1)/W_Inner)

        Auto_mesh[thickness:(thickness + L_Inner), int(W_Outer/2 + W_Inner/2)] = self.T0
        Auto_mesh[thickness:(thickness + L_Inner), int(W_Outer/2 - W_Inner/2)] = self.T0
        Auto_mesh[thickness + L_Inner,int( W_Outer/2 - W_Inner/2): int(W_Outer/2 + W_Inner/2)] = self.T0
        Auto_mesh[thickness, int(W_Outer / 2 - W_Inner / 2): int(W_Outer / 2 + W_Inner / 2)] = self.T0
        Auto_mesh[thickness:thickness + L_Inner, thickness:thickness+W_Inner] = self.T0

        return Auto_mesh, kA_mesh, rhoA_mesh, CpA_mesh, Temp_N, Temp_S, Temp_E, Temp_W

    def degree_of_cure_prime(self, T, alpha):

        """
        This function applies an empirical relation to compute the cure rate of the CFRP as a function of temperature
        :param T: (float) the current average temperature in the mould assembly
        :param alpha: (float) the current cure rate
        :return: (float) the new cure rate
        """

        return ((np.exp(1.5 - 3300/T) + np.exp(12.6 - 6140/T)*alpha)*(1 - alpha)**2 + np.exp(39 - 19000/T)*(1 - alpha)**(1.219))

    def heat_transfer(self, T_steel):

        """
        This function computes the heat transfer coefficient of the autoclave gas for a given internal gas temperature
        :param T: The gas temperature at a given location in the autoclave
        :return: h (float): the heat transfer coefficient
        """

        kair = 0.02587  # W / (m K)
        rhoair = 1.204  # kg / m^3
        muair = 1.813e-5  # kg/m*s
        Cpair = 1006  # J/K*kg
        beta = 3.43e-3  # 1/K (Coefficient of thermal expansion)
        g = 9.80665  # m/s2
        T0 = 20
        Da = self.Autoclave.D_Out


        Gr_a = g * beta * Da ** 3 * rhoair ** 2 * (T_steel - T0) / (muair ** 2)  # Grashof number, free flow
        Pr_a = Cpair * muair / kair  # Prandtl number, free flow
        Ra_a = Gr_a * Pr_a

        Nu = (0.60 + (0.387 * Ra_a ** (1 / 6)) / (1 + (0.0559 / Pr_a) ** (9 / 16)) ** (8 / 27)) ** 2
        h = Nu * kair / Da

        #
        #
        # rho = 1.225
        # D = self.Autoclave.D_Out
        # mu_0 = 1.81 * 10 ** (-5)
        # T_0 = self.T0
        # mu = mu_0 * (T / T_0) ** 1.5 * (T_0 + 104.7) / (T + 104.7)
        # k = 0.029
        # C = 0.53
        # N = 0.25
        # alpha = k/rho/self.Gas.Cp
        # beta = 3.41
        # g = 9.81
        # h = k/D*C*((g*beta*D**3*abs(T-T_0)/(mu/rho)**2)*(mu/rho/alpha))**N

        return h

    def compute_CFRP_frame(self, M_mesh, km_x, km_z, rho_m, Cp_m, T_mesh, kt, rho_t, Cp_t, Gas_Temp, alpha):

        """
        This function computes one frame of the heat transfer simulation for a CFRP panel.
        :param M_mesh: (numpy array) the temperature distribution of the CFRP panel + Bleeder/Breather
        :param km_x: (numpy array) the x-axis conductivity of the mould assembly (CFRP + Bleeder)
        :param km_z: (numpy array) the z-axis conductivity of the mould assembly (CFRP + Bleeder)
        :param rho_m: (numpy array) the density distribution of the mould
        :param Cp_m: (numpy array) the specific heat distribution of the mould
        :param T_mesh: (numpy array) the temperature distribution of the tool plate
        :param kt: (numpy array) the conductivity of the tool plate
        :param rho_t: (numpy array) the density distribution of the tool plate
        :param Cp_t: (numpy array) the specific heat distribution of the tool plate
        :param Gas_Temp: (float) the temperature of the gas near the mould surface
        :param alpha: (float) the cure rate
        :return: M_mesh_new (numpy array) T_mesh_new (numpy array): the updated mould and tool plate temperature distributions
        """

        # initializing the mould assembly and tool plate temperature meshes for the next frame

        M_mesh_new = np.zeros((len(M_mesh), len(M_mesh[0])))
        T_mesh_new = np.zeros((len(T_mesh), len(T_mesh[0])))

        # Computing the new mould and tool plate themperature distributions

        def f_M(M): # updating the mould temperature mesh using the heat equation discretized with a central difference scheme
            mesh = np.zeros((len(M), len(M[0])))
            mesh[1:-1, 1:-1] = (km_x / (rho_m * Cp_m))[1:-1, 1:-1] * (np.roll(M, -1, axis=0)[1:-1, 1:-1] - 2 * M[1:-1, 1:-1]+ np.roll(M, 1, axis=0)[1:-1, 1:-1]) / self.dx ** 2 +(km_z / (rho_m * Cp_m))[1:-1, 1:-1] * (
                                                                         np.roll(M, -1, axis=1) [1:-1, 1:-1]- 2 * M[1:-1, 1:-1] + np.roll(
                                                                     M, 1, axis=1)[1:-1, 1:-1]) / self.dz ** 2
            return mesh

        def f_T(T): # updating the tool plate temperature mesh using the heat equation discretized with a central difference scheme
            mesh = np.zeros((len(T), len(T[0])))
            mesh[1:-1, 1:-1] = (
                kt[1:-1, 1:-1] / (rho_t[1:-1, 1:-1] * Cp_t[1:-1, 1:-1]) * (np.roll(T, -1, axis=0)[1:-1, 1:-1] - 2 * T[1:-1, 1:-1] + np.roll(
            T, 1, axis=0)[1:-1, 1:-1]) / self.dx ** 2 + kt[1:-1, 1:-1] / (rho_t[1:-1, 1:-1] * Cp_t[1:-1, 1:-1]) * (
                        np.roll(T, -1, axis=1)[1:-1, 1:-1] - 2 * T[1:-1, 1:-1] + np.roll(T, 1, axis=1)[1:-1, 1:-1]) / self.dz ** 2)
            return mesh

        # apply a 4th order Runge-Kutta scheme to time-march the temperature distribution of the mould

        k1m = self.dt*f_M(M_mesh)
        k2m = self.dt*f_M(M_mesh + k1m/2)
        k3m = self.dt*f_M(M_mesh + k2m/2)
        k4m = self.dt*f_M(M_mesh + k3m)

        # compute the temperature distribution of the mould at the next time step

        M_mesh_new[1:-1, 1:-1] = M_mesh[1:-1, 1:-1] + 1/6*(k1m + 2*k2m + 2*k3m + k4m)[1:-1, 1:-1]

        # apply a 4th order Runge-Kutta scheme to time-march the temperature distribution of the tool plate

        k1t = self.dt * f_T(T_mesh)
        k2t = self.dt * f_T(T_mesh + k1t / 2)
        k3t = self.dt * f_T(T_mesh + k2t / 2)
        k4t = self.dt * f_T(T_mesh + k3t)

        # compute the temperature distribution of the tool plate at the next time step

        T_mesh_new[1:-1, 1:-1] = T_mesh[1:-1, 1:-1] + 1/6*(k1t + 2*k2t + 2*k3t + k4t)[1:-1, 1:-1]

        # Computing the mould-tool plate domains contact points

        x_t_st = int((self.Tool_plate.Length - self.Bleeder.Length) / 2 / self.dx)
        x_t_end = int((self.Tool_plate.Length + self.Bleeder.Length) / 2 / self.dx) + 1

        # handling the boundary between the mould and the tool plate in both domains

        M_mesh_new[1:-1, 0] = M_mesh[1:-1, 0] + self.dt * ((km_x / (rho_m * Cp_m))[1:-1, 0] * (
                np.roll(M_mesh, -1, axis=0)[1:-1, 0] - 2 * M_mesh[1:-1, 0] + np.roll(M_mesh, 1, axis=0)[1:-1,
                                                                             0]) / self.dx ** 2 +
                                                           (km_z / (rho_m * Cp_m))[1:-1, 0] * (
                                                                   T_mesh[x_t_st + 2:x_t_end - 1, -1] - 2 * M_mesh[
                                                                                                            1:-1,
                                                                                                            0] + np.roll(
                                                               M_mesh, -1, axis=1)[1:-1, 0]) / self.dz ** 2)
        T_mesh_new[x_t_st + 2:x_t_end - 1, -1] = T_mesh[x_t_st + 2:x_t_end - 1, -1] + self.dt * (
                kt[x_t_st + 2:x_t_end - 1, -1] / (
                rho_t[x_t_st + 2:x_t_end - 1, -1] * Cp_t[x_t_st + 2:x_t_end - 1, -1]) * (
                        np.roll(T_mesh, -1, axis=0)[x_t_st + 2:x_t_end - 1, -1] - 2 * T_mesh[x_t_st + 2:x_t_end - 1,
                                                                                      -1] + np.roll(
                    T_mesh, 1, axis=0)[x_t_st + 2:x_t_end - 1, -1]) / self.dx ** 1 +
                kt[x_t_st + 2:x_t_end - 1, -1] / (
                        rho_t[x_t_st + 2:x_t_end - 1, -1] * Cp_t[x_t_st + 2:x_t_end - 1, -1]) * (
                        np.roll(T_mesh, -5, axis=1)[x_t_st + 2:x_t_end - 1, -1] - 2 * T_mesh[x_t_st + 2:x_t_end - 1,
                                                                                      -1] + M_mesh[1:-1,
                                                                                            2]) / self.dz ** 2)

        # Computing the heat transfer coefficient at the borders of the entire domain with the Nitrogen gas

        h = self.heat_transfer(Gas_Temp)

        # Forcing convective boundary conditions at the mould and tool plate gas-exposed surfaces

        M_mesh_new[0, :] = M_mesh[1, :] * (1 - h * self.dx / km_x[1, :]) + Gas_Temp * h * self.dx / km_x[1, :]
        M_mesh_new[-1, :] = M_mesh[-2, :] * (1 - h * self.dx / km_x[-2, :]) + Gas_Temp * h * self.dx / km_x[-2, :]
        M_mesh_new[:, -1] = M_mesh[:, -2] * (1 - h * self.dz / km_z[:, -2]) + Gas_Temp * h * self.dz / km_z[:, -2]
        T_mesh_new[1, :] = T_mesh[0, :] * (1 - h * self.dx / kt[0, :]) + Gas_Temp * h * self.dx / kt[0, :]
        T_mesh_new[-2, :] = T_mesh[-1, :] * (1 - h * self.dx / kt[-1, :]) + Gas_Temp * h * self.dx / kt[-1, :]
        T_mesh_new[:, 0] = T_mesh[:, 1] * (1 - h * self.dz / kt[:, 1]) + Gas_Temp * h * self.dz / kt[:, 1]
        T_mesh_new[:x_t_st + 2, -1] = T_mesh[:x_t_st + 2, -2] * (1 - h * self.dz / kt[:x_t_st + 2, -2]) + Gas_Temp * h * self.dz / kt[:x_t_st + 2, -2]
        T_mesh_new[x_t_end - 1:, -1] = T_mesh[x_t_end - 1:, -2] * (1 - h * self.dz / kt[x_t_end - 1:, -2]) + Gas_Temp * h * self.dz / kt[x_t_end - 1:, -2]

        return M_mesh_new, T_mesh_new

    def compute_Auto_frame(self, A_mesh, k_m, rho_m, Cp_m, Gas_Temp, Temp_N, Temp_S, Temp_E, Temp_W, Temp_0):

        """
        This function marches the autoclave 2D domain in time with one step
        :param A_mesh: (float[][]) - the temperature mesh of the autoclave body
        :param k_m: (float[][]) - the conductivity mesh of the autoclave body
        :param rho_m: (float[][]) - the density mesh of the autoclave body
        :param Cp_m: (float[][]) - the specific heat mesh of the autoclave body
        :param Gas_Temp: (float[][]) - the temperature mesh of the enclosed gas outside the 2D domain
        :param Temp_N: (float[]) - the north wall temperature variation
        :param Temp_S: (float[]) - the south wall temperature variation
        :param Temp_E: (float[]) - the east wall temperature variation
        :param Temp_W: (float[]) - the west wall temperature variation
        :param Temp_0: (float) - the mold temperature
        :return: A_mesh_new (float[][]) - the updated autoclave body temperature mesh
        """

        ds = self.dx_a  # import the autoclave spatial step into a local variable

        # compute the autoclave domain dimensions based on its volume and geometry

        L_Outer = int(4 / np.pi * self.Autoclave.Vol_tot / self.Autoclave.D_Out ** 2 / ds)
        W_Outer = int(self.Autoclave.D_Out / ds)
        L_Inner = int(4 / np.pi * self.Autoclave.V_gas / (self.Autoclave.D_Out - 2 * self.Autoclave.t_wall) ** 2 / ds)
        W_Inner = int((self.Autoclave.D_Out - 2 * self.Autoclave.t_wall) / ds)
        thickness = int(self.Autoclave.t_wall / ds)
        #print(L_Outer, L_Inner)

        A_mesh_new = np.zeros((len(A_mesh), len(A_mesh[0]))) # initialize the autoclave temperature mesh for the next frame

        def f_M(M): # update the temperature mesh using a central difference scheme and the heat equation
            mesh = np.zeros((len(M), len(M[0])))
            mesh[1:-1, 1:-1] = (k_m / (rho_m * Cp_m))[1:-1, 1:-1] * (np.roll(M, -1, axis=0)[1:-1, 1:-1] - 2 * M[1:-1, 1:-1]+ np.roll(M, 1, axis=0)[1:-1, 1:-1]) / ds ** 2 +(k_m / (rho_m * Cp_m))[1:-1, 1:-1] * (
                                                                         np.roll(M, -1, axis=1) [1:-1, 1:-1]- 2 * M[1:-1, 1:-1] + np.roll(
                                                                     M, 1, axis=1)[1:-1, 1:-1]) / ds ** 2
            return mesh

        # apply 4th order Runge-Kutta time propagation to compute the temperature mesh at the next frame

        k1m = self.dt * f_M(A_mesh)
        k2m = self.dt * f_M(A_mesh + k1m / 2)
        k3m = self.dt * f_M(A_mesh + k2m / 2)
        k4m = self.dt * f_M(A_mesh + k3m)

        A_mesh_new[1:-1, 1:-1] = A_mesh[1:-1, 1:-1] + 1 / 6 * (k1m + 2 * k2m + 2 * k3m + k4m)[1:-1, 1:-1] # compute the new temperature field

        # apply the linearly-scaled recorded Nitrogen temperature field variations as Dirichlet boundary conditions for each wall of the domain

        A_mesh_new[thickness:thickness + L_Inner, int(W_Outer / 2 + W_Inner / 2)] = self.T0 + Temp_N/Temp_0 * (Gas_Temp + 273.15 - self.T0)
        A_mesh_new[thickness:thickness + L_Inner, int(W_Outer / 2 - W_Inner / 2)] = self.T0 + Temp_S/Temp_0 * (Gas_Temp + 273.15 - self.T0)
        A_mesh_new[thickness + L_Inner, int(W_Outer / 2 - W_Inner / 2): int(W_Outer / 2 + W_Inner / 2)] = self.T0 + Temp_E/Temp_0 * (Gas_Temp + 273.15 - self.T0)
        A_mesh_new[thickness, int(W_Outer / 2 - W_Inner / 2): int(W_Outer / 2 + W_Inner / 2)] = self.T0 + Temp_W/Temp_0 * (Gas_Temp + 273.15 - self.T0)

        # apply the ambient temperature to the areas outside the simulation domain

        A_mesh_new[thickness + 1 :thickness + L_Inner - 1, thickness + 1:thickness + W_Inner - 1] = self.T0

        # simulate the heat loss through the outer walls of the domain using the free-convection heat transfer coefficient

        h = self.heat_transfer(np.average(A_mesh[1, :]))
        A_mesh_new[0, :] = A_mesh[1, :] * (1 - h * ds / k_m[1, :]) + self.T0 * h * ds / k_m[1, :]
        h = self.heat_transfer(np.average(A_mesh[-2, :]))
        A_mesh_new[-1, :] = A_mesh[-2, :] * (1 - h * ds / k_m[-2, :]) + self.T0 * h * ds / k_m[-2, :]
        h = self.heat_transfer(np.average(A_mesh[:, -2]))
        A_mesh_new[:, -1] = A_mesh[:, -2] * (1 - h * ds / k_m[:, -2]) + self.T0 * h * ds / k_m[:, -2]
        h = self.heat_transfer(np.average(A_mesh[:, 1]))
        A_mesh_new[:, 0] = A_mesh[:, 1] * (1 - h * ds / k_m[:, 1]) + self.T0 * h * ds / k_m[:, 1]

        return A_mesh_new

    def sim_CFRP_Temp(self, Temp_log, time_log):

        """
        This function simulates the temperature variation and heat transfer in time for the CFRP panel within the autoclave.
        It plots the average and surface temperatures of the panel against the temperature log.
        It plots the temperature maps for every computed frame
        :param Temp_log: (numpy array) - the given temperature log array for the autoclave cycle
        :param time_log: (numpy array) - the accompanying time array for the temp. log
        :return: None
        """

        # initialize the CFRP mould assembly meshes in this order (mould temperature, mould x-axis-conducitvity, mould z-axis-conductivity,
        # mould specific heat, tool temperature, tool conductivity, tool density, tool specific heat)

        M_mesh, km_x, km_z, rho_m, Cp_m, T_mesh, kt, rho_t, Cp_t = self.set_initial_CFRP_mesh(Temp_Log=Temp_log)

        C_M_mesh = np.copy(M_mesh) # store the previous temperature distribution of the mould
        C_T_mesh = np.copy(T_mesh) # store the previous temperature distribution of the tool plate

        steps = int(time_log[-1] / self.dt) # compute the needed simulation steps for one cure cycle

        Temp = interpolate.interp1d(time_log, Temp_log, kind='linear', fill_value="extrapolate") # interpolate the experimental temperature log

        # initialize the time and heat log arrays

        time = []
        Mold_Temp_Avg = []
        Mold_Temp_Max = []
        Heat_Transfered = [0]
        Degree_of_Cure = []

        alpha = 0 # the initial degree of cure (no longer used in the code)

        # the simulation loop

        for i in range(steps):

            # compute the mesh positions where the tool plate and panel begin (in the x-axis and z-axis)

            x_t_st = int((self.Tool_plate.Length - self.Bleeder.Length) / 2 / self.dx) -1
            x_t_end = int((self.Tool_plate.Length + self.Bleeder.Length) / 2 / self.dx)
            x_p_st = x_t_st + int((self.Bleeder.Length - self.Panel.Length) / 2 / self.dx)
            x_p_end = x_t_st + int((self.Bleeder.Length + self.Panel.Length) / 2 / self.dx)
            z_p_st = len(T_mesh[0])
            z_p_end = len(T_mesh[0]) + int(self.Panel.Height / 2 / self.dz)

            alpha += self.degree_of_cure_prime(Temp(i * self.dt), alpha) * self.dt  # update the degree of cure

            # update the tool plate and mould temperature distributions based on current properties

            M_mesh, T_mesh = self.compute_CFRP_frame(M_mesh, km_x, km_z, rho_m, Cp_m, T_mesh, kt, rho_t, Cp_t,
                                                     Temp(i * self.dt), alpha)

            if i % 1000 == 0: # every 1000 steps

                # initialize the total mesh domain

                crossSect_matrix = np.ones((len(T_mesh), len(T_mesh[0]) + len(M_mesh[0]))) * Temp(i * self.dt) # initialize the matrix of the entire domain
                crossSect_matrix[:, :len(T_mesh[0])] = T_mesh # append the tool temperature distribution to the domain mesh
                crossSect_matrix[x_t_st + 1:x_t_end, len(T_mesh[0]) - 1:-1] = M_mesh # append the mould temperature distribution to the domain mesh

                print("Frame:", int(i / 1000)) # print the current computed frame

                # plot and save the CFRP mould assembly domain for the current frame.
                # The temperature is represented with a colormap. The components are displayed with black lines

                fig1 = plt.figure(figsize=(6, 3))
                ax1 = fig1.add_subplot()
                crossSection = ax1.imshow(np.matrix.transpose(crossSect_matrix[:, :]), origin='lower',
                                          norm=plt.Normalize(0, 150),
                                          cmap=plt.get_cmap('jet'), interpolation='none', extent=[0, self.Tool_plate.Length, 0, (len(T_mesh[0]) + len(M_mesh[0]))*self.dz*100])
                x = np.arange(0, len(T_mesh), 1)
                y = np.arange(0, len(T_mesh[0]) + len(M_mesh[0]), 1)
                X, Y = np.meshgrid(x, y)
                fig1.colorbar(cm.ScalarMappable(norm=plt.Normalize(0, 150), cmap=plt.get_cmap('jet')), ax=ax1, label=r"Temperature [$^{\circ}$ C]")
                fig1.suptitle('Mould Assembly Temperature Distribution')
                plt.xlabel("X-axis [m]")
                plt.ylabel("Y-axis [cm]")

                sum = 0
                x_tool = np.array([0, 0, len(T_mesh) - 1, len(T_mesh) - 1, 0])
                z_tool = np.array([0, len(T_mesh[0]), len(T_mesh[0]), 0, 0])
                x_breeder = np.array([x_t_st, x_t_st, x_t_end, x_t_end, x_t_st])
                z_breeder = np.array(
                    [len(T_mesh[0]), len(T_mesh[0]) + len(M_mesh[0]) - 1, len(T_mesh[0]) + len(M_mesh[0]) - 1,
                     len(T_mesh[0]), len(T_mesh[0])])
                ax1.plot(x_tool*self.dx, z_tool*self.dz*100, color='black')
                ax1.plot(x_breeder*self.dx, z_breeder*self.dz*100, color='black')
                for j in range(len(self.kx)):
                    x = np.array([x_p_st, x_p_st, x_p_end, x_p_end, x_p_st])*self.dx
                    z = np.array([z_p_st + int(sum / self.dz), z_p_st + int((sum + self.Panel.Thicknesses[j]) / self.dz),
                         z_p_st + int((sum + self.Panel.Thicknesses[j]) / self.dz), z_p_st + int(sum / self.dz),
                         z_p_st + int(sum / self.dz)])*self.dz*100
                    sum += self.Panel.Thicknesses[j]
                    ax1.plot(x, z, color='black')

                plt.savefig("frames/" + str(int(i / 1000)) + ".png",
                            dpi=200)
                plt.close(fig1)

                # compute and append the maximum and average temperature in the mould assembly

                time.append(i * self.dt) # append the current time to the time array

                Mold_Temp_Avg.append(np.average(crossSect_matrix[x_p_st:x_p_end, z_p_st:z_p_end]))   # compute and append the average temperature of the mould assembly
                Mold_Temp_Max.append(np.max(crossSect_matrix[x_p_st:x_p_end, int(z_p_end / 2):z_p_end]))  # compute and append the maximum temperature in the mould assembly

                # plot and save the simulated temperature log in the mould assembly next to the preset cycle temperature log

                fig = plt.figure()
                ax = fig.add_subplot()
                plt.suptitle("Temperature Log")
                ax.set_xlabel("Time [min]")
                ax.set_ylabel(r"Temperature [$\degree$ C]")
                ax.plot(np.array(time_log) / 60, Temp_log, color='black', label="Air Temperature in Autoclave")
                ax.plot(np.array(time) / 60, Mold_Temp_Avg, linestyle='--', color='blue',
                        label="Average CFRP Panel Temperature")
                ax.plot(np.array(time) / 60, Mold_Temp_Max, linestyle='-.', color='blue',
                        label="Surface Temperature of CFRP Panel")
                ax.legend(loc=2, prop={'size': 7})
                ax.grid()
                plt.savefig("TempLog/" + str(int(i / 1000)) + ".png", dpi=200)
                plt.close(fig)

                # the mould assembly transferred heat is computed and appended to the heat array

                Heat_Transfered.append(Heat_Transfered[-1] + max((np.sum((rho_m * self.dx * self.dz * Cp_m * (M_mesh - C_M_mesh))[x_p_st:x_p_end]) * self.Panel_Width +
                                       np.sum((rho_m * self.dx * self.dz * Cp_m * (M_mesh - C_M_mesh))[:x_p_st]) * self.Bleeder_Width +
                                       np.sum((rho_m * self.dx * self.dz * Cp_m * (M_mesh - C_M_mesh))[x_p_end:]) * self.Bleeder_Width +
                                       np.sum(rho_t * self.dx * self.dz * Cp_t * (T_mesh - C_T_mesh)) * self.Tool_Width), 0))

                C_M_mesh = np.copy(M_mesh)  # the copy of the mould temperature  distribution is updated
                C_T_mesh = np.copy(T_mesh)  # the copy of the tool plate temperature distribution is updated

                # the consumed heat log for the mould assembly is plotted and saved

                fig = plt.figure()
                ax = fig.add_subplot()
                plt.suptitle("Heat Log")
                ax.set_xlabel("Time [min]")
                ax.set_ylabel(r"Heat [MJ]")
                ax.plot(np.array(time) / 60, np.array(Heat_Transfered)[:-1] / 10 ** 6, linestyle='-', color='black',
                        label="Heat Transferred in Mold")
                ax.legend(loc=2, prop={'size': 7})
                ax.grid()
                plt.savefig("HeatLog/" + str(int(i / 1000)) + ".png", dpi=200)
                plt.close(fig)

                # the degree of cure log is plotted and saved

                Degree_of_Cure.append(alpha)
                fig = plt.figure()
                ax = fig.add_subplot()
                plt.suptitle("Degree of Cure Log")
                ax.set_xlabel("Time [min]")
                ax.set_ylabel(r"Degree of cure [-]")
                ax.plot(np.array(time) / 60, np.array(Degree_of_Cure), linestyle='-', color='black',
                        label="Degree of Cure in Mold")
                ax.legend(loc=2, prop={'size': 7})
                ax.grid()
                plt.savefig("CureLog/" + str(int(i / 1000)) + ".png", dpi=200)
                plt.close(fig)
        Matrix = np.transpose(np.array(
            [np.array(time), np.array(Mold_Temp_Avg), np.array(Mold_Temp_Max), Temp(time)]))
        Titles = ['Time [s]', 'Mold Average Temperature [deg C]', 'Mold Maximum Temperature [deg C]', 'Cycle Temperature [deg C]']
        self.saveCSV('CFRP_Temp_Log', Titles, Matrix, True)

    def sim_Auto_Temp(self, Temp_log, time_log, Wall_Temperatures, Temp_0, name):

        """
        This function simulates the heat transfer through the autoclave body and with the environment
        :param Temp_log: (float[]) the preset temperature log
        :param time_log: (float[]) the time array of the temperature log
        :param Wall_Temperatures: (float[]) the temperature variations of the autoclave walls
        :param Temp_0: (float) the mold temperature
        :param name: the name of the .csv file
        :return: time (float[]) - the time array
                 Heat_Transferred (float[]) - the total heat consumed array
                 Heat_Auto (float[]) - the autoclave body transferred heat array
                 Heat_Mould (float[]) - the mold assembly transferred heat array
                 Heat_Lost (float[]) - the autoclave body lost heat to the environment array
        """

        # Initialize the meshes used by the simulation in this order (Temperature, Conductivity, Density, Specific Heat, Temperature Variations on Walls)

        Auto_mesh, kA_mesh, rhoA_mesh, CpA_mesh, Temp_N, Temp_S, Temp_E, Temp_W = self.set_initial_A_mesh(Temp_log, Wall_Temperatures)

        ds = self.dx_a # the spatial step of the autoclave mesh

        L_Outer = int(4 / np.pi * self.Autoclave.Vol_tot / self.Autoclave.D_Out ** 2 / ds)
        W_Outer = int(self.Autoclave.D_Out / ds)
        L_Inner = int(4 / np.pi * self.Autoclave.V_gas / (self.Autoclave.D_Out - 2 * self.Autoclave.t_wall) ** 2 / ds)
        W_Inner = int((self.Autoclave.D_Out - 2 * self.Autoclave.t_wall) / ds)
        thickness = int(self.Autoclave.t_wall / ds)
        r = np.linspace(-W_Outer / 2 * ds, W_Outer / 2 * ds, len(Auto_mesh[0]))
        l = np.linspace(0, L_Outer * ds, len(Auto_mesh))
        R, L = np.meshgrid(r, l)

        steps = int(time_log[-1] / self.dt) # the number of simulation steps needed to simulate the cycle
        Temp = interpolate.interp1d(time_log, Temp_log, kind='linear', fill_value="extrapolate") # The experimental temperature log is interpolated
        C_Auto_mesh = np.copy(Auto_mesh) # The autoclave temperature distribution at the previous time step is recorded
        Max_A_mesh = np.copy(Auto_mesh)  # The initial value of the autoclave temperature distribution for which the average temp. is maximum
        T_max = 0 # Initial value of the maximum average autoclave temperature

        # initializing the arrays for time, autoclave max. and average temperature and heat transfer components

        time = []
        Auto_Temp_Avg = []
        Auto_Temp_Max = []
        Heat_Transfered = [0]
        Heat_Lost = [0]
        Heat_Mould = [0]
        Heat_Auto = [0]
        Energy_Fan = [0]



        # simulation loop

        for i in range(steps):
            # Update the autoclave temperature field based on current properties
            Auto_mesh = self.compute_Auto_frame(Auto_mesh, kA_mesh, rhoA_mesh, CpA_mesh, Temp(i * self.dt), Temp_N, Temp_S, Temp_E, Temp_W, Temp_0)

            T_avg = np.average(Auto_mesh) # compute the average temperature in the autoclave body

            if T_max < T_avg: # update the maximum average temperature, and the corresponding temperature distribution
                T_max = T_avg
                Max_A_mesh = np.copy(Auto_mesh)

            if i % 1000 == 0: # every 1000th step

                time.append(i * self.dt) # append a new element to the time array

                print("Frame:", int(i / 1000)) # print the current frame

                # print and save the color-map frame of the autoclave mesh

                fig1 = plt.figure(figsize=(6, 3))
                ax1 = fig1.add_subplot()
                crossSection = ax1.imshow(np.matrix.transpose(Auto_mesh[:, :]), origin='lower',
                                          norm=plt.Normalize(273.15, 350 + 273.15),
                                          cmap=plt.get_cmap('jet'), interpolation='none', extent=[0, len(Auto_mesh)*self.dx_a, 0, len(Auto_mesh[0])*self.dx_a])
                fig1.colorbar(cm.ScalarMappable(norm=plt.Normalize(0, 350), cmap=plt.get_cmap('jet')), ax=ax1, label=r"Temperature [$^{\circ}$ C]")
                fig1.suptitle('Autoclave Body Temperature Distribution')
                plt.xlabel("X-axis [m]")
                plt.ylabel("Y-axis [m]")
                x_A = np.array([0, 0, len(Auto_mesh) - 1, len(Auto_mesh) - 1, 0])*self.dx_a
                y_A = np.array([0, len(Auto_mesh[0]), len(Auto_mesh[0]), 0, 0])*self.dx_a
                x_A_i = np.array([thickness, thickness, thickness + L_Inner, thickness + L_Inner, thickness])*self.dx_a
                y_A_i = np.array(
                    [thickness, thickness + W_Inner, thickness + W_Inner,
                     thickness, thickness])*self.dx_a
                ax1.plot(x_A, y_A, color='black', linewidth=0.1)
                ax1.plot(x_A_i, y_A_i, color='black', linewidth=0.1)
                plt.savefig("frames_Autoclave/" + str(int(i / 1000)) + ".png",
                            dpi=200)
                plt.close(fig1)

                # import the CFRP mould assembly objects into local variables

                fibre = self.Panel.Fibre
                resin = self.Panel.Resin
                bleeder = self.Bleeder
                tool = self.Tool_plate
                gas = self.Gas

                # compute an estimation of the heat transfered to the CFRP mould assembly and Nitrogen gas (assuming constant temperature for the CFRP mould)

                Extra_Heat = (fibre.Mass*fibre.Cp + resin.Mass*resin.Cp + bleeder.Mass*bleeder.Cp + tool.Mass*tool.Cp + gas.Mass*gas.Cp)*max(0, np.average((Auto_mesh - C_Auto_mesh)))

                # compute and append the total autoclave consumed heat

                Heat_Transfered.append(Heat_Transfered[-1] + max((np.sum((rhoA_mesh * self.dx_a**2 * CpA_mesh * (Auto_mesh - C_Auto_mesh)) * np.abs(R) * np.pi)), 0) + Extra_Heat)

                # compute and append the autoclave body consumed heat

                Heat_Auto.append(Heat_Auto[-1] + max((np.sum((rhoA_mesh * self.dx_a**2 * CpA_mesh * (Auto_mesh - C_Auto_mesh)) * np.abs(R) * np.pi)), 0))

                # compute the heat lost to the environment through the autoclave walls

                Energy_Fan.append(Energy_Fan[-1] + self.Fans.Power * self.dt)

                Heat_Transfered[-1] += Energy_Fan[-1]

                # compute the average surface temperature on the outer walls of the autoclave
                T_surface = 0.25*(np.average(Auto_mesh[0, :]) + np.average(Auto_mesh[-1, :]) + np.average(Auto_mesh[:, 0]) + np.average(Auto_mesh[:, -1]))
                Area = self.Autoclave.Area # import the autoclave walls surface area
                h = self.heat_transfer(T_surface) # estimate the free-convection heat transfer of the environment air
                Heat_Lost.append(Heat_Lost[-1] + max(self.dt*1000*h*Area*(T_surface - self.T0), 0)) # compute and append the lost heat
                Heat_Transfered[-1] += max(self.dt*1000*h*Area*(T_surface - self.T0), 0) # append the same heat loss to the total autoclave consumed heat array

                Heat_Mould.append(Heat_Mould[-1] + Extra_Heat) # append the heat transferred to the mould and gass to the corresponding array

                C_Auto_mesh = np.copy(Auto_mesh) # update the autoclave temperature field copy

                # plot the total consumed heat log and save it

                fig = plt.figure()
                ax = fig.add_subplot()
                plt.suptitle("Heat Log")
                ax.set_xlabel("Time [min]")
                ax.set_ylabel(r"Heat [kWh]")
                ax.plot(np.array(time) / 60, np.array(Heat_Transfered)[:-1] / 3600000, linestyle='-', color='black',
                        label="Heat Transferred to Autoclave Body")
                ax.legend(loc=2, prop={'size': 7})
                ax.grid()
                plt.savefig("Auto_HeatLog/" + str(int(i / 1000)) + ".png", dpi=200)
                plt.close(fig)

        # plot and save the autoclave temperature mesh for the temperature distribution corresponding to the maximum average temperature

        fig1 = plt.figure(figsize=(6, 3))
        ax1 = fig1.add_subplot()
        crossSection = ax1.imshow(np.matrix.transpose(Max_A_mesh[:, :]), origin='lower',
                                  norm=plt.Normalize(273.15, 150 + 273.15),
                                  cmap=plt.get_cmap('jet'), interpolation='none', extent=[0, len(Max_A_mesh) * self.dx_a, 0, len(Max_A_mesh[0]) * self.dx_a])
        fig1.colorbar(cm.ScalarMappable(norm=plt.Normalize(0, 150), cmap=plt.get_cmap('jet')), ax=ax1, label=r"Temperature [$^{\circ}$ C]")
        fig1.suptitle('Autoclave Body Temperature Distribution')
        plt.xlabel("X-axis [m]")
        plt.ylabel("Y-axis [m]")
        x_A = np.array([0, 0, len(Auto_mesh) - 1, len(Auto_mesh) - 1, 0]) * self.dx_a
        y_A = np.array([0, len(Auto_mesh[0]), len(Auto_mesh[0]), 0, 0]) * self.dx_a
        x_A_i = np.array([thickness, thickness, thickness + L_Inner, thickness + L_Inner, thickness]) * self.dx_a
        y_A_i = np.array(
            [thickness, thickness + W_Inner, thickness + W_Inner,
             thickness, thickness]) * self.dx_a
        ax1.plot(x_A, y_A, color='black', linewidth=0.1)
        ax1.plot(x_A_i, y_A_i, color='black', linewidth=0.1)
        plt.savefig("Results/" + "Autoclave_Temperature_Distribution" + ".png",
                    dpi=200)
        plt.close(fig1)

        # save the experimental and simulated energy logs and their components into .csv files

        e_time, e_Energy_Log = self.import_Energy_Log('Data\Energy_Log.csv')
        E_Energy_Log = interpolate.interp1d(e_time, e_Energy_Log, kind='linear', fill_value="extrapolate")
        E_Energy = E_Energy_Log(np.array(time))
        Matrix = np.transpose(np.array([np.array(time), np.array(Heat_Auto)[:-1]/ 3600000, np.array(Heat_Lost)[:-1]/3600000, np.array(Heat_Mould)[:-1]/3600000, np.array(Heat_Transfered)[:-1]/3600000, E_Energy]))
        Titles = ['Time [s]', 'Autoclave Body Heat Transfer [kWh]', 'Heat Lost To The Environment[kWh]', 'Mould Assembly Heat Transfer [kWh]', 'Simulated Energy Log [kWh]', 'Experimental Energy Log [kWh]']
        self.saveCSV(name, Titles, Matrix, True)

        return time, Heat_Transfered, Heat_Auto, Heat_Mould, Heat_Lost

    def saveCSV(self, name, Titles, Matrix, withTitles):

        """
        This function saves arrays to a .csv file
        :param name: (String) - the file name
        :param Titles: (String[]) - the titles of the columns
        :param Matrix: (float[][]) - the matrix of data
        :param withTitles: (boolean) - True if titles should be saved
        :return:
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

    def import_Energy_Log(self, file):

        """
        This function imports the experimental energy log into arrays
        :param file: (String) - the file name
        :return: the arrays
        """

        print('Importing', file, '...')
        data = np.genfromtxt(file, skip_header = 1, delimiter=',')
        data = np.transpose(np.array(data))
        return data[0], data[1]

    def import_S_Energy_Log(self, file):

        """
        This function imports the simulated energy log into arrays
        :param file: (String) - the file name
        :return: the arrays
        """

        print('Importing', file, '...')
        data = np.genfromtxt(file, skip_header=1, delimiter=',')
        data = np.transpose(np.array(data))
        return data[0], data[1], data[2], data[3], data[4]

class Simulation_Simple:


    def __init__(self, CFRP_Fibre_Mass, CFRP_Fibre_Cp, CFRP_Resin_Mass, CFRP_Resin_Cp, Bleeder_Mass, Bleeder_Cp,
                 Tool_Plate_Mass, Tool_Plate_Cp, Vac_Bag_Mass, Vac_Bag_Cp, Autoclave_Mass, Autoclave_Cp, Autoclave_k,
                 Autoclave_Wall_Area, Autoclave_thickness, Autoclave, Gas_Mass, Gas_Cp, dt):

        """
        This class computes the heat consumed by the autoclave using only thermodynamic laws and averaged system properties
        :param CFRP_Fibre_Mass: the CFRP panel fibre mass
        :param CFRP_Fibre_Cp: the CFRP panel fibre specific heat
        :param CFRP_Resin_Mass: the CFRP panel resin mass
        :param CFRP_Resin_Cp: the CFRP panel resin specific heat
        :param Bleeder_Mass: the bleeder/breather mass
        :param Bleeder_Cp: the bleeder/breather specific heat
        :param Tool_Plate_Mass: the tool plate mass
        :param Tool_Plate_Cp: the tool plate specific heat
        :param Vac_Bag_Mass: the vacuum bag mass
        :param Vac_Bag_Cp: the vacuum bag specific heat
        :param Autoclave_Mass: the autoclave body mass
        :param Autoclave_Cp: the autoclave body specific heat
        :param Autoclave_k: the autoclave body conductivity
        :param Autoclave_Wall_Area: the autoclave body wall surface area
        :param Autoclave_thickness: the autoclave wall thickness
        :param Gas_Mass: the autoclave nitrogen mass
        :param Gas_Cp: the autoclave nitrogen specific heat
        :param dt: the time step
        """

        self.M_bleed = Bleeder_Mass
        self.Cp_bleed = Bleeder_Cp
        self.M_panel = CFRP_Fibre_Mass + CFRP_Resin_Mass
        self.Cp_panel = (CFRP_Fibre_Mass*CFRP_Fibre_Cp + CFRP_Resin_Mass*CFRP_Resin_Cp)/self.M_panel
        self.M_tool = Tool_Plate_Mass
        self.Cp_tool = Tool_Plate_Cp
        self.M_vac = Vac_Bag_Mass
        self.Cp_vac = Vac_Bag_Cp
        self.M_auto = Autoclave.Mass
        self.Cp_auto = Autoclave.Cp
        self.k_auto = Autoclave.k
        self.Area_auto = Autoclave.Area
        self.Thickness_auto = Autoclave.t_wall
        self.M_gas = Autoclave.V_gas*5.45
        self.Cp_gas = Gas_Cp
        self.Da = Autoclave.D_In
        self.V_gas = Autoclave.V_gas
        self.dt = dt
        self.efficiency = 0.9

    def simulate(self, time_log, Temp_log):

        """
        This function simulates the heat transfer through all components and with the environment
        :param time_log: the preset time log
        :param Temp_log: the preset temperature log
        :return:
        """

        t = np.array(time_log)
        T = np.array(Temp_log)

        Mb = self.M_bleed
        Ma = self.M_auto
        Mp = self.M_panel
        Mt = self.M_tool
        Cpb = self.Cp_bleed
        Cpa = self.Cp_auto
        Cpt = self.Cp_tool
        Cpp = self.Cp_panel
        Mv = self.M_vac
        Cpv = self.Cp_vac
        ka = self.k_auto
        Aa = self.Area_auto
        ta = self.Thickness_auto
        Mg = self.M_gas
        Cpg = self.Cp_gas
        Heat = [0]
        Q_loss = [0]
        Q_m_consumed = [0]
        Q_g_consumed = [0]
        Q_a_consumed = [0]
        for i in range(1, len(t)-1):
            Q_m_consumed.append(Q_m_consumed[i-1] + (Mb*Cpb+ Mp*Cpp + Mt*Cpt + Mv*Cpv)*max((T[i] - T[i-1]), 0))
            Q_g_consumed.append(Q_g_consumed[i-1] + Mg*Cpg*max((T[i] - T[i-1]), 0))
            Q_a_consumed.append(Q_a_consumed[i-1] + Ma*Cpa*max((T[i] - T[i-1]), 0))
            Heat.append(Heat[i-1] + (Mb*Cpb + Ma*Cpa + Mp*Cpp + Mt*Cpt + Mg*Cpg + Mv*Cpv)*max((T[i] - T[i-1]), 0))
            if T[i] >= T[i-1]:
                q_loss = ka * (T[i] - T[0]) * Aa / ta * (t[i] - t[i - 1])
                Heat[i] += q_loss
                Q_loss.append(q_loss + Q_loss[i - 1])
            else:
                Q_loss.append(Q_loss[i - 1])

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.suptitle("Heat Log")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel(r"Heat [kWh]")
        ax.plot(np.array(t[:-1]) / 60, np.array(Heat) / 3600000, color='black',
                label="Heat Consumed during Cure Cycle")
        ax.legend(loc=2, prop={'size': 7})
        ax.grid()
        plt.savefig("Heat_Simplified/" + "Heat_Log_Simplified.png", dpi=200)
        plt.close(fig)
        Q_a_consumed = np.array(Q_a_consumed)
        Q_g_consumed = np.array(Q_g_consumed)
        Q_m_consumed = np.array(Q_m_consumed)
        Q_loss = np.array(Q_loss)
        Heat = np.array(Heat)
        Matrix = np.transpose([t[:-1], T[:-1], Q_m_consumed/3600000, Q_g_consumed/3600000, Q_a_consumed/3600000, Q_loss/3600000, Heat/self.efficiency/3600000])
        Titles = ["Time [s]", "Temperature of Mould [C]", "Heat Transfered to Mould [kWh]", "Heat Transfered to Gas [kWh]", "Heat Transfered to Autoclave [kWh]", "Heat Lost To The Environment [kWh]", "Total Energy Consumed [kWh]"]
        self.saveCSV(name = "Energy_Log_Simple", Titles=Titles, Matrix=Matrix, withTitles=True)

    def compute(self, time_log, Temp_log):
        Q_a = self.M_auto*self.Cp_auto*(np.max(Temp_log) - Temp_log[0])
        Q_m = (self.M_vac*self.Cp_vac + self.M_panel*self.Cp_panel + self.M_tool*self.Cp_tool + self.M_bleed*self.Cp_bleed)*(np.max(Temp_log - Temp_log[0]))
        Q_g = self.M_gas*self.Cp_gas*(np.max(Temp_log) - Temp_log[0])

        T_integral = 0
        for i in range(1, len(time_log)):
            T_integral += (Temp_log[i] - Temp_log[0])*(time_log[i] - time_log[i-1])

        nu_g = 1.89e-5
        nu_a = 1.78e-5
        rho = 5.846
        g = 9.81
        beta = 0.0034
        k_G = 0.0257
        Cp_a = 1039
        hG = 0.023*k_G/self.Da*(rho*self.V_gas*self.Da/nu_g)**(0.8)*(nu_g*self.Cp_gas/k_G)**(0.4)
        hA = 4.5
        Q_loss = T_integral/(1/hA + 1/hG + self.Thickness_auto/self.Area_auto/self.k_auto)
        error = 1
        while error > 1e-5:
            delta_T = Q_loss/hA/time_log[-1]
            hA_n = 0.53*k_G/self.Da*((g*beta*self.Da**3*delta_T*rho**2/nu_a**2)*(nu_a*Cp_a/k_G))**(0.25)
            error = hA_n - hA
            hA = hA_n
            Q_loss = T_integral/(1/hA + 1/hG + self.Thickness_auto/self.Area_auto/self.k_auto)
        return (Q_a + Q_g + Q_m + Q_loss)/10**6/3.6


    def saveCSV(self, name, Titles, Matrix, withTitles):

        """
                This function saves arrays to a .csv file
                :param name: (String) - the file name
                :param Titles: (String[]) - the titles of the columns
                :param Matrix: (float[][]) - the matrix of data
                :param withTitles: (boolean) - True if titles should be saved
                :return:
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

