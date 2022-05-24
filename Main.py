# This is the Main module. It initializes and calls all the classes from the other modules in the script
#
# 4 separate simulations are run by Main in this order:
# A MRT Lattice Boltzmann simulation of the autoclave Nitrogen gas --> returns the temperature variations in the gas at the walls of the autoclave.
# A solid heat conduction simulation for the autoclave steel body + insulation + rudimentary CFRP mould heat transfer (using the temp. variations of the previous sim) --> returns the energy consumption components (high accuracy)
# (Optional) A solid heat conduction simulation for the CFRP mould (in case more accuracy is desired for this component) --> returns the energy consumed for heating the mould
# (Optional) A simple heat transfer simulation for the entire autoclave, using thermodynamic laws and assuming constant temperature everywhere --> returns the energy consumption components (low accuracy)
#

# Import all required libraries

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import Materials, Simulation_Tools, Lattice_Boltzmann, Import_Tools, Plotter

factor = 7 / 7

# Initialize the materials in the CFRP mould + the autoclave gas (Nitrogen)

Carbon_fibre = Materials.Fibre(E_modulus=228 * 10 ** 9, Therm_Cond=170, Mass=0.48 * factor, Specific_Heat=750)
Resin = Materials.Resin(E_modulus=3.81 * 10 ** 9, Therm_Cond=0.1, Mass=0.205 * factor, Specific_Heat=1850)
Gas = Materials.Gas(Molar_Mass=28, Specific_Heat=1039, Thermal_Conductivity=0.0257, Mass=31.98 * factor)

# Initialize the mould assembly (important) components (CFRP panel, Bleeder, Tool plate) and the autoclave body

panel_thickness = 2 / 1000  # thickness of CFRP panel

orientations = [0, 90, 0, 90, 0, 90, 0, 90, 0, 90, 0, 90]  # array of orientations of plies in the panel [deg]

# Each object is initialized 3 times: an object with the minimum estimated properties, an object with the expected properties and an object with the/ maximum estimated properties


CFRP_Panel = Materials.CFRP_Panel(dimensions=[0.45, 0.45, panel_thickness], fibre=Carbon_fibre, resin=Resin,
                                  thicknesses=np.ones(len(orientations)) * panel_thickness / len(orientations), orientations=orientations,
                                  fibre_fractions=np.ones(len(orientations)) * 0.6)  # the CFRP panel object
Bleeder = Materials.Bleeder(dimensions=[0.6, 0.6, 7 / 10000 + panel_thickness], therm_cond=0.5, specific_heat=1350, mass=0.242 * factor)  # the bleeder object
Tool = Materials.Tool_Plate(dimensions=[0.8, 0.8, 1 / 1000], therm_cond=200, specific_heat=921, mass=12.15 * factor)  # the tool plate object
Autoclave = Materials.Autoclave(Length_Outer=2.4 * factor ** (1 / 3), Length_Inner=2.3 * factor ** (1 / 3), Diameter_Outer=1.5 * factor ** (1 / 3),
                                Diameter_Inner=1.20 * factor ** (1 / 3), Area=13.838 * factor ** (2 / 3), Cap_Radius=0.75 * factor ** (1 / 3), Vol_tot=7 * factor,
                                Vol_Gas=4 * factor, thickness=0.14 * factor ** (1 / 3), Mold_Dist=0.7, Mold_Height=-0.8,
                                Mold_Dimensions=[0.8 * factor ** (1 / 3), 8 / 100 * factor ** (1 / 3)], Mass=1602 * factor, Therm_cond=0.85,
                                Specific_heat=485)  # the autoclave body object

CFRP_Panel_min = Materials.CFRP_Panel(dimensions=[0.45, 0.45, panel_thickness], fibre=Carbon_fibre, resin=Resin,
                                      thicknesses=np.ones(len(orientations)) * panel_thickness / len(orientations), orientations=orientations,
                                      fibre_fractions=np.ones(len(orientations)) * 0.6)  # the CFRP panel object with minimum estimates of the properties
Bleeder_min = Materials.Bleeder(dimensions=[0.6, 0.6, 7 / 10000 + panel_thickness], therm_cond=0.5 * 0.9, specific_heat=1350 * 0.9,
                                mass=0.242)  # the bleeder object with minimum estimates of the properties
Tool_min = Materials.Tool_Plate(dimensions=[0.8, 0.8, 1 / 1000], therm_cond=200 * 0.9, specific_heat=921 * 0.9,
                                mass=12.15)  # the tool plate object with minimum estimates of the properties
Autoclave_min = Materials.Autoclave(Length_Outer=2.4, Length_Inner=2.3, Diameter_Outer=1.5, Diameter_Inner=1.28, Area=1.9792, Cap_Radius=0.75, Vol_tot=7, Vol_Gas=4,
                                    thickness=0.14, Mold_Dist=0.7, Mold_Height=-0.8, Mold_Dimensions=[0.8, 8 / 100], Mass=1602, Therm_cond=0.85 * 0.9,
                                    Specific_heat=485 * 0.9)  # the autoclave body object with minimum estimates of the properties

CFRP_Panel_max = Materials.CFRP_Panel(dimensions=[0.45, 0.45, panel_thickness], fibre=Carbon_fibre, resin=Resin,
                                      thicknesses=np.ones(len(orientations)) * panel_thickness / len(orientations), orientations=orientations,
                                      fibre_fractions=np.ones(len(orientations)) * 0.6)
Bleeder_max = Materials.Bleeder(dimensions=[0.6, 0.6, 7 / 10000 + panel_thickness], therm_cond=0.5 * 1.1, specific_heat=1350 * 1.1, mass=0.242)
Tool_max = Materials.Tool_Plate(dimensions=[0.8, 0.8, 1 / 1000], therm_cond=200 * 1.1, specific_heat=921 * 1.1, mass=12.15)
Autoclave_max = Materials.Autoclave(Length_Outer=2.4, Length_Inner=2.3, Diameter_Outer=1.5, Diameter_Inner=1.28, Area=1.9792, Cap_Radius=0.75, Vol_tot=7, Vol_Gas=4,
                                    thickness=0.14, Mold_Dist=0.7, Mold_Height=-0.8, Mold_Dimensions=[0.8, 8 / 100], Mass=1602, Therm_cond=0.85 * 1.1,
                                    Specific_heat=485 * 1.1)

Fans = Materials.Fans(Number=0, Radius=0.5, BladeNumber=4, RPM=4300, AirSpeed=6, Blade_Width=10 / 100)

# Import the experimental temperature log

time_log, temp2, Temp_log = Import_Tools.import_Temp_Cycle('Data/Temperature cycle 450 X 450.csv')

# Initialize the Lattice Boltzmann simulation and simulate the steady-state temperature field

LBM_sim = Lattice_Boltzmann.LBM_Simulation(Autoclave=Autoclave, ds=0.05, vel=6, viscosity=0.0023, density=1.225, width=1.65 * factor**(1/3), length=4.3 * factor**(1/3))  # Lattice Boltzmann simulation object
#Temp_N, Temp_S, Temp_E, Temp_W, Temp_MN, Temp_MS, Temp_ME, Temp_MW, Mold_Temp = LBM_sim.simulate() # simulate and save the temperature variations (Comment if already simulated)
Temp_N, Temp_S, Temp_E, Temp_W, Temp_MN, Temp_MS, Temp_ME, Temp_MW, Mold_Temp = LBM_sim.load_Temp_Gradient(
    'CSV_Data/Temperature_Gradient.csv')  # alternatively, import them from a pre-simulated file

# create the simulation objects for the advanced CFRP mould simulation. 3 objects are created with the min, max and expected component properties

sim = Simulation_Tools.Simulation(CFRP_Panel=CFRP_Panel, Bleeder=Bleeder, Tool_Plate=Tool, Gas=Gas, Autoclave=Autoclave, Fans=Fans, dx=0.0075, dz=0.00015, dx_a=0.02, dt=0.1)
sim_min = Simulation_Tools.Simulation(CFRP_Panel=CFRP_Panel_min, Bleeder=Bleeder_min, Tool_Plate=Tool_min, Gas=Gas, Autoclave=Autoclave_min, Fans=Fans, dx=0.0075, dz=0.00015,
                                      dx_a=0.02, dt=0.1)
sim_max = Simulation_Tools.Simulation(CFRP_Panel=CFRP_Panel_max, Bleeder=Bleeder_max, Tool_Plate=Tool_max, Gas=Gas, Autoclave=Autoclave_max, Fans=Fans, dx=0.0075, dz=0.00015,
                                      dx_a=0.02, dt=0.1)

# simulate and save all the autoclave body heat transfers for the 3 min. max and expected object properties. (Uncomment to do this)

# (Comment if already simulated)
# time, Total_Heat, Autoclave_Heat, Mould_Heat, Lost_Heat = sim.sim_Auto_Temp(Temp_log=Temp_log[1:], time_log=time_log[1:], Wall_Temperatures= [Temp_N, Temp_S, Temp_E, Temp_W], Temp_0= Mold_Temp, name='Autoclave_Energy_Data.csv')
# time_min, Total_Heat_min, Autoclave_Heat_min, Mould_Heat_min, Lost_Heat_min = sim_min.sim_Auto_Temp(Temp_log=Temp_log[1:], time_log=time_log[1:], Wall_Temperatures= [Temp_N, Temp_S, Temp_E, Temp_W], Temp_0= Mold_Temp, name='Autoclave_Energy_min_Data.csv')
# time_max, Total_Heat_max, Autoclave_Heat_max, Mould_Heat_max, Lost_Heat_max = sim_max.sim_Auto_Temp(Temp_log=Temp_log[1:], time_log=time_log[1:], Wall_Temperatures= [Temp_N, Temp_S, Temp_E, Temp_W], Temp_0= Mold_Temp, name='Autoclave_Energy_max_Data.csv')

# names = ['CSV_Data/Autoclave_Energy_Data_Scaled20.csv.csv',
#          'CSV_Data/Autoclave_Energy_Data_Scaled21.csv.csv',
#          'CSV_Data/Autoclave_Energy_Data_Scaled22.csv.csv',
#          'CSV_Data/Autoclave_Energy_Data_Scaled23.csv.csv',
#          'CSV_Data/Autoclave_Energy_Data_Scaled24.csv.csv',
#          'CSV_Data/Autoclave_Energy_Data_Scaled25.csv.csv']
# Plotter.plotEnergies(names, sim, time_log,  Temp_log, 'Data/SensAnLog.csv')
names2 = ['CSV_Data/Autoclave_Energy_Data_Scaled0.csv.csv',
          'CSV_Data/Autoclave_Energy_Data_Scaled1.csv.csv',
          'CSV_Data/Autoclave_Energy_Data_Scaled2.csv.csv',
          'CSV_Data/Autoclave_Energy_Data_Scaled3.csv.csv',
          'CSV_Data/Autoclave_Energy_Data_Scaled4.csv.csv',
          'CSV_Data/Autoclave_Energy_Data_Scaled5.csv.csv']
Plotter.plotEnergies(names2, sim, time_log, Temp_log, 'Data/AutAnLog.csv')

# (optional) perform advanced heat transfer simulation for the CFRP mould assembly, for the expected component properties (Uncomment to do this)

# sim.sim_CFRP_Temp(Temp_log=Temp_log[1:], time_log=time_log[1:]) # (comment if already simulated)

# import the saved energy transfer components for all 3 autoclave body simulations (only if already simulated once)

e_time, e_Energy_Log = sim.import_Energy_Log('Data\Energy_Log.csv')
s_time, s_A_Heat, s_L_Heat, s_M_Heat, s_Energy_Log = sim.import_S_Energy_Log('CSV_Data\Autoclave_Energy_Data.csv.csv')
ss_time, ss_Energy_Log = np.genfromtxt('Data/CureAnLog.csv', delimiter=',').T[0:2]
s_time_min, s_A_Heat_min, s_L_Heat_min, s_M_Heat_min, s_Energy_Log_min = sim.import_S_Energy_Log('CSV_Data\Autoclave_Energy_min_Data.csv.csv')
s_time_max, s_A_Heat_max, s_L_Heat_max, s_M_Heat_max, s_Energy_Log_max = sim.import_S_Energy_Log( 'CSV_Data\Autoclave_Energy_max_Data.csv.csv')

# create a discrete time axis and linearly interpolate all the energy logs to fit it

# time = np.arange(0, e_time[-1], e_time[1] - e_time[0])
time = s_time
s_Energy_Log = interpolate.interp1d(s_time, s_Energy_Log, 'linear', fill_value='extrapolate')(time)
s_A_Heat = interpolate.interp1d(s_time, s_A_Heat, 'linear', fill_value='extrapolate')(time)
s_M_Heat = interpolate.interp1d(s_time, s_M_Heat, 'linear', fill_value='extrapolate')(time)
s_L_Heat = interpolate.interp1d(s_time, s_L_Heat, 'linear', fill_value='extrapolate')(time)
s_Energy_Log_min = interpolate.interp1d(s_time_min, s_Energy_Log_min, 'linear', fill_value='extrapolate')(time)
s_A_Heat_min = interpolate.interp1d(s_time_min, s_A_Heat_min, 'linear', fill_value='extrapolate')(time)
s_M_Heat_min = interpolate.interp1d(s_time_min, s_M_Heat_min, 'linear', fill_value='extrapolate')(time)
s_L_Heat_min = interpolate.interp1d(s_time_min, s_L_Heat_min, 'linear', fill_value='extrapolate')(time)
s_Energy_Log_max = interpolate.interp1d(s_time_max, s_Energy_Log_max, 'linear', fill_value='extrapolate')(time)
s_A_Heat_max = interpolate.interp1d(s_time_max, s_A_Heat_max, 'linear', fill_value='extrapolate')(time)
s_M_Heat_max = interpolate.interp1d(s_time_max, s_M_Heat_max, 'linear', fill_value='extrapolate')(time)
s_L_Heat_max = interpolate.interp1d(s_time_max, s_L_Heat_max, 'linear', fill_value='extrapolate')(time)
e_Energy_Log = interpolate.interp1d(e_time, e_Energy_Log, 'linear', fill_value='extrapolate')(time)
ss_Energy_Log = interpolate.interp1d(ss_time, ss_Energy_Log, 'linear', fill_value='extrapolate')(time)
mean_e = np.average(e_Energy_Log)  # compute the mean energy for the experimental energy log

R_2 = 1 - np.sum((e_Energy_Log - s_Energy_Log) ** 2) / np.sum(
    (e_Energy_Log - mean_e) ** 2)  # compute the R^2 value between the somulated expected energy log and the experimental one

print("The R^2 value of the simulation is:", np.round(R_2, 3))  # print the R^2 value

# plot the total simulated energy log against the experimental one. For the simulated log, 3 lines are plotted for the min. max and expected input values

fig = plt.figure()
ax = fig.add_subplot()
plt.suptitle("Simulated Autoclave Energy Consumption")

ax.plot(np.array(time_log) / 60, Temp_log, color='black', label='Cure Cycle Temperature Log', linewidth=1)

ax.set_xlabel("Time [min]")
ax.set_ylabel(r"Temperature [$^{\circ}$C]")
ax.legend(loc=2, prop={'size': 7})
ax.grid()
plt.savefig("Results/" + "Temperature_Log" + ".png", dpi=200)
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot()
plt.suptitle("Simulated Autoclave Energy Consumption")

ax.plot(time / 60, e_Energy_Log, color='red', label='Experimental Energy Log', linewidth=1)
# ax.plot(time/60, s_Energy_Log_max, color = 'black', label = 'Simulated Energy Log Upper Estimate', linestyle='--', linewidth = 0.7)
# ax.plot(time/60, s_Energy_Log_min, color = 'black', label = 'Simulated Energy Log Lower Estimate', linestyle='--', linewidth = 0.7)
ax.plot(time / 60, s_Energy_Log, color='black', label='Simulated Lattice Boltzmann Energy Log', linestyle='-', linewidth=1)
ax.plot(time / 60, ss_Energy_Log * 20.3 / 23.6, color='black', label='Simulated Analytical Model Energy Log', linestyle='--', linewidth=1)
ax.set_xlabel("Time [min]")
ax.set_ylabel(r"Energy Log [-]")
ax.legend(loc=2, prop={'size': 7})
ax.grid()
plt.savefig("Results/" + "Simulated_Autoclave_Energy_Consumption" + ".png", dpi=200)
plt.close(fig)

# plot all the simulated energy transfer components in one graph. Each component will be plotted for the minimum, maximum and expected input values

# fig = plt.figure()
# ax = fig.add_subplot()
# plt.suptitle("Simulated Heat Log")
# ax.set_xlabel("Time [min]")
# ax.set_ylabel(r"Heat [kWh]")
# ax.plot(np.array(time) / 60, np.array(ss_Energy_Log), linestyle='-', color='black',
#         label="Total Heat Consumed")
# ax.plot(np.array(time) / 60, np.array(ss_A_Heat), linestyle='-', color='red',
#         label="Heat Transfered to Autoclave Body")
# ax.plot(np.array(time) / 60, np.array(ss_L_Heat), linestyle='-', color='blue',
#         label="Heat Lost To The Environment")
# ax.plot(np.array(time) / 60, np.array(ss_M_Heat), linestyle='-', color='green',
#         label="Heat Transferred to Mould Assembly")
# ax.plot(np.array(time) / 60, np.array(s_Energy_Log_min), linestyle='--', color='black', linewidth=0.7,
#                 label="Total Heat Consumed Lower Estimate")
# ax.plot(np.array(time) / 60, np.array(s_A_Heat_min), linestyle='--', color='red', linewidth=0.7,
#                 label="Heat Transfered to Autoclave Body Lower Estimate")
# ax.plot(np.array(time) / 60, np.array(s_L_Heat_min), linestyle='--', color='blue', linewidth=0.7,
#                 label="Heat Lost To The Environment Lower Estimate")
# ax.plot(np.array(time) / 60, np.array(s_M_Heat_min), linestyle='--', color='green', linewidth=0.7,
#                 label="Heat Transferred to Mould Assembly Lower Estimate")
# ax.plot(np.array(time) / 60, np.array(s_Energy_Log_max), linestyle='--', color='black', linewidth=0.7,
#                 label="Total Heat Consumed Higher Estimate")
# ax.plot(np.array(time) / 60, np.array(s_A_Heat_max), linestyle='--', color='red', linewidth=0.7,
#                 label="Heat Transfered to Autoclave Body Higher Estimate")
# ax.plot(np.array(time) / 60, np.array(s_L_Heat_max), linestyle='--', color='blue', linewidth=0.7,
#                 label="Heat Lost To The Environment Higher Estimate")
# ax.plot(np.array(time) / 60, np.array(s_M_Heat_max), linestyle='--', color='green', linewidth=0.7,
#                 label="Heat Transferred to Mould Assembly Higher Estimate")
ax.legend(loc=2, prop={'size': 5})
ax.grid()
plt.savefig("Results/" + "Heat_Log" + ".png", dpi=200)
plt.close(fig)

# # # create the simple autoclave heat transfer simulation object
# #
# # sim_simple = Simulation_Tools.Simulation_Simple(CFRP_Fibre_Mass=0.48, CFRP_Fibre_Cp=750, CFRP_Resin_Mass=0.205, CFRP_Resin_Cp=1850, Bleeder_Mass=0.242, Bleeder_Cp=1350,
# #                                                 Tool_Plate_Mass=12.15, Tool_Plate_Cp=921.1, Vac_Bag_Mass = 0.059, Vac_Bag_Cp = 1670, Autoclave_Mass=1102, Autoclave_Cp=510.79,
# #                                                 Autoclave_k=14, Autoclave_thickness=0.12, Autoclave_Wall_Area=1.9792, Gas_Mass=31.98, Gas_Cp=1039, dt=5)
# #
# # # simulate the energy consumed during a cycle assuming constant temperature
# #
# # sim_simple.simulate(time_log, Temp_log)
