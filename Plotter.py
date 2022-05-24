import numpy as np
import matplotlib.pylab as plt
import Import_Tools


def import_to_plot(files):
    Data = []
    Titles = ['Dwell Time: 1.5h', 'Dwell Time: 1h']
    for file in files:
        data = np.array(np.genfromtxt(file, skip_header=1, delimiter=',')).T
        Data.append([data[0], data[1], data[3]])

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.suptitle(r"CFRP Mould Average Temperature for 120 $^{\circ}$ C and 135 $^{\circ}$ C cycles")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel(r"Temperature [$^{\circ}$ C]")
    for i in range(len(Data)):
        cycle = Data[i]
        ax.plot(np.array(cycle[0]) / 60, cycle[2], linestyle='-', color='black',
                label=Titles[i] + " Cycle Temperature")
        ax.plot(np.array(cycle[0]) / 60, cycle[1], linestyle='--',
                label=Titles[i] + " Mould Temperature")
        ax.legend(loc=1, prop={'size': 7})
    ax.grid()
    plt.savefig("Results/Temp_Cycles.png", dpi=1500)
    plt.close(fig)


def plotEnergies(names, sim, time, Temp, simpleFile):
    fig2 = plt.figure()
    ax = fig2.add_subplot()
    plt.suptitle(r"Autoclave Total Energy Consumption vs. time")

    ax.set_xlabel(r"Total Autoclave Volume [$m^3$]")
    ax.set_ylabel(r"Total Energy Consumption [kWh]", color='blue')
    ax.set_yscale('linear')
    ax.grid()
    ax.tick_params(axis='y', labelcolor='blue')
    # ax.set_ylim([0.1, 10000])
    # ax2 = ax.twinx()
    # ax2.set_ylabel(r"Temperature [$^{\circ}$ C]", color='gray')
    # ax2.plot(time / 60, Temp, linewidth=0.8, color='black', linestyle='--')
    # ax2.tick_params(axis='y', labelcolor='gray')

    volumes = [5, 84, 163, 242, 321, 400]

    simpleData = np.genfromtxt(simpleFile, delimiter=',', skip_header=1).T
    print(simpleData)
    time = simpleData[0]
    simpleEnergy = simpleData[1:]
    boltzmanEnergy = []

    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))

    for i in range(len(names)):
        s_time, s_A_Heat, s_L_Heat, s_M_Heat, s_Energy_Log = sim.import_S_Energy_Log(names[i])
        # ax.plot(s_time / 60, s_Energy_Log, linewidth=2, color=colors[i], label='Total Volume =' + str(volumes[i]) + r'$m^3$')
        # ax.plot(time / 60, simpleEnergy[i], linewidth = 1, color=colors[i], label='Analytical Total Volume =' + str(volumes[i]) + r'$m^3$', linestyle='--')
        boltzmanEnergy.append(s_Energy_Log[-1])
    boltzmanEnergy = np.array(boltzmanEnergy)
    ax.plot(volumes, boltzmanEnergy, label='LBM Method', color = 'black', marker='o')
    ax.plot(volumes, simpleEnergy[:,-1], label='Analytical Method', color='blue', marker='o')

    ax.legend(fontsize=6)
    plt.savefig("AutoclaveScalingEnergy.png", dpi=1000)
    plt.close(fig2)


files = ['CSV_Data\CFRP_Temp_Log_1.csv', 'CSV_Data\CFRP_Temp_Log_2.csv']

import_to_plot(files)

Volume = np.array([5, 84, 163, 242, 321, 400])
Auto_Heat = np.array([16.0647, 144.9424, 279.633, 400.717, 552.6815, 665.7014])
Mold_Heat = np.array([0.33144, 2.706, 5.18573, 7.4342, 10.30868, 12.3811])
Wall_Heat = np.array([6.08651, 19.4034, 25.399, 29.59546, 34.1579, 36.7156])
Total_Heat = Auto_Heat + Wall_Heat + Mold_Heat

Volume2 = np.array([5, 84, 163, 242, 321, 400])
Auto_Heat2 = np.array([16.4635, 230.095, 448.7938, 665.8997, 932.1664, 1121.826])
Mold_Heat2 = np.array([0.4255, 4.29744, 8.411, 12.3178, 17.3235, 21.261])
Wall_Heat2 = np.array([24.9997, 105.00063, 142.6137, 174.346, 193.0303, 219.7341])
Total_Heat2 = Mold_Heat2 + Auto_Heat2 + Wall_Heat2

fig2 = plt.figure()
ax = fig2.add_subplot()
plt.suptitle(r"Autoclave Energy Consumption vs. Total Volume")

# ax.plot(Volume2, Total_Heat2, linewidth=1, label='Total Heat Consumption')
# ax.fill_between(Volume2, Total_Heat2)
ax.plot(Volume, Mold_Heat + Wall_Heat + Auto_Heat, linewidth=1, label='Mould Assembly Heat Consumption')
ax.fill_between(Volume, Mold_Heat + Wall_Heat + Auto_Heat)
ax.plot(Volume, Auto_Heat + Wall_Heat, linewidth=1, label='Autoclave Body Heat Consumption')
ax.fill_between(Volume, Auto_Heat + Wall_Heat)
ax.plot(Volume, Wall_Heat, linewidth=1, label='Wall Heat Loss')
ax.fill_between(Volume, Wall_Heat)
ax.set_xlabel(r"Total Internal Volume [$m^3$]")
ax.set_ylabel(r"Energy Consumption per Cycle [kWh]")
ax.set_yscale('linear')
# ax.set_ylim([0.1, 10000])
ax.legend(fontsize=7)
ax.grid()
plt.savefig("AutoclaveScalingEnergy.png", dpi=1000)
plt.close(fig2)

fig2 = plt.figure()
ax = fig2.add_subplot()
plt.suptitle(r"Autoclave Energy Consumption vs. Total Volume")

# ax.plot(Volume2, Total_Heat2, linewidth=1, label='Total Heat Consumption')
# ax.fill_between(Volume2, Total_Heat2)
ax.plot(Volume2, Mold_Heat2 + Wall_Heat2 + Auto_Heat2, linewidth=1, label='Mould Assembly Heat Consumption')
ax.fill_between(Volume2, Mold_Heat2 + Wall_Heat2 + Auto_Heat2)
ax.plot(Volume2, Auto_Heat2 + Wall_Heat2, linewidth=1, label='Autoclave Body Heat Consumption')
ax.fill_between(Volume2, Auto_Heat2 + Wall_Heat2)
ax.plot(Volume2, Wall_Heat2, linewidth=1, label='Wall Heat Loss')
ax.fill_between(Volume2, Wall_Heat2)

ax.set_xlabel(r"Total Internal Volume [$m^3$]")
ax.set_ylabel(r"Energy Consumption per Cycle [kWh]")
ax.set_yscale('linear')
# ax.set_ylim([0.1, 10000])
ax.legend(fontsize=7)
ax.grid()
plt.savefig("AutoclaveScalingEnergy_2.png", dpi=1000)
plt.close(fig2)

time_log, press, temp0 = Import_Tools.import_Temp_Cycle('Data/Temperature cycle 450 X 450.csv')
pressure, time_log1, temp1 = Import_Tools.import_Temp_Cycle('Data/TempCycle1.csv')
time_log2, temp2, dump = Import_Tools.import_Temp_Cycle('Data/TempCycle2.csv')

time1, Cycle1, time2, Cycle2, time3, Cycle3 = np.genfromtxt('Data/CureAnLog.csv', delimiter=',').T
times1, dump1, dump2, dump3,  Cycles1, dump4 = np.genfromtxt('CSV_Data/Autoclave_Energy_Data_Temp0.csv.csv', delimiter=',', skip_header=1).T
times2, dump1, dump2, dump3,  Cycles2, dump4 = np.genfromtxt('CSV_Data/Autoclave_Energy_Data_Temp1.csv.csv', delimiter=',', skip_header=1).T
times3, dump1, dump2, dump3,  Cycles3, dump4 = np.genfromtxt('CSV_Data/Autoclave_Energy_Data_Temp2.csv.csv', delimiter=',', skip_header=1).T

fig2 = plt.figure()
ax = fig2.add_subplot()
ax2 = ax.twinx()
ax2.set_ylim(0, 400)
ax2.set_ylabel(r"Preset Temperature [$^{\circ} C$]")
plt.suptitle(r"Autoclave Energy Consumption for Different Cycles")

# ax.plot(Volume2, Total_Heat2, linewidth=1, label='Total Heat Consumption')
# ax.fill_between(Volume2, Total_Heat2)
ax.plot(time1[time1!=0] / 60, Cycle1[time1!=0]* 20.3 / 23.6, linewidth=1, linestyle='--', color='blue', label=r'Thermoset 120 $^{\circ} C$ - Analytical')
ax.plot(times1 / 60, Cycles1, linewidth=1, linestyle='-',color='blue', label=r'Thermoset 120 $^{\circ} C$ - Simulated')
ax2.plot(time_log / 60, temp0, linewidth=0.5, linestyle='--', color='blue')
ax.plot(time2[time2!=0] / 60, Cycle2[time2!=0] * 20.3 / 23.6, linewidth=1, linestyle='--',color='black', label=r'Thermoset 180 $^{\circ} C$ - Analytical')
ax.plot(times2 / 60, Cycles2, linewidth=1, linestyle='-',color='black', label=r'Thermoset 180 $^{\circ} C$ - Simulated')
ax2.plot(time_log1 / 60, temp1, linewidth=0.5, linestyle='--', color='black')
ax.plot(time3[time3!=0] / 60, Cycle3[time3!=0] * 20.3 / 23.6, linewidth=1, linestyle='--',color='green', label=r'Thermoplastic 380 $^{\circ} C$ - Analytical')
ax.plot(times3 / 60, Cycles3 , linewidth=1, linestyle='-',color='green', label=r'Thermoplastic 380 $^{\circ} C$ - Simulated')
ax2.plot(time_log2 / 60, temp2, linewidth=0.5, linestyle='--', color='green')
ax.set_xlabel(r"Time [$min$]")
ax.set_ylabel(r"Energy Consumption per Cycle [kWh]")
ax.set_yscale('linear')
# ax.set_ylim([0.1, 10000])
ax.legend(fontsize=7)
ax.grid()
plt.savefig("AutoclaveCycleAnalysis.png", dpi=1000)
plt.close(fig2)