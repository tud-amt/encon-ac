# This module contains the classes for the autoclave gas, the CFRP panel materials, the mould assembly components and autoclave geometry and properties
#



import numpy as np

class Gas:

    def __init__(self, Molar_Mass, Specific_Heat, Thermal_Conductivity, Mass):
        self.Cp = Specific_Heat
        self.Mu = Molar_Mass
        self.k = Thermal_Conductivity
        self.Mass = Mass

class Fibre:

    def __init__(self, E_modulus, Therm_Cond, Mass, Specific_Heat):
        self.E_mod = E_modulus
        self.Therm_Cond = Therm_Cond
        self.Mass = Mass
        self.Cp = Specific_Heat

class Resin:

    def __init__(self, E_modulus, Therm_Cond, Mass, Specific_Heat):
        self.E_mod = E_modulus
        self.Therm_Cond = Therm_Cond
        self.Mass = Mass
        self.Cp = Specific_Heat

class CFRP_Panel:

    def __init__(self, dimensions, fibre, resin, thicknesses, orientations, fibre_fractions):
        self.Length = dimensions[0]
        self.Height = dimensions[2]
        self.Width = dimensions[1]
        self.Fibre = fibre
        self.Resin = resin
        self.Mass = self.Fibre.Mass + self.Resin.Mass
        self.Thicknesses = thicknesses
        self.Orientations = np.array(orientations)*np.pi/180
        self.Fibre_fractions = fibre_fractions

class Bleeder:

    def __init__(self, dimensions, therm_cond, specific_heat, mass):
        self.Length = dimensions[0]
        self.Height = dimensions[2]
        self.Width = dimensions[1]
        self.Cp = specific_heat
        self.k = therm_cond
        self.Mass = mass
        self.Density = mass/dimensions[0]/dimensions[1]/dimensions[2]


class Tool_Plate:

    def __init__(self, dimensions, therm_cond, specific_heat, mass):
        self.Length = dimensions[0]
        self.Height = dimensions[2]
        self.Width = dimensions[1]
        self.Cp = specific_heat
        self.k = therm_cond
        self.Mass = mass
        self.Density = mass/dimensions[0]/dimensions[1]/dimensions[2]

class Autoclave:

    def __init__(self, Length_Outer, Length_Inner, Diameter_Outer, Diameter_Inner, Cap_Radius, Area, Vol_Gas, Vol_tot, thickness, Mold_Dist, Mold_Height, Mold_Dimensions, Mass, Therm_cond, Specific_heat):
        self.L_Out = Length_Outer
        self.L_In = Length_Inner
        self.D_Out = Diameter_Outer
        self.D_In = Diameter_Inner
        self.C_Rad = Cap_Radius
        self.M_Dist = Mold_Dist
        self.M_Height = Mold_Height
        self.Area = Area
        self.Mold_Dim = Mold_Dimensions
        self.k = Therm_cond
        self.Cp = Specific_heat
        self.Mass = Mass
        self.rho = Mass/(Vol_tot - Vol_Gas)
        self.A_gas = 4/np.pi/(Diameter_Outer - 2*thickness) * Vol_Gas
        self.A_tot = 4/np.pi/Diameter_Outer * Vol_tot
        self.t_wall = thickness
        self.V_gas = Vol_Gas
        self.Vol_tot = Vol_tot

class Fans:

    def __init__(self, Number, Radius, BladeNumber, RPM, AirSpeed, Blade_Width):

        self.Number = Number
        self.R = Radius
        self.BladeNumber = BladeNumber
        self.omega = RPM / 60 * 2 * np.pi
        self.V = AirSpeed
        self.C = Blade_Width
        self.AoA = np.arctan(self.V / (self.omega * self.R / 2))
        self.CL = 2 * np.pi * self.AoA
        self.CD0 = 0.0012
        self.e = 0.85
        self.A = self.R / self.C
        self.S = self.R * self.C
        self.rho = 1.225 * 6.3
        self.BNumber = BladeNumber
        self.Drag = (self.CD0 + self.CL**2 / self.A / self.e / np.pi) * 0.5 * self.rho * (self.omega * self.R / 2)**2 * self.S
        self.Power = self.Drag * self.BNumber * self.Number * self.omega * self.R / 2


