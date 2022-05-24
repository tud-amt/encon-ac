import numpy as np



def import_Temp_Cycle(filename):

    file = open(filename, 'r+')
    data = np.genfromtxt(file, skip_header = 2, delimiter = ',')
    data = np.array(data)
    data = data.transpose()
    return np.array(data[0]), np.array(data[1]), np.array(data[2])