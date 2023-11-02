import numpy as np

def MAD(m1, m2): # Mean Absolute Deviation
    return np.absolute(np.subtract(m1, m2)).mean()
def MSE(m1, m2): # Mean Square Error
    return np.square(np.subtract(m1, m2)).mean()
def SAD(m1, m2): # Sum of Absolute Difference
    return np.absolute(np.subtract(m1, m2))
def SSD(m1, m2): # Sum of Square Deviation
    return np.square(np.subtract(m1, m2))