import numpy as np

def MAD(m1, m2): # Mean Absolute Deviation
    return np.mean(np.absolute(np.subtract(m1, m2)))
def MSE(m1, m2): # Mean Square Error
    return np.mean(np.square(np.subtract(m1, m2)))
def SAD(m1, m2): # Sum of Absolute Difference
    return np.sum(np.absolute(np.subtract(m1, m2)))
def SSD(m1, m2): # Sum of Square Deviation
    return np.sum(np.square(np.subtract(m1, m2)))