import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from sympy import *
import matplotlib.pyplot as plt
import numpy.linalg
import math


with open('train.csv', newline='') as trainfile:
    rows = csv.reader(trainfile)
    for row in rows:
        Raw_data_x.append(float(row[0]))
        Raw_data_t.append(float(row[1]))