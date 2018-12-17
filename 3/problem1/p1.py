# ML HW3 problem1
import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from sympy import *
import matplotlib.pyplot as plt
import numpy.linalg
import math


train_data_x = []
train_data_t = []
test_data_x = []
test_data_t = []


with open('gp.csv', newline='') as rowfile:
    rows = csv.reader(rowfile)
    i = 0
    for row in rows:
        if i < 60:
            train_data_x.append(float(row[0]))
            train_data_t.append(float(row[1]))
        else:
            test_data_x.append(float(row[0]))
            test_data_t.append(float(row[1]))
        i+=1