'''Microsoft Cloud Society Project'''

'''Project details: A python program that takes input of years and population from user and shows a graph showing its regression line with its projection.
It also generates 10 random numbers for both x and y axis.'''

import seaborn as sns; sns.set()
import numpy as np
import matplotlib.pyplot as plt




#Below is the code for entering your own data if you do not want to import data. You only need to remove quotations.

''' m=yearsBase=list(input("Enter value(s) of x axis i.e Years:"))
t=meanBase=list(input("Enter value(s) of y axis i.e Mean Temperature:")) '''

import pandas as pd
'''Using Panda'''

#yearsBase, meanBase =pd.read_csv (r'C:\Users\VICTOR\Documents\Programming (Python)\5-year-mean-1951-1980.csv', delimiter=',', usecols=(0, 1))
#years, mean =pd.read_csv (r'C:\Users\VICTOR\Documents\Programming (Python)\5-year-mean-1882-2014.csv', delimiter=',', usecols=(0, 1))

yearsBase, meanBase = np.loadtxt(r"C:\\Users\\VICTOR\\Documents\\Programming\\Python Programming\\5-year-mean-1951-1980.csv", delimiter=',', usecols=(0, 1),unpack=True)
years, mean = np.loadtxt(r"C:\\Users\\VICTOR\\Documents\\Programming\\Python Programming\5-year-mean-1882-2014.csv", delimiter=',', usecols=(0, 1),unpack=True)

df = pd.read_csv(r"C:\Users\VICTOR\Documents\Programming (Python)\\Python Programming\\FlightData.csv")

#Below is the code that generates random numbers for axes(m and d). You only need to remove quotations.

''' meanBase=np.round(np.random.normal(1.74,0.20,10),10)   
yearBase=np.round(np.random.normal(1.84,0.80,10),10) '''


m,t =(yearsBase, meanBase)

def f(x):
    return m*x + t


plt.xlabel('Mean Temperature', fontsize=10)
plt.ylabel('Years',fontsize=10)
plt.title('Mean Temperature versus Years.',fontsize=15)
plt.grid()

plt.yticks(color='darkblue')
plt.xticks(color='darkred')

plt.scatter(yearsBase,meanBase,edgecolors='darkblue')
plt.plot(yearsBase,meanBase)

plt.show()


print('y={0}*x + {1}'.format(m,t))   #To display slope and intercept on the graph.

sns.regplot(yearsBase, meanBase)         #To show projection of line.
plt.show()
