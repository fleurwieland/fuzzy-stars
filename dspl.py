'''
Assignment 
Filename: "dspl.py"
Date: 2020
Authors: Fleur Wieland, Nancy Subke, Justus Holman
Group: 1
'''

###########################################################
### Imports
import numpy as np                       # for numerical methods
import matplotlib.pyplot as plt          # for plotting 
import pandas as pd                      # for data input and output
import scipy.stats as st                 # for statistical functions
from numpy.polynomial import Polynomial as P
from scipy.interpolate import interp1d
import matplotlib.dates as mdates
import math 
import copy

###########################################################

def parta():
    #global df
    #global data
    #global df2
    
    names= ["salary","pcsalary", "sales", "roe", "pcroe", "ros", "indus",
            "finance", "consprod", "utility", "lsalary", "lsales"]
    
    data = pd.read_csv('data1.csv', delimiter=';', names = np.arange(0,len(names)))
    #print(data)
    #data = pd.DataFrame(columns=np.arange(0,data.shape[1]))
    
    
    df = pd.DataFrame(columns=np.arange(0,len(names)))
    
    
    df.loc[0] = names
    df = df.append([data], ignore_index=True)
    
    df_copy = df.copy()
    
    print(df_copy)
    
    df2 = df_copy.drop([6, 7, 8, 9, 10, 11], axis=1)
    
    
    
    #df2.columns = [0,1,2,3,4,5,6,7]
    print(df2)


    #print(df.iloc[0, 0])
    
    salary = df[0]   #in thousands
    print(salary)
    
    for i in range(1, df2.shape[1]):
        
        title = df2.iloc[0,i]
        print(title)
        vCol = df2[i]
        #print(vCol[1:])
        
        createhist(vCol[1:], title)
        #summaryStatistics(vCol[1:], len(df), title)
        createscatter(salary[1:], vCol[1:], title, salary.iloc[0])
        
 
# =============================================================================
#     plt.subplot(1,3,2)
#     createhist(data2018_T, data1962_T, "temperature", "celcius", "2018", "1962")
#     
#     plt.subplot(1,3,3)
#     createhist(dataRH_sub2018, dataRH_sub1962, "rain", "mm", "2018", "1962")
# =============================================================================
 
    
    
    #You hypothesize that return on equity is a proper measure.
 
    roe = df2[3] #return on equity
    
    simple_lin_reg(roe)


def simple_lin_reg(X1):
    
    

# =============================================================================
# Compute con
dence intervals and hypothesis tests (signi
cance and
# magnitude of coecients);
# Interpret coecients (depends on how X and Y are measured!);
# Assess the accuracy of the model (e.g. residual standard error, R2,
#  R2);
# Make predictions for Y .
# Perform (misspeci
cation) tests on residuals ^" (why?).
# =============================================================================
  
#Estimating a simple linear regression could be a nice starting point, but is probably not
#very realistic as the CEO's compensation is modeled in such a way that it only depends
#on a single variable.

def createhist(data, title):
    #Creates a histogram of the given data with its accessory information.    
    
    #min_data = (min(data))
    #max_data = (max(data))
    
    plt.hist(data, color=('red'))
    plt.title(title,fontweight='bold')
    #plt.xticks(np.arange(min_data, max_data, max_data/10))
    #plt.xlabel(xlabel) 
    #plt.legend([a_label,b_label])  
    plt.show()
    
def summaryStatistics(vX, iN, text):
    print (text)
    dMean = np.mean( vX ) 
    print('mean:', "%.2f" %  dMean )
    dVar = iN/(iN-1) * np.var( vX ) 
    dStdev = np.sqrt( dVar ) 
    print('stdev:', "%.2f" %  dStdev )
    dMax = np.max(vX)
    print('max:',"%.2f" % dMax )
    dMin = np.min(vX)
    print('min:', "%.2f" % dMin)
    
def createscatter(vX, vY, xaxis, yaxis):
    plt.plot( vX , vY , "." )
    plt.ylabel(yaxis)
    plt.xlabel(xaxis)
    #plt.title(title)
    plt.show()
    
    
   


###########################################################
### main
def main():
    parta()
    

###########################################################
### start main
if __name__ == "__main__":
    main()