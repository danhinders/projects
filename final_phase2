# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:40:11 2019

@author: danhi
"""

#final project phase 2
#
# Date: 11/17/2019
# Author: Daniel Hinders


import pandas as pd 
import numpy as np
import math
from scipy.spatial import distance

#this function takes the data frame and the row number and provides back the row data values in a list
def getrows(row,df):
    
    all_rows = []
    
    #adding all values from the df to the list of all values
    for i in range((df.shape[0])): 
        all_rows.append(list(df.iloc[i, :])) 
    
    
    #to slice the row we want correctly we iterate 1 numbers below the index to grab the correct row below
    row_index = row - 1
    
    #create other number for index slicing
    row_plus1 = (row_index) + 1
    #we slice to get the row index number that was passed into the function 
    selected_full_row = all_rows[row_index:(row_plus1)]
    
    selected_row = []
    for i in selected_full_row:
        selected_row.extend(i)

    selected_row = selected_row[1:10]
#    print(all_rows)
#    print(selected_full_row)    
#    print(selected_row)
#    print(row)


    return(selected_row)
    

def main():
    #column names for the data
    col = ["Scn", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Class"]
    #build the dataset 
    df = pd.read_csv('breast-cancer-wisconsin.data',
    names = col)
    
    #y = convert A7 to numeric inseting NaN instead of ?
    y = pd.to_numeric(df['A7'], errors='coerce')
    
    #use y to find the mean = z
    z = y.mean()
    z = (round(z,1))

    #replace ? in the dataset (all found in A7) with the mean of A7 rounded to 2 decimal places
    df = df.replace('?', z)
    
    #convert to numeric again
    df = df.apply(pd.to_numeric, errors='coerce')
    
    #select our first random centroid rows 
    row = np.random.randint(low=1, high=699)
    u2 = getrows(row,df)

    row = np.random.randint(low=1, high=699)
    u4 = getrows(row,df)
 
    
    Predicted_Class = []
    
    for row in range(1,700):

            df_row = getrows(row,df)

            distance_u2 = distance.euclidean(df_row, u2)
            distance_u4 = distance.euclidean(df_row, u4)
            
            if distance_u2 > distance_u4:
                Predicted_Class.append(2)

            else:
                Predicted_Class.append(4)

    
    df.insert(11, "Predicted_Class", Predicted_Class, True)
    print(df)
   

main()
