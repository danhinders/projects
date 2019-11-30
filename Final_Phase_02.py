# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:40:11 2019

@author: danhi
"""
import pandas as pd 
import numpy as np

def main():
    

     #column names for the data
    col = ["Scn", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Class"]
    
    #build the dataset 
    df = pd.read_csv('breast-cancer-wisconsin.data',
    names = col)
    
    #y = convert A7 to numeric inseting NaN instead of ?
    y = pd.to_numeric(df['A7'], errors='coerce')
    
    #use y to find the mean = z to be used instead of the ?s 
    z = y.mean()
    z = (round(z,1))

    #replace ? in the dataset (all found in A7) with the mean of A7 rounded to 2 decimal places
    df = df.replace('?', z)
    
    #convert to numeric again
    df = df.apply(pd.to_numeric, errors='coerce')
    
    #grab 2 random rows
    u2_row = np.random.randint(low=1, high=699)
    u4_row = np.random.randint(low=1, high=699)

    #use random rows to build intial centroids
    u2 = df.iloc[u2_row]["A2":"A10"]
    u4 = df.iloc[u4_row]["A2":"A10"]
    
    #getting all df values in the A columns
    data_points = df.iloc[0:700, 1:10]
    
    #find Euclidian distance between each A column and the selected centroid
    d2 = np.sqrt(np.sum((data_points - u2)**2, axis=1))
    d4 = np.sqrt(np.sum((data_points - u4)**2, axis=1))
    
    #where u2's distance is less, put a 2 in column pc, if not put 4 in column
    pc = np.where(d2 < d4, 2, 4)
    
    #insert the pc column into the df
    df.insert(11, "Predicted_Class", pc, True)
    
    #grab all rows where pc is 2 and pc is 4
    class_2_all = df.loc[df['Predicted_Class'] == 2]
    class_4_all = df.loc[df['Predicted_Class'] == 4]
   
    #thin out those rows just to include the A columns
    class_2 = class_2_all.iloc[:, 1:10]
    class_4 = class_4_all.iloc[:, 1:10]
    
    #find the mean of each column in each grouping 
    mu_u2 = np.mean(class_2)
    mu_u4 = np.mean(class_4)
    
    #create and populate  first two columns of dataframe for u2 to collect all centriods we will create in the for loop below
    centroids_u2 = pd.DataFrame()
    centroids_u2.insert(0, "intial u2", u2, True)
    centroids_u2.insert(1, "mu_u2", mu_u2, True)
    
    #create and populate first two columns of dataframe for u2 to collect all centriods we will create in the for loop below
    centroids_u4 = pd.DataFrame()
    centroids_u4.insert(0, "intial u4", u4, True)
    centroids_u4.insert(1, "mu_u4", mu_u4, True)
    
    #for loop to find Euclidian distance and new centroids 50 times then store the resuts for u2 and u4 centroids in dataframes
    for i in range (1,51):
        #find Euclidian distance between each A column and the selected centroid
        d2 = np.sqrt(np.sum((data_points - centroids_u2.iloc[:,i])**2, axis=1))
        d4 = np.sqrt(np.sum((data_points - centroids_u4.iloc[:,i])**2, axis=1))
        
        #where u2's distance is less, put a 2 in column pc, if not put 4 in column
        pc = np.where(d2 < d4, 2, 4)
        
        
        #remove previous predicted class to make way for the new one
        df = df.drop(columns=["Predicted_Class"])
        
        
        #insert the pc column into the df
        df.insert(11, "Predicted_Class", pc, True)
        
        #grab all rows where pc is 2 and pc is 4
        class_2_all = df.loc[df['Predicted_Class'] == 2]
        class_4_all = df.loc[df['Predicted_Class'] == 4]

        #thin out those rows just to include the A's columns
        class_2 = class_2_all.iloc[:, 1:10]
        class_4 = class_4_all.iloc[:, 1:10]

        #find the mean of each column in each grouping 
        mu_u2_new = np.mean(class_2)
        mu_u4_new = np.mean(class_4)
        
        #insert new centroids into dateframe and provide column heading with iteration number the centroid was created
        label = "Iteration " + str(i)
        centroids_u2.insert((i+1), label, mu_u2_new, True)
        centroids_u4.insert((i+1), label, mu_u4_new, True)

    #print out the original ramdomly selected centroid 
    print("\nRandomly selected row",u2_row,"for centroid mu_2.\n")
    
    print("Initial centroid mu_2:")
    print(u2)
    
    print("\nRandomly selected row",u4_row,"for centroid mu_4.\n")
    print("Initial centroid mu_4:")
    print(u4)
    
    #create df to store counter related to for loop below  
    equal_centroids_u2 = pd.DataFrame()
    
    #for loop to check if each centriod in the centriod is equal to the previous centriod
    for i in range (0,51):
        u2_equal = centroids_u2.iloc[:,i] == centroids_u2.iloc[:,(i+1)]

        int_u2_equal = list(map(int,u2_equal))
        
        label = "Iteration " + str(i)
        equal_centroids_u2.insert((i), label, int_u2_equal, True)
    
    #if all centroids were equal on the first run then there would be 450 total identical counts
    # we subtract the sum the number of identical counts from 450 and divide by the number of columns
    #this gives us the number of iterations it took to get an equal set of centroids
    sum_u2_equal =  np.sum(equal_centroids_u2[0:])
    sum_u2_total = np.sum(sum_u2_equal)
    iterations  = (450 - sum_u2_total)/9 
    
    #print the rest of the data we want to see about the program outputs
    print("\nProgram ended after", int(iterations),"iterations.")
    
    print("\nFinal centroid mu_2:")
    print(centroids_u2.iloc[:,int(iterations+1)],"\n")
    
    print("Final centroid mu_4:")
    print(centroids_u4.iloc[:,int(iterations+1)])

    print("\nFinal cluster assignment:\n")
    print(df.iloc[:21, [0,10,11]])

main()
