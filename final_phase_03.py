# -*- coding: utf-8 -*-
"""
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
    
    #counter for the loop below to check how many times the loop runs
    iterations = 0
    
    #dataframes to grab the new centroids we create in the loop below
    mu_u2_new = pd.DataFrame()
    mu_u4_new = pd.DataFrame()
    
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
            iterations = iterations + 1
            
            #check if the  new centroid matches the previous one, if it does break out of the loop
            if mu_u2_new.equals(centroids_u2.iloc[:,i]) and \
               mu_u4_new.equals(centroids_u4.iloc[:,i]):
               break




    ##
    ##Section below finds and stores all the various errors between predicted class and class.
    ##also finds the specific error rows and the values each error groups and totals.
    ##

    #number of data points predicted as class 2, while the correct class is 4
    counting_nums1 = np.array(np.where(df['Class'] > df['Predicted_Class']))
    counting_nums1 = counting_nums1.flatten().tolist()
    error_24 = len(counting_nums1)


    #number of data points predicted as class 4, while the correct class is 2
    counting_nums2 = np.array(np.where(df['Class'] < df['Predicted_Class']))
    counting_nums2 = counting_nums2.flatten().tolist()
    error_42 = len(counting_nums2)

    
    #number of data points with predicted class not equal to correct class
    error_all = error_42 + error_24

    
    #number of data points with predicted class equal to 2
    counting_nums3 = np.array(np.where(df['Predicted_Class'] == 2 ))
    counting_nums3 = counting_nums3.flatten().tolist()
    pclass_2 = len(counting_nums3)
     
    #number of data points with predicted class equal to 2
    counting_nums4 = np.array(np.where(df['Predicted_Class'] == 4 ))
    counting_nums4 = counting_nums4.flatten().tolist()
    pclass_4 = len(counting_nums4)

    
    #number of data points
    class_all = pclass_4 + pclass_2

    
    #error rate for the benign cells
    error_B = (error_24 / pclass_2) * 100

    
    #error rate for the malign cells
    error_M = (error_42 / pclass_4) * 100

    #total error rate
    error_T = (error_all) / (class_all) * 100


    #grab all rows that have the 4-2 and 2-4 errors
    error_24_all = df.iloc[counting_nums1]
    error_42_all = df.iloc[counting_nums2]
          
    #thin out those rows just to include the 1st and last two columns
    error_24_some_columns = error_24_all.iloc[:, [0,10,11]]
    error_42_some_columns = error_42_all.iloc[:, [0,10,11]]
    
    ##
    ##Section below uses if/else to determine if total error rate more than 50%
    ##if not then we can utilize the prints statements in the if statement below, if above 50% drop to the else statement
    ##    

    if error_T < 50:
       
       print("Data points in Predicted Class 2: ", pclass_2)
       print("Data points in Predicted Class 4: ", pclass_4)
       print("\nError data points, Predicted Class 2:\n\n",error_24_some_columns)
       print("\nError data points, Predicted Class 4:\n\n",error_42_some_columns)
       
       print("\nNumber of all data points:        ",class_all)
       
       print("\nNumber of all data points:        ",error_all)

       print("\nError rate for class 2:           ",round(error_T,1),"%")
       print("Error rate for class 4:           ",round(error_B,1),"%")
       print("Total errors:                     ",round(error_M,1),"%")
    else:
        
        ##
        ##if we drop to the else statement the centroids are swapped. We have the logic to
        ##swap the value of predicited class and re-insert it into new df = df1
        ##
        
        print("Total errors:            ", round(error_T,1),"%")
        print("Centroids are swapped!")
        print("Swapping Predicted_Class\n")

        
        #swap value of 2 and 4 in the predicted class column

        pc = np.where(df['Predicted_Class']> 2, 2, 4)
        
        #drop current predicted class column
        df1 = df.drop(columns=["Predicted_Class"])
        
        #insert the new pc column into new df1

        df1.insert(11, "Predicted_Class", pc, True)
        
        
        ##
        ##when the centroids are swapped we use sections below to recalualte the 
        ##various errors between the updated predicted class and class.
        ##also finds the specific error rows and the values each error groups and totals and prints the results.
        ##
        
        
        #number of data points predicted as class 2, while the correct class is 4
        counting_nums1 = np.array(np.where(df1['Class'] > df1['Predicted_Class']))
        counting_nums1 = counting_nums1.flatten().tolist()
        error_24 = len(counting_nums1)
 
    
        #number of data points predicted as class 4, while the correct class is 2
        counting_nums2 = np.array(np.where(df1['Class'] < df1['Predicted_Class']))
        counting_nums2 = counting_nums2.flatten().tolist()
        error_42 = len(counting_nums2)
       
        
        #number of data points with predicted class not equal to correct class
        error_all = error_42 + error_24
       
        
        #number of data points with predicted class equal to 2
        counting_nums3 = np.array(np.where(df1['Predicted_Class'] == 2 ))
        counting_nums3 = counting_nums3.flatten().tolist()
        pclass_2 = len(counting_nums3)
        
        
        #number of data points with predicted class equal to 2
        counting_nums4 = np.array(np.where(df1['Predicted_Class'] == 4 ))
        counting_nums4 = counting_nums4.flatten().tolist()
        pclass_4 = len(counting_nums4)

        
        #number of data points
        class_all = pclass_4 + pclass_2

        
        #error rate for the benign cells
        error_B = (error_24 / pclass_2) * 100

        
        #error rate for the malign cells
        error_M = (error_42 / pclass_4) * 100

        #total error rate
        error_T = (error_all) / (class_all) * 100

        
        #grab all rows that have the 4-2 and 2-4 errors
        error_24_all = df1.iloc[counting_nums1]
        error_42_all = df1.iloc[counting_nums2]

 
        #thin out those rows just to include the 1st and last two columns
        error_24_some_columns = error_24_all.iloc[:, [0,10,11]]
        error_42_some_columns = error_42_all.iloc[:, [0,10,11]]

        
    
        print("Data points in Predicted Class 2: ", pclass_2)
        print("Data points in Predicted Class 4: ", pclass_4)
        print("\nError data points, Predicted Class 2:\n\n",error_24_some_columns)
        print("\nError data points, Predicted Class 4:\n\n",error_42_some_columns)
       
        print("\nNumber of all data points:        ",class_all)
       
        print("\nNumber of all data points:        ",error_all)

        print("\nError rate for class 2:           ",round(error_T,1),"%")
        print("Error rate for class 4:           ",round(error_B,1),"%")
        print("Total errors:                     ",round(error_M,1),"%")

  
main()
