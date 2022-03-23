#############################################################
# 10 locations/ trajectory
# First location is known
# Date: 8 June 2018
#############################################################

import time
import tensorflow
# import lstm
import os
#import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
# import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
import math
from pandas import DataFrame
from keras import backend as K

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
def err_absolute(y_true, y_pred):
        err = K.sqrt(K.square(y_pred - y_true))
        return err 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

# Open File to write results
File1 = open('Error_Model2.csv','w')
File2 = open('Traj_Model2.csv','w') 
# File3 = open('Loss_PredictedL_50k_NoFilter.csv','w') 
# File4 = open('Output_PredictedL_50k_NoFilter.csv','w') 

NumTotalTraj = 120000

# Configure the sampling set -----------------
epochs_num  = 300
NumTrajTraining = 40000
NumSam_PerTraj = 1
StartingTraj = 30000
StartingSample = StartingTraj*NumSam_PerTraj # the factor of NumSam_PerTraj 
NumSample = NumTrajTraining*NumSam_PerTraj # NumTraj = NumSample/NumSam_PerTraj

# Configure the validation set -----------------
NumTrajVal = 10000
StartingTrajVal = 100000

StartingValidation = StartingTrajVal*NumSam_PerTraj
NumValidation = NumTrajVal*NumSam_PerTraj

# Main Run Thread
global_start_time = time.time()
num_rssi_reading = 11
num_out_location = 2
output_layer = 2
timestep = 9
input_layer = (num_rssi_reading + num_out_location)*timestep # X, Y & 11 MAC Addresses
NumBatch = 256
hidden_layer1 = 100
hidden_layer2 = 100

##########################################################
###### Test Trajectory Splitting #########################
########################################################## 
# Read Training data -------5-------------------------------------------
print('> Loading data... ')
filename = os.path.join(fileDir, '../RSSI_6AP_Experiments/Nexus4_Data/Input_Location_RSSI_10points_365k.csv')
filename = os.path.abspath(os.path.realpath(filename))
df1=pd.read_csv(filename)
df1= np.asarray(df1)
df1_split = df1[StartingSample:StartingSample+NumSample,:]
input_training = df1_split.reshape(len(df1_split),1,input_layer)
print(input_training.shape)

filename = os.path.join(fileDir, '../RSSI_6AP_Experiments/Nexus4_Data/Output_Location_RSSI_10points_365k_Model2.csv')
filename = os.path.abspath(os.path.realpath(filename))
df2=pd.read_csv(filename)
df2= np.asarray(df2)
df2_split = df2[StartingSample:StartingSample+NumSample,:]
output_training = df2_split.reshape(len(df2_split),1,output_layer)
print(output_training.shape)

########################################################################
# Read Validation data --------------------------------------------------
#######################################################################
df3= df1[StartingValidation:StartingValidation+NumValidation,:]
input_validation = df3.reshape(len(df3),1,input_layer)
print(input_validation.shape)

df4=df2[StartingValidation:StartingValidation+NumValidation,:]
output_validation = df4.reshape(len(df4),1,output_layer)
print(output_validation.shape)

####################################################################
# Testing data ------------------------
####################################################################
# RSSI array
filename = os.path.join(fileDir, '../RSSI_6AP_Experiments/Nexus4_Data/Long_Traj_6July_v1_AverageFilter.csv')
filename = os.path.abspath(os.path.realpath(filename))
TestData_origin =pd.read_csv(filename)
print(TestData_origin.shape)
TestData_origin = np.asarray(TestData_origin)

LengthTest = len(TestData_origin)
StartTestIdx = 0
StopTestIdx = StartTestIdx + LengthTest

# print(TestData.shape)
TestData = TestData_origin[StartTestIdx:StopTestIdx, :]
# LengthTest = len(TestData)

# List of Locations 
filename = os.path.join(fileDir, '../RSSI_6AP_Experiments/Nexus4_Data/Long_Traj_Location.csv')
filename = os.path.abspath(os.path.realpath(filename))
Location_origin = pd.read_csv(filename)
print(Location_origin.shape)
Location_origin = np.asarray(Location_origin)
Location = Location_origin[StartTestIdx:StopTestIdx, :]

print('> Data Loaded. Compiling...')

####################################################################
# Init 9 first Location ###########################################
####################################################################
L1 = Location[0,:]
L2 = Location[0,:]
L3 = Location[0,:]
L4 = Location[0,:]
L5 = Location[0,:]
L6 = Location[0,:]
L7 = Location[0,:]
L8 = Location[0,:]
L9 = Location[0,:]
L10 = Location[0,:]
L_combine = np.concatenate((L1,L2,L3,L4,L5,L6,L7,L8,L9), axis=0)

# Take RSSI
RSSI_L1 = TestData[0,:]
RSSI_L2 = TestData[1,:] 
RSSI_L3 = TestData[1,:]
RSSI_L4 = TestData[1,:] 
RSSI_L5 = TestData[1,:]
RSSI_L6 = TestData[1,:] 
RSSI_L7 = TestData[1,:]
RSSI_L8 = TestData[1,:] 
RSSI_L9 = TestData[1,:]
RSSI_L10 = TestData[1,:]
RSSI_combine = np.concatenate((RSSI_L2, RSSI_L3, RSSI_L4, RSSI_L5,RSSI_L6, RSSI_L7, RSSI_L8, RSSI_L9, RSSI_L10), axis=0)

# Build the network  --------------------------------------------
model1 = Sequential()
model1.add(LSTM(hidden_layer1, input_shape=(1, input_layer), return_sequences=True))
model1.add(Dropout(0.2))
model1.add(LSTM(hidden_layer2,return_sequences=True))
model1.add(Dropout(0.2))
model1.add(TimeDistributed(Dense(output_layer)))
model1.summary()
start = time.time()
model1.compile(loss="mse", optimizer="adam",metrics=[rmse])
# model1.compile(loss=rmse, optimizer="adam",metrics=[err_absolute]) #rmse
# root_mean_squared_error
print("> Compilation Time : ", time.time() - start)

# Training --------------------------------------------------------
loss = list()
ResultPlot = DataFrame()

#### Create a copy of input training ##########################
input_training_org = input_training
input_validation_org = input_validation
###############################################################
model1.load_weights("lstm_20k_model2.h5")
StartingValidation = 0
if epochs_num > 0:
 #   iTimeStep = 1
  #  iTimeStep_val = 1
    for ep in xrange(epochs_num):
        print("Iteration {} ----- ".format(ep))

        # Fit Model
      #  hist = model1.fit(input_training,output_training, validation_data=(input_validation, output_validation), epochs=1, batch_size=NumBatch, verbose=1)
        hist = model1.fit(input_training,output_training, epochs=1, batch_size=NumBatch, verbose=1)
        loss.append(hist.history['loss'][0])

        # model1.fit(input_training,output_training, validation_split=0.2, epochs=epochs_num, batch_size=512)
        model1.save_weights("lstm_20k_model2.h5")
    print('Training duration (s) : ', time.time() - global_start_time)
else:
    print("TESTING...........")
    pass

# Testing --------------------------------------------------------- 
#########################################################
# The first steps: Fill up the buffer
# #######################################################
CountTest = 0
Average_Err = 0 
error = np.zeros(LengthTest)

for Step in xrange(timestep):
    print("Location {}------------".format(CountTest))
    # Update the Locations & RSSIs buffer
    # Update the Locations & RSSIs buffer
    LocationIdx = 0
    RSSIdx = 0
    TestingData = np.zeros(len(L_combine)+len(RSSI_combine))
    for i in xrange(timestep):
        TestingData[LocationIdx] = L_combine[i*2]
        TestingData[LocationIdx+1] = L_combine[i*2+1]
        for j in xrange(len(RSSI_L2)):
            TestingData[j+LocationIdx+2] = (float(RSSI_combine[j+RSSIdx])+100)/100
        LocationIdx = LocationIdx + num_rssi_reading + 2
        RSSIdx = RSSIdx + num_rssi_reading
        # print(TestingData)

    TestingData = TestingData.reshape(1,1,input_layer)
    # Prediction 
    Predicted_L = model1.predict(TestingData)   
    Final_L = Predicted_L[0,0,:] 
    
    Correct_L = Location[CountTest,:]
    error[CountTest] = np.sqrt(np.power((Final_L[0] - Correct_L[0]),2)+np.power((Final_L[1] - Correct_L[1]),2))
    print "Predict: {}--- Exact: {} , Error: {}" .format((Final_L[0], Final_L[1]), (Correct_L[0], Correct_L[1]), (error[CountTest]))
    Average_Err = Average_Err + error[CountTest]
    CountTest = CountTest + 1

    IdxTemp = 0
    # Update Location
    for j in xrange(Step*2,len(L_combine)):
        L_combine[j] = Final_L[IdxTemp]
        if IdxTemp == 0:
            IdxTemp = 1
        else:
            IdxTemp = 0
            pass

    # Take RSSI
    IdxTemp = 0
    for j in xrange(Step*num_rssi_reading,len(RSSI_combine)):
        RSSI_combine[j] = TestData[Step+1,IdxTemp]
        IdxTemp = IdxTemp + 1
        if IdxTemp == num_rssi_reading: # Reach the end
            IdxTemp = 0

#########################################################
# AFter the buffer is full -
# #######################################################  

while CountTest < LengthTest-1:
    print("Location {}------------".format(CountTest))
# Update the Locations & RSSIs buffer
    LocationIdx = 0
    RSSIdx = 0
    TestingData = np.zeros(len(L_combine)+len(RSSI_combine))
    for i in xrange(timestep):
        TestingData[LocationIdx] = L_combine[i*2]
        TestingData[LocationIdx+1] = L_combine[i*2+1]
        for j in xrange(len(RSSI_L2)):
            TestingData[j+LocationIdx+2] = (float(RSSI_combine[j+RSSIdx])+100)/100
        LocationIdx = LocationIdx + num_rssi_reading + 2
        RSSIdx = RSSIdx + num_rssi_reading

    TestingData = TestingData.reshape(1,1,input_layer)

    # Prediction 
    Predicted_L = model1.predict(TestingData)   
    Final_L = Predicted_L[0,0,:]   
    
    Correct_L = Location[CountTest,:]
    error[CountTest] = np.sqrt(np.power((Final_L[0] - Correct_L[0]),2)+np.power((Final_L[1] - Correct_L[1]),2))
    print "Predict: {}--- Exact: {} , Error: {}" .format((Final_L[0], Final_L[1]), (Correct_L[0], Correct_L[1]), (error[CountTest]))
    Average_Err = Average_Err + error[CountTest]

    File1.write(str(error[CountTest]) + ' , ') # write to file
    File2.write(str(Final_L[0]) + ',') 
    File2.write(str(Final_L[1]) + '\n') # write to file

    # Re-arrange L_combine
    for t in xrange(timestep-1): 
        L_combine[t*2] = L_combine[(t+1)*2]
        L_combine[t*2+1] = L_combine[(t+1)*2+1] 
    # Update 
    L_combine[(timestep-1)*2] = Final_L[0] 
    L_combine[(timestep-1)*2+1] = Final_L[1] 

    # Re-arrange RSSI_combine
    for t in xrange(timestep-1): 
        for k in xrange(num_rssi_reading):
            RSSI_combine[t*num_rssi_reading+k] =  RSSI_combine[(t+1)*num_rssi_reading+k]

    for k in xrange(num_rssi_reading):
        RSSI_combine[(timestep-1)*num_rssi_reading+k] = TestData[CountTest+1,k]      

    CountTest = CountTest + 1 
###################################################################
############### The Last Locations ###############################
###################################################################

Average_Err = Average_Err/LengthTest
print "Average Error: ", Average_Err

Std_Err = 0
for k in xrange(LengthTest):
    Std_Err = Std_Err +  np.power((error[k] - Average_Err),2)
Std_Err = Std_Err/(LengthTest-1)
Std_Err = np.sqrt(Std_Err)
print "Std: ", Std_Err 

#### Show Figure ######################
if epochs_num > 0:
    ResultPlot['neurons_500'] =  loss
    ResultPlot.plot()
    pyplot.show()

File1.close()
File2.close()
# File3.close()
# File4.close()