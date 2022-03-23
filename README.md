# Recurrent Neural Networks for Accurate RSSI Indoor Localization

Source code for M.T. Hoang, B. Yuen, X. Dong, T. Lu, R. Westendorp and K. Reddy, “Recurrent Neural Networks for Accurate RSSI Indoor Localization,” IEEE Internet of Things Journal, 2019


# Folder Structure
*  Step1_FilterDatabase.m: Filter the database with Average Weighted Filter or Mean Filter
*  Step2_Create_RandomTraj.m: Generate random training trajectories under the constraints that the distance between consecutive locations is bounded by 
the maximum distance a user can travel within the sample interval in practical scenarios.
*  Step2_CreateInputTraining_Model5: Create the input training data for P-MIMO LSTM
*  RNN models training code (Using Keras and Tensorflow) 
   *  LSTM_Model_1.py 
   *  LSTM_Model_2.py
   *  LSTM_Model_3.py
   *  LSTM_Model_4.py
   *  LSTM_Model_5.py


