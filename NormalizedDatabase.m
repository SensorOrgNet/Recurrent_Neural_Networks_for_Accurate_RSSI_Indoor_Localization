%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X Y Z MAC1 Mean_RSSI1 MAC2 Mean_RSSI2 ...   
% Normalize Database - Standardization
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter - 1 Unit = 40 inches
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Num_Mac = 11; % Number of APs per vector in database

% Normalize ------------------------------------
% r = (r- mean)/sigma 
% -----------------------------------------------

% Test
myFolder = 'C:\Users\minh_\Desktop\CSI_RSSI_Database\RSSI_6AP_Experiments\Nexus4_Data\'; % Database Folder
InputTest = importdata([myFolder 'UpdatedDatabase_8June2018_11MAC_AverageFilter.csv']);
Database = InputTest;
Temp = size(Database);
LengthDatabase = Temp(1); % Number of Test points

% % % % InputMean = importdata('NormalizeddDatabase_AverageFilter.csv');

RSSI_Array = Database(:,3:end);
Location_Array = Database(:,1:2);
NumLocation = length(Location_Array);

%%%% Locations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MeanArray_Location = zeros(1,2);
StdArray_Location = zeros(1,2);
for ii = 1:2
    MeanArray_Location(ii) = mean(Location_Array(:,ii));
end
for Count = 1:NumLocation
    for CountMac = 1:2
        StdArray_Location(CountMac) =  StdArray_Location(CountMac) + (Location_Array(Count,CountMac)-MeanArray_Location(CountMac))^2;
    end
end
StdArray_Location = StdArray_Location/LengthDatabase;
StdArray_Location = sqrt(StdArray_Location);

%%%% RSSI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate Mean & Standard deviation for every AP
MeanArray_RSSI = zeros(1,Num_Mac);
%%%% Mean Calculation %%%%%%%%%%%%%%%%
for Count = 1:LengthDatabase
    for CountMac = 1:Num_Mac
        MeanArray_RSSI(CountMac) =  MeanArray_RSSI(CountMac) + RSSI_Array(Count,CountMac);
    end
end
MeanArray_RSSI = round(MeanArray_RSSI / LengthDatabase);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Standard Deviation %%%%%%%%%%%%%%%%%%%%%
StdArray_RSSI = zeros(1,Num_Mac);
for Count = 1:LengthDatabase
    for CountMac = 1:Num_Mac
        StdArray_RSSI(CountMac) =  StdArray_RSSI(CountMac) + (RSSI_Array(Count,CountMac)-MeanArray_RSSI(CountMac))^2;
    end
end
StdArray_RSSI = StdArray_RSSI/LengthDatabase;
StdArray_RSSI = round(sqrt(StdArray_RSSI));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Normalized Step
Normalized_Database = zeros(size(Database));
for Count = 1:LengthDatabase
    % Location 
    for ii = 1:2
        Normalized_Database(Count,ii) = (Database(Count,ii) - MeanArray_Location(ii))/StdArray_Location(ii);
    end
    % RSSI
    for ii = 3:13
        Normalized_Database(Count,ii) = (Database(Count,ii) - MeanArray_RSSI(ii-2))/StdArray_RSSI(ii-2);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
csvwrite([myFolder 'Normalize_Database_AverageFilter.csv'],Normalized_Database); 
