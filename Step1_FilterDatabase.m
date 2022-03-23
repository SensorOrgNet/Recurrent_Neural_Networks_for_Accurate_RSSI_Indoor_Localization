%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wifi Indoor Localization
% Minhtu 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filter the database with Average Weighted Filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;

FILTER_OPTION = 1;  % 1: Average Filter
                    % 0: Median Filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter - 1 Unit = 1m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Num_Mac = 11; % Number of APs per vector in database

% Test
myFolder = 'C:\Users\minh_\Desktop\CSI_RSSI_Database\RSSI_6AP_Experiments\Nexus4_Data\'; % Database Folder
Input = importdata([myFolder  'UpdatedDatabase_24Jan2018_Full.csv']);
Database = Input.data;
Temp = size(Database);
LengthDatabase = Temp(1); % Number of RPs

%%% Spit locaion from Database
PreX = Database(1,1);
PreY = Database(1,2);
StartingPoint = 1;
CountLocation = 0;
for CountBlock = 2:LengthDatabase
    X = Database(CountBlock,1);
    Y = Database(CountBlock,2);
    if (X ~= PreX) || (Y ~= PreY) || (CountBlock == LengthDatabase)
        PreX = X;
        PreY = Y;
        EndingPoint = CountBlock-1;
        CountLocation = CountLocation + 1 % Count Number of Location
        LengthBlock = EndingPoint - StartingPoint + 1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%% Filter RSSI in a specific location %%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        RSSI_Original = Database(StartingPoint:EndingPoint, 3:end);
        RSSI_Filtered = zeros(size(RSSI_Original));
        F1_Before = zeros(1,3);
        F2_Before = zeros(1,3);
        RSSI_Before = zeros(1,3);

        for CountMac = 1:Num_Mac
            RSSI_Array_Temp = RSSI_Original(1:LengthBlock, CountMac);
            RSSI_Array_After = zeros(LengthBlock, 1);
            MeanValue = mean(RSSI_Array_Temp);
            n = 1;
            for CountPoint = 1:LengthBlock  
                RSSI_Temp = RSSI_Array_Temp(CountPoint);
                if RSSI_Temp == -100 % Avoid -100
                    RSSI_Temp = MeanValue;
                end
                if n == 1
                    RSSI_Before(1) = RSSI_Temp;
                elseif n == 2
                     RSSI_Before(2) = RSSI_Temp;  
                elseif n == 3
                     RSSI_Before(3) = RSSI_Temp;     
                else
                    RSSI_Before(1) = RSSI_Before(2);
                    RSSI_Before(2) = RSSI_Before(3);  
                    RSSI_Before(3) = RSSI_Temp; 
                end
                % Median Filter
                if FILTER_OPTION == 0 % Median Filter
                    RSSI_After_Median = Median_Filter(RSSI_Before,n);
                    RSSI_Array_After(CountPoint) = round(RSSI_After_Median);
                    n = n + 1;
                else          
                    % Average Filter
                    % Filter
                    [RSSI_After, F1, F2, TimeCount] = Average_Filter(RSSI_Before, F1_Before, F2_Before,n);               
                    % Updated Array
                    n = TimeCount;
                    F1_Before = F1;
                    F2_Before = F2;
                    RSSI_Array_After(CountPoint) = round(RSSI_After);
                end
            end
            RSSI_Filtered(1:LengthBlock, CountMac) = RSSI_Array_After;
        end
        % Update to Database
        Database(StartingPoint:EndingPoint, 3:end) = RSSI_Filtered;
        StartingPoint = CountBlock; % Restart Starting Point
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
csvwrite([myFolder 'UpdatedDatabase_8June2018_11MAC_AverageFilter.csv'],Database);    
    