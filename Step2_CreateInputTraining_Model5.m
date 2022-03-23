%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wifi Indoor Localization
% Minhtu 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;

% Database
myFolder = 'C:\Users\minh_\Desktop\CSI_RSSI_Database\RSSI_6AP_Experiments\Nexus4_Data\'; % Database Folder

Input = importdata([myFolder 'UpdatedDatabase_8June2018_11MAC_AverageFilter.csv']);
Database = Input;
L_Data = length(Database);
NumReading = 11;

TrajInput = importdata([myFolder 'Traj_10points_5k_5m_s.csv']);
TrajData = TrajInput;

SizeTraj = size(TrajData);
L = SizeTraj(1);
NumPointPerTraj = SizeTraj(2)/2; % How many Time Steps in LSTM trajectory

%-----------------------------------------------------------
%%%% Add RSSI to the trajectory 
%-----------------------------------------------------------
RNN_Database = zeros(L, NumPointPerTraj*2+(NumPointPerTraj-1)*NumReading);
for ii = 1:L % Scan all trajectory
    CntColRNN = 1;
    for jj = 1: NumPointPerTraj % Scan all points in the trajectory
        
        if jj == NumPointPerTraj % Don't take the last RSSI
            break;
        end
        X = TrajData(ii,(jj-1)*2+1); 
        Y = TrajData(ii,(jj-1)*2+2); 
        RNN_Database(ii, CntColRNN) = X; 
        RNN_Database(ii, CntColRNN+1) = Y; 
        CntColRNN = CntColRNN + 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        X = TrajData(ii,jj*2+1); % Take the next RSSI for current point
        Y = TrajData(ii,jj*2+2); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for kk = 1:5:L_Data % Scan all   
            if (abs(Database(kk,1) - X) < 10^-3)&&(abs(Database(kk,2) - Y) < 10^-3)
                RandNum = round(100*rand(1)); % Randomly pick up 1 RSSI Reading Row
                if kk+RandNum > length(Database)
                    RandNum = 0;
                end
                if (abs(Database(kk+RandNum,1) - X) < 10^-3)&&(abs(Database(kk+RandNum,2) - Y) < 10^-3)
                    for mm = 1:NumReading
%                        RNN_Database(ii, CntColRNN) =  Database(kk+RandNum,mm+2);
                       RNN_Database(ii, CntColRNN) =  (Database(kk+RandNum,mm+2)+100)/100;
                       CntColRNN = CntColRNN+1;   
                    end
                else % if out of range, pick up the 1st one
                    for mm = 1:NumReading
%                        RNN_Database(ii, CntColRNN) =  Database(kk,mm+2);
                       RNN_Database(ii, CntColRNN) =  (Database(kk,mm+2)+100)/100;
                       CntColRNN = CntColRNN+1;   
                    end
                end
                break;
            end    
        end
    end
end
csvwrite([myFolder 'Input_Location_RSSI_10points.csv'],RNN_Database);
