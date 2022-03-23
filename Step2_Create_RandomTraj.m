%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% WiFi Indoor Localization
% Minhtu 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % Generate random training trajectories under the constraints 
% % % that the distance between consecutive locations is bounded by 
% % % the maximum distance a user can travel within the sample 
% % % interval in practical scenarios.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;

% Database
Norm_On = 0; % 1: Stardadization Normalization
             % 0: Mean Normalization

myFolder = 'C:\Users\minh_\Desktop\CSI_RSSI_Database\RSSI_6AP_Experiments\Nexus4_Data\'; % Database Folder
if Norm_On == 1
    Input = importdata('MeanDatabase_Normalize_AverageFilter.csv');
    Mean_Location = [11.1397910665714,-3.78622840030012];
    Std_Location = [6.31809099260756,6.42186397145320];
else
    Input = importdata([myFolder 'MeanDatabase_8June2018_11MAC.csv']);
end

Database = Input;
L = length(Database);
% Number of times which the database is repeated
NumRepeatedDatabase = 15; % Num of Trajectories = NumRepeatedDatabase*L 
NumPointPerTraj = 10;     % Number of Time Steps in LSTM networks

MapDistance = zeros(L+2, L+2);
NumNeighboursArray = zeros(L,1);
SumProbArray = zeros(L,1);

% Assumption
v_user_max = 5; % m/s % Bounded Speed
t_request = 1; % Period of request time is 5 seconds
distance_max = v_user_max*t_request;
sigma = distance_max;

if Norm_On == 1 
    distance_max = v_user_max*t_request/Std_Location(1);
    sigma = distance_max;
end
%-------------------------------------------------------------
%%%% Build Map Distance 
%-------------------------------------------------------------

for ii = 1:L
    X = Database(ii,1);
    Y = Database(ii,2);
    MapDistance(ii+2,1) = X;
    MapDistance(ii+2,2) = Y;
    MapDistance(1,ii+2) = X;
    MapDistance(2,ii+2) = Y;
    
    CountNeighbour = 0;
    Sum_Prob = 0;
    for jj = 1:L
        X1 = Database(jj,1);
        Y1 = Database(jj,2);
        
        % Calculate Distance
        MapDistance(ii+2,jj+2) = sqrt((X-X1)^2 + (Y-Y1)^2);
      
        if MapDistance(ii+2,jj+2)>distance_max
            MapDistance(ii+2,jj+2) = 0;
        else
          %  Weight = exp(MapDistance(ii+2,jj+3)/(2*sigma^2)); % Based on Gaussian
            ProbFactor = -1/(2*(sigma^2)*(exp(-(distance_max^2)/(2*sigma^2)) - 1));
            % ProbFactor = 1/(sqrt(2*pi)*sigma);
            P_l = ProbFactor*exp(- MapDistance(ii+2,jj+2)/(2*sigma^2)); % Based on Gaussian
            Sum_Prob = Sum_Prob+ P_l;
            MapDistance(ii+2,jj+2) = P_l;
            CountNeighbour = CountNeighbour + 1;
        end
    end
    NumNeighboursArray(ii) = CountNeighbour;
    SumProbArray(ii) = Sum_Prob;
end

%%% Normalized Map
for ii = 1:L
  Sum_Prob = SumProbArray(ii);
  for jj = 1:L
     if MapDistance(ii+2,jj+2) ~= 0
        MapDistance(ii+2,jj+2) = MapDistance(ii+2,jj+2)/Sum_Prob;
     end
  end  
end

%%% Create CDF Map
for ii = 1:L
  SumCDF = 0;
  for jj = 1:L
     if MapDistance(ii+2,jj+2) ~= 0
        SumCDF = SumCDF + MapDistance(ii+2,jj+2);
        MapDistance(ii+2,jj+2) = SumCDF;
     end
  end  
end

%%% Create Map with position 
Pos_Map = MapDistance;
for ii = 1:L
    X = MapDistance(ii+2,1);
    Y = MapDistance(ii+2,2);
    for jj = 1:L
         X1 = MapDistance(1,jj+2);
         Y1 = MapDistance(2,jj+2);
         % Searching for Position in the array
         if MapDistance(ii+2,jj+2) ~= 0
            for kk = 1:L 
                X2 = MapDistance(kk+2,1);
                Y2 = MapDistance(kk+2,2);
                if (X2 == X1) && (Y2 == Y1)
                    Pos_Map(ii+2,jj+2) = kk;
                    break;
                end
            end
         end
    end
end

%-------------------------------------------------------------
%%%% Structure: x1 y1 x2 y2 x3 y3 ... ----------------------------
%-------------------------------------------------------------
TrajArray = zeros(L*NumRepeatedDatabase, NumPointPerTraj*2);
TrajOrder = zeros(L*NumRepeatedDatabase, NumPointPerTraj); 
%%% Generate random number to choose Trajectory
 for jj = 1:NumRepeatedDatabase % Scan all Traj
    for ii = 1:L % Scan all database
        TrajArray(ii+(jj-1)*L,1) = MapDistance(ii+2,1); % Point 1
        TrajArray(ii+(jj-1)*L,2) = MapDistance(ii+2,2);
        
        NextPos = ii;
        TrajOrder(ii+(jj-1)*L,1) = NextPos;
       
        for NumPoint = 2: NumPointPerTraj
            RanNum = rand(1);
            for kk=1:L % Find the neighbour
                if MapDistance(NextPos+2,kk+2) > RanNum
                     TrajArray(ii+(jj-1)*L,(NumPoint-1)*2+1) = MapDistance(1,kk+2); % Next Point 
                     TrajArray(ii+(jj-1)*L,(NumPoint-1)*2+2) = MapDistance(2,kk+2);
                     NextPos = Pos_Map(NextPos+2,kk+2);
                     TrajOrder(ii+(jj-1)*L,NumPoint) = NextPos;
                     break;
                end
            end 
            
        end
    end
end
csvwrite([myFolder 'Traj_10points_5k_5m_s.csv'],TrajArray);
