%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Median Filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% N: Window Size
% RSSI_Before: Array
% RSSI_After : Scalar
% n: number of available samples in the RSSI_Before

function RSSI_After = Median_Filter(RSSI_Before, n)
  
  % 
  if n < length(RSSI_Before) % if don't have enough samples
      RSSI_After =   RSSI_Before(n);
  else % have enough sample
      RSSI_After = median(RSSI_Before); 
  end
  
end