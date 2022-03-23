% Recursive weighted average fltering algorithm
% RSSI_After: scalar
% RSSI_Before: vector with 3 values

function [RSSI_After, F1, F2, TimeCount] = Average_Filter(RSSI_Before, F1_Before, F2_Before,n)

    beta1 = 0.2;
    beta2 = 0.8;
    beta3 = 0.05;
    beta4 = 0.15;
    beta5 = 0.8;
    
    TimeCount = n;
    F1 = F1_Before;
    F2 = F2_Before;
    
    if n == 1
       F1(1) = RSSI_Before(1);
       F2(1) = RSSI_Before(1);
       RSSI_After = RSSI_Before(1);
    end
    
    if n == 2
        F1(2) = beta1*RSSI_Before(n-1) + beta2*RSSI_Before(n);
        F2(2) = beta1*F1(n-1) + beta2*F1(n);
        F3 = beta1*F2(n-1) + beta2*F2(n);
        RSSI_After = F3;
    end
    
    if n >= 3
        if n > 3
            n = 3;
            F1(1) =  F1(2); F1(2) = F1(3); 
            F2(1) =  F2(2); F2(2) = F2(3); 
        end
        F1(3) = beta3*RSSI_Before(n-2) + beta4*RSSI_Before(n-1)+beta5*RSSI_Before(n);
        F2(3) = beta3*F1(n-2) + beta4*F1(n-1)+beta5*F1(n);
        F3 = beta3*F2(n-2) + beta4*F2(n-1)+beta5*F2(n);
        RSSI_After = F3;
    end
    TimeCount = TimeCount + 1;
end