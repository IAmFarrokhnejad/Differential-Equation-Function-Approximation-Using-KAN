% RBF Networks
% ITEC560 Introduction to Neural Networks
% Example 2
% Function Approximation (Least Squares Estimation)

clear;

% Network Paramaters

ni=1;               % # of input neurons
nh=9;               % # of hidden neurons
no=1;               % # of output neurons
ns=21;              % # of samples


% Weights of the first layer - centers of the RBF's
w1=-0.8:0.2:0.8;

% Check if leng of w1 matches with number of hidden units
if nh~=length(w1),
    disp(['Number of units in the hidden layer (', num2str(nh), ...
         ') must be equal to the number of weights (', num2str(length(w1)), ...
         ') in the first layer.']);
     return;
end;

     
    % Generate ns uniform values between -1 and 1
    x=-1:2/(ns-1):1;
    
    % Calculate the corresponding desired outputs;
    d=0.5*x+2*x.^2-x.^3;

    
    for j=1:ns,
             
    % Calculate the outputs of the RB Functions
    q(j,:)=exp(-(x(j)-w1).^2);   
        
    end;

    
    w2=inv(q'*q)*q'*d';

% Plot the exact and approximated function together

t=-1:1/20:1;              % time period
ge=0.5*t+2*t.^2-t.^3;     % exact fucntion

% Approximation of RBF network

for j=1:length(t),
    qi=exp(-(t(j)-w1).^2);
    ga(j)=w2'*qi';
end;
    
% Plot functions

plot(t,ge,'r', t, ga,'b--');

% Calculate the estimation error

erplt=sum(abs(ge-ga))/length(t);
disp(['Average error between the original function and it''s approximation is : ', num2str(erplt)]);

