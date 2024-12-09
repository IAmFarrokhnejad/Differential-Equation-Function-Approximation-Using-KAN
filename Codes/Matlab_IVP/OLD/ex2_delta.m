% RBF Networks
% ITEC560 Introduction to Neural Networks
% Example 2
% Function Approximation (delta rule)

clear;

% Network Paramaters

ni=1;               % # of input neurons
nh=9;               % # of hidden neurons
no=1;               % # of output neurons
ns=21;              % # of samples
m=0.1;              % learning rate
maxep=10000;          % Maximum Epoch

% Desired error tolerance
er=1e-2;

% Weights of the first layer - centers of the RBF's
w1=-0.8:0.2:0.8;

% Check if leng of w1 matches with number of hidden units
if nh~=length(w1),
    disp(['Number of units in the hidden layer (', num2str(nh), ...
         ') must be equal to the number of weights (', num2str(length(w1)), ...
         ') in the first layer.']);
     return;
end;

% Initial weights for the output layer between -1 & 1
w2=2*rand(1,nh)-1;

% ep_er is for testing if any error has accured in any epoch
ep_er=1; 

% Epoch number
n=0;


while  (ep_er>0 & n<maxep),
    
     n=n+1;
     ep_er=0;
     erav=0;    % Average error
     
    % Generate ns random values between -1 and 1
    x=(2*rand(1,ns)-1);
    
    % Calculate the corresponding desired outputs;
    d=0.5*x+2*x.^2-x.^3;

    % Weigth Update With Delta Rule
    
    for j=1:ns,
             
    % Calculate the outputs of the RB Functions
    q=exp(-(x(j)-w1).^2);   
    
    % Calculate the RBF output
    y=w2*q';
    
    % Calculate the error
    del=d(j)-y;
    
    % New Weights 
    w2=w2+m*del*q;
    
    % Check if error is less than the tolerance
          if abs(del)>er,
            ep_er=ep_er+1;
            erav=erav+abs(del);
          end;
  
    end;
    
    % At the end of each epoch check if all errors are
    % less than the error tolarance
    if ep_er==0,
        disp(['Function is approximated in ', num2str(n), ' epochs.']);
        disp(['The average error is the last epoch is ', num2str(erav/ns), '.']);
    elseif n==maxep,
         disp(['At the end of ', num2str(n), ' epochs.']);
         disp(['The average error in the last epoch is ', num2str(erav/ns), '.']);
    end;
        
end;


% Plot the exact and approximated function together

t=-1:1/20:1;              % time period
ge=0.5*t+2*t.^2-t.^3;     % exact fucntion

% Approximation of RBF network

for j=1:length(t),
    q=exp(-(t(j)-w1).^2);
    ga(j)=w2*q';
end;
    
% Plot functions

plot(t,ge,'r', t, ga,'b--');

% Calculate the estimation error

erplt=sum(abs(ge-ga))/length(t);
disp(['Average error between the original function and it''s approximation is : ', num2str(erplt)]);

