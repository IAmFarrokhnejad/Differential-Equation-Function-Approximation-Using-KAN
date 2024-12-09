% RBF Networks
% ITEC560 Introduction to Neural Networks
% Example 2
% Function Approximation (delta rule)

clear;
strt=tic;

% Network Paramaters

ni=1;                % # of input neurons
nh=32;               % # of hidden neurons
no=1;                % # of output neurons
ns=21;               % # of samples
m=0.02;              % learning rate
maxep=200000000;     % Maximum Epoch

rng=[0, 0.5];     % Range, [min, max]
vi=1/4;            % Initial value/
sig=0.6;         % variance

% Desired error tolerance
er=1e-3;

% Weights of the first layer - centers of the RBF's
w1=8*(-rng(2):((2*rng(2))/(nh-1)):rng(2));

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

m=0.2;
m_inx=(m-0.0001)/maxep;

while  (ep_er>0 & n<maxep),
    
     m=m-m_inx;
    
     n=n+1;
     ep_er=0;
     erav=0;    % Average error
     
    % Generate ns random values inside range
    x = rng(1) + (rng(2)-rng(1))*rand(1,ns);
    
    % Calculate the value of the function;
    d=4*cos(2*pi*x)-(3*pi/2)*sin(6*pi*x);

    % Weigth Update With Delta Rule
    
    for j=1:ns,
             
    % Calculate the outputs of the RB Functions
    q=exp((-(x(j)-w1).^2)/(sig.^2));   
    
    % Calculate the RBF output
    y=w2*q';
    
    % Calculate the error
    
    del1=y-(2/sig^2)*(x(j)-rng(1))*(x(j)-w1)*(w2.*q)'-d(j);
    
    %del=del1*(1-(2/sig^2)*(x(j)-rng(1)))*(x(j)-w1).*q;
       
    %del1=(y-2*(x(j)-rng(1))/(sig^2))*((x(j)-w1)*(w2.*q)');
       
    %   del=-(d(j)-del1);
    
    % New Weights 
    w2=w2-m*del1*q;
    
    %w2=w2-m*del;
    
    delx=sum(del1);
    
    % Check if error is less than the tolerance
          if abs(delx)>er,
            ep_er=ep_er+1;
            erav=erav+abs(delx);
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

t=rng(1):1/50:rng(2);               % time period
ge=(2/pi)*sin(2*pi*t)+(1/4)*cos(6*pi*t);                 % exact fucntion
 

% Approximation of RBF network

for j=1:length(t),
    q=exp((-(t(j)-w1).^2)/(sig^2));
    ga(j)=w2*q';
    gas(j)=vi+(t(j)-rng(1))*ga(j);
end;
    

% Plot functions

plot(t,ge,'r', t, gas,'b--');

% Calculate the estimation error

erplt=sum(abs(ge-gas))/length(t);
disp(['Average error between the original function and it''s approximation is : ', num2str(erplt)]);

[ge' gas']

entr=toc(strt);

disp(['Total time taken is : ' num2str(entr/60), ' minutes.']);

