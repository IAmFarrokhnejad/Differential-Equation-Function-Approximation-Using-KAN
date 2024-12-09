% Solvin IVP
% F.B.Rizaner & A.Rizaner
%
% In order to use this code or any (modified) part of it in any publication,
% please cite the paper: Rizaner, F.B. & Rizaner, A., "Approximate Solutions of
% Initial Value Problems for Ordinary Differential Equations Using Radial
% Basis Function Networks", Neural Process Lett (2018) 48: 1063.
% https://doi.org/10.1007/s11063-017-9761-9
%
% Brief description: This code estimates the RBF based solution of ex. 1 
% as described in the above paper.
%

clear;
strt=tic;

% Define the functions
% Equation:

% Estimation function
fe=@(a1,a2) -(a1+(1+3*a1^2)/(1+a1+a1^3))*a2+a1^3+2*a1+(a1^2)*(1+3*a1^2)/(1+a1+a1^3);

% Solution
fr=@(a1) a1.^2+exp(-(a1.^2)/2)./(a1.^3+a1+1);

rng=[0, 1];      % Range, [min, max]
vi=1;            % Initial value
sig=0.8;         % variance


% Network Paramaters

ni=1;                % # of input neurons
nh=9;               % # of hidden neurons
no=1;                % # of output neurons
ns=21;               % # of samples
ms=0.2;              % learning rate start
me=0.005;            % lerning rate end
maxep=20000000;       % Maximum Epoch
  

% Desired error tolerance
er=1e-4;

% Weights of the first layer - centers of the RBF's
w1=2*(-rng(2):((2*rng(2))/(nh-1)):rng(2));

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

m_inx=(ms-me)/maxep;


while  (ep_er>0 & n<maxep),
    
     m=ms-m_inx;
    
     n=n+1;
     ep_er=0;
     erav=0;    % Average error
     
    % Generate ns random values inside range
    x = rng(1) + (rng(2)-rng(1))*rand(1,ns);
    
    % Weigth Update With Delta Rule
    
    for j=1:ns,
             
    % Calculate the outputs of the RB Functions
    q=exp((-(x(j)-w1).^2)/(sig.^2));   
    
    % Calculate the RBF output
    y=w2*q';
    
    % Calculate the error
    
    wxx=vi+x(j)*y;
    del1=y-(2/sig^2)*(x(j)-rng(1))*(x(j)-w1)*(w2.*q)'-fe(x(j),wxx);
    
    
    % New Weights 
    w2=w2-m*del1*q;
    
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

t=rng(1):1/20:rng(2);              % time period
ge=fr(t);                   % exact fucntion

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
disp(' ');
disp('Example 1');
disp(['#Hidden Unit : ', num2str(nh)]);

AE=abs(ge-gas);
APE=abs(AE./gas);
SE=(ge-gas).^2;

MAE=mean(AE);
MAPE=mean(APE);
MSE=mean(SE);

disp([' MAE  = ', num2str(MAE)]);
disp([' MAPE = ', num2str(MAPE)]);
disp([' MSE  = ', num2str(MSE)]);

disp(' ');

ResulT=[t' ge' gas' AE' APE' SE'];
disp(ResulT);

entr=toc(strt);

disp(['Total time taken is : ' num2str(entr/60), ' minutes.']);

