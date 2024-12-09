%% Example 2
% Solve y'(t)=-2y(t)+cos(4t) with y0=3

y0 = 3;                  % Initial Condition
h = 0.01;% Time step
t = 0:h:3;               % t goes from 0 to 2 seconds.
% Exact solution, hard to find in this case (in general we won't have it)
yexact = 0.1*cos(4*t)+0.2*sin(4*t)+2.9*exp(-2*t);
ystar = zeros(size(t));  % Preallocate array (good coding practice)

ystar(1) = y0;           % Initial condition gives solution at t=0.

for i=1:(length(t)-1)
    k1 = -2*ystar(i)+cos(4*t(i));  % Previous approx for y gives approx for deriv
    ystar(i+1) = ystar(i) + k1*h; % Approximate solution for next value of y
end

plot(t,yexact,t,ystar);
legend('Exact','Approximate');

[t' yexact' ystar']




