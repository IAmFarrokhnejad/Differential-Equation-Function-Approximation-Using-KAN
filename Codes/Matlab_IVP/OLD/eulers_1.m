% Solve y'(x)=4x+2, y(0)=3
% exact solution y(t)=2x^2+2x+3

clear;

yo=3;       % Initial value
h=0.05;      % Step size

x=0:h:1;    % range

yex=2*x.^2+2*x+3;    % Solution

ysol(1)=yo;

for i=1:(length(x)-1),
    k1=4*x(i)+2;             % Initial approx.
    ysol(i+1)=ysol(i)+k1*h;     % Approximate solution
end;

plot(x, yex, x, ysol);
legend('Exact', 'Approximate');

[x' yex' ysol']




