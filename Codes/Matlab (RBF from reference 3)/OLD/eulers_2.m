% Solve y'(t)=exp(x).*cos(x)+exp(x).*sin(x);, y(2)=3
% exact solution y(t)=exp(t).*sin(t);
% range [2, 6]

clear;

yo=6.7188;       % Initial value
h=0.01;      % Step size

x=2:h:6;    % range

yex=exp(x).*sin(x);    % Solution

ysol(1)=yo;

for i=1:(length(x)-1),
    k1=exp(x(i)).*cos(x(i))+exp(x(i)).*sin(x(i));             % Initial approx.
    ysol(i+1)=ysol(i)+k1*h;     % Approximate solution
end;

plot(x, yex, x, ysol);
legend('Exact', 'Approximate');

[x' yex' ysol']




