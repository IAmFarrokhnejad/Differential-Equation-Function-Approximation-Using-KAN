% Solve y'(t)=4*cos(2*pi*x(i))-(3*pi/2)*sin(6*pi*x(i)), y(0)=1/4
% exact solution y(t)=(2/pi)*sin(2*pi*x)+(1/4)*cos(6*pi*x)
% range [0, .5]

clear;

yo=1/4;       % Initial value
h=0.02;      % Step size

x=0:h:0.5;    % range

yex=(2/pi)*sin(2*pi*x)+(1/4)*cos(6*pi*x);    % Solution

ysol(1)=yo;

for i=1:(length(x)-1),
    k1=4*cos(2*pi*x(i))-(3*pi/2)*sin(6*pi*x(i));             % Initial approx.
    ysol(i+1)=ysol(i)+k1*h;     % Approximate solution
end;

plot(x, yex, x, ysol);
legend('Exact', 'Approximate');

[x' yex' ysol']




