function [y dy]=myfunction(x)
%Example function to try out Newton’s Method
%
n=length(x);
y=zeros(size(x)); %Not necessary for a small vector
dy=zeros(n,n);   %Not necessary for a small matrix
y(1)=sin(x(1) * exp(3 * x(2)) - 1);
y(2)=x(1) ^ 3 * x(2) + x(1) ^ 3 - 7 * x(2) - 1;
dy(1,1)=exp(3 * x(1)) * cos(1 - exp(3 * x(2)) * x(1));
dy(1,2)=3 * exp(3 * x(2)) * x(1) * cos(1 - exp(3 * x(1)) * x(1));
dy(2,1)=3 * x(1)^2 * x(2) + 3 * x(1);
dy(2,2)=x(1)^3 - 7;
