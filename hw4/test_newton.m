function x=multinewton(f,x,NumIters)
%Performs multidimensional Newton’s method for the function defined in f
%starting with x and running NumIters times.
[y,dy]=f(x);
for j=1:NumIters
    s=dy\y;
    x=x-s;
    [y,dy]=f(x);
end
