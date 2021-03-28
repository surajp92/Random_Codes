fun = @(x,y) exp(-x.*y);

A1 = [0.0,0.0];
A2 = [1.0,0.0];
A3 = [0.0,1.0];
vert = [A1; A2; A3];

ng = 7;
pd1 = 1;
pd2 = 1;
d1 = [1,0];
d2 = [0,0];
mat = localMatrix2D(fun, vert, pd1, d1, pd2, d2, ng)
