function b = localVec(fun, vert, pd, dind, ng)
%% USAGE: generate local vector on a triangle
%            int_T f(x,y) Dv(x,y)dxdy
% INPUTS:
% fun --- a given function
% vert --- 3-by-2 matrix, vertices of the triangle 
% pd --- polynomial degree of FE spaces
% dind --- derivative info for test function
% ng --- number of Gaussian nodes in a triangle 
% OUTPUTS:
% b --- the local vector (pd+1)(pd+2)/2 - by- 1
% Last Modified: 01/31/2021 by Xu Zhang
%%
b = zeros((pd+1)*(pd+2)/2,1);
[gw, gx, gy] = gaussQuadT(vert, ng);
f = feval(fun, gx, gy);
for i = 1:(pd+1)*(pd+2)/2
    Li = bas2D(gx, gy, vert, pd, i, dind);
    b(i) = sum(gw.*f.*Li);
end