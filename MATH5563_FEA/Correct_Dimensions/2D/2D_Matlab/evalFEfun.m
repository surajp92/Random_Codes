function f = evalFEfun(x, y, uhK, vert, pd, dind)
%% USAGE: evaluate the FE function at certain point(s).
%             f = u1*L1(x) + u2*L2(x) + ... + ud*Ld(x).
% INPUTS:
% x --- x-coordinates of query points. 
% y --- y-coordinates of query points.
% uhK --- vector [u1,u2,...,ud] of coefficients of FE function.
%          local dimension is d where d = (p+1)(p+2)/2. 
% vert --- vertices of the element (triangle or rectangle)
% pd --- polynomial degree. possible value = 1,2,3.
% dind --- derivative info for basis function
%          dind = [0,0] if f = u1*L1(x,y)+ ... + ud*Ld(x,y)
%          dind = [1,0] if f = u1*DxL1(x,y)+ ... + ud*DxLd(x,y)
%          dind = [0,1] if f = u1*DyL1(x,y)+ ... + ud*DyLd(x,y)
% OUTPUTS:
% f --- the value of the FE function at point x. 
%
% Last Modified: 01/28/2021 by Xu Zhang
%%
d = (pd+1)*(pd+2)/2;
f = zeros(size(x));
for ibas = 1:d
    f = f + uhK(ibas)*bas2D(x, y, vert, pd, ibas, dind);    
end