function u = bas2D(x,y,vert,pd,ibas,dind)
%% USAGE: evaluate Lagrange P2 basis defined on triangle at point(s):(x,y)
%
% INPUTS:
% x --- x-coordinate of query point(s).
% y --- y-coordinate of query point(s).
%          x and y must be arrays with the same dimension.
% vert --- vertices of a triangle, which are ordered counter-clock-wisely.
%          vert = [x1 y1; x2 y2; x3 y3];
% pd --- polynomial degree. possible value = 1,2,3.
% ibas --- basis index.
% dind --- derivative indices. First index for derivative of x. Second
%          index for derivative of y.
%          possible values: [0,0] for f   [1,0] for Dx(f)   [0,1] for Dy(f)
%
% OUTPUTS:
% f --- basis value(s)
% Last Modified: 05/28/2014 by Xu Zhang
%%
if pd == 1
    u = basP1(x,y,vert,ibas,dind);
elseif pd == 2
    u = basP2(x,y,vert,ibas,dind);
elseif pd == 3
    u = basP3(x,y,vert,ibas,dind);
end