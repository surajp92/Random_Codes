function f = basP1(x,y,vert,ibas,dind)
%% USAGE: evaluate Lagrange P1 basis defined on triangle at point(s):(x,y)
% 
% INPUTS:
% x --- x-coordinate of query point(s). 
% y --- y-coordinate of query point(s). 
%          x and y must be arrays with the same dimension.
% vert --- vertices of a triangle, which are ordered counter-clock-wisely. 
%          vert = [x1 y1; x2 y2; x3 y3];
% ibas --- basis index. possible values are 1,2,3.
% dind --- derivative indices. First index for derivative of x. Second
%          index for derivative of y. 
%          possible values: [0,0] for f   [1,0] for Dx(f)   [0,1] for Dy(f)
% OUTPUTS:
% f --- basis value(s)

% Last Modified: 03/05/2021 by Xu Zhang
%%
x1 = vert(1,1); x2 = vert(2,1); x3 = vert(3,1); 
y1 = vert(1,2); y2 = vert(2,2); y3 = vert(3,2);
denom = x1*y2+x2*y3+x3*y1-x1*y3-x2*y1-x3*y2;

if ibas == 1
    a = x2*y3-x3*y2;
    b = y2-y3;
    c = x3-x2;
elseif ibas == 2
    a = x3*y1-x1*y3;
    b = y3-y1;
    c = x1-x3;
elseif ibas == 3
    a = x1*y2-x2*y1;
    b = y1-y2;
    c = x2-x1;
end

if dind(1) == 0 && dind(2) == 0  
    f = (a + b*x + c*y)./denom;
elseif dind(1) == 1 && dind(2) == 0
    f = b*ones(size(x))./denom;
elseif dind(1) == 0 && dind(2) == 1
    f = c*ones(size(y))./denom;
end
