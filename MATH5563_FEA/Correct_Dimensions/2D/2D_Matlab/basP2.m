function u = basP2(x,y,vert,ibas,dind)
%% USAGE: evaluate Lagrange P2 basis defined on triangle at point(s):(x,y)
% 
% INPUTS:
% x --- x-coordinate of query point(s). 
% y --- y-coordinate of query point(s). 
%          x and y must be arrays with the same dimension.
% vert --- vertices of a triangle, which are ordered counter-clock-wisely. 
%          vert = [x1 y1; x2 y2; x3 y3];
% ibas --- basis index. possible values are 1,2,3,4,5,6.
%                         A3
%                        /  \
%                      A6    A5
%                      /       \
%                    A1---A4---A2
% dind --- derivative indices. First index for derivative of x. Second
%          index for derivative of y. 
%          possible values: [0,0] for f   [1,0] for Dx(f)   [0,1] for Dy(f)
%
% OUTPUTS:
% f --- basis value(s)
% Last Modified: 05/28/2014 by Xu Zhang
%%
if dind(1) == 0 && dind(2) == 0 
    if ibas <= 3
        lamI0 = basP1(x,y,vert,ibas,[0,0]); 
        u = lamI0.*(2*lamI0-1);
    elseif ibas >= 4
        i = ibas - 3; j = (i<3)*(i+1)+(i+1>3)*mod(i+1,3);
        lamI0 = basP1(x,y,vert,i,[0,0]); 
        lamJ0 = basP1(x,y,vert,j,[0,0]);
        u = 4*lamI0.*lamJ0;
    end
elseif (dind(1) == 1 && dind(2) == 0) || (dind(1) == 0 && dind(2) == 1)
    if ibas <= 3
        lamI0 = basP1(x,y,vert,ibas,[0,0]); 
        lamI1 = basP1(x,y,vert,ibas,dind); 
        u = lamI1.*(4*lamI0-1);
    elseif ibas >= 4
        i = ibas - 3; j = (i<3)*(i+1)+(i+1>3)*mod(i+1,3);
        lamI0 = basP1(x,y,vert,i,[0,0]); 
        lamJ0 = basP1(x,y,vert,j,[0,0]); 
        lamI1 = basP1(x,y,vert,i,dind); 
        lamJ1 = basP1(x,y,vert,j,dind); 
        u = 4*(lamI1.*lamJ0 + lamI0.*lamJ1);
    end
end