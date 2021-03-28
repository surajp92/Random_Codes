function u = basP3(x,y,vert,ibas,dind)
%% USAGE: evaluate Lagrange P3 basis defined on triangle at point(s):(x,y)
% 
% INPUTS:
% x --- x-coordinate of query point(s). 
% y --- y-coordinate of query point(s). 
%          x and y must be arrays with the same dimension.
% vert --- vertices of a triangle, which are ordered counter-clock-wisely. 
%          vert = [x1 y1; x2 y2; x3 y3];
% ibas --- basis index. possible values are 1,2,3,4,5,6.
%                         A3
%                       /    \
%                     A6      A8
%                    /          \
%                  A9     A10    A5
%                 /                \
%               A1-----A4-----A7----A2
% dind --- derivative indices. First index for derivative of x. Second
%          index for derivative of y. 
%          possible values: [0,0] for f   [1,0] for Dx(f)   [0,1] for Dy(f)
% OUTPUTS:
% u --- basis value(s)

% Last Modified: 05/28/2014 by Xu Zhang
%%
if dind(1) == 0 && dind(2) == 0     
    if ibas <= 3
        lamI0 = basP1(x,y,vert,ibas,[0,0]); 
        u = (1/2)*lamI0.*(3*lamI0-1).*(3*lamI0-2);
    elseif ibas >= 4 && ibas <= 6
        i = ibas - 3; j = (i<3)*(i+1)+(i+1>3)*mod(i+1,3);
        lamI0 = basP1(x,y,vert,i,[0,0]); 
        lamJ0 = basP1(x,y,vert,j,[0,0]); 
        u = (9/2)*lamI0.*lamJ0.*(3*lamI0-1);
    elseif ibas >= 7 && ibas <= 9
        i = ibas - 6; j = (i<3)*(i+1)+(i+1>3)*mod(i+1,3);
        lamI0 = basP1(x,y,vert,i,[0,0]); 
        lamJ0 = basP1(x,y,vert,j,[0,0]); 
        u = (9/2)*lamI0.*lamJ0.*(3*lamJ0-1);
    elseif ibas == 10
        lamI0 = basP1(x,y,vert,1,[0,0]); 
        lamJ0 = basP1(x,y,vert,2,[0,0]); 
        lamK0 = basP1(x,y,vert,3,[0,0]); 
        u = 27*lamI0.*lamJ0.*lamK0;
    end
elseif (dind(1) == 1 && dind(2) == 0) || (dind(1) == 0 && dind(2) == 1)  
    if ibas <= 3
        lamI0 = basP1(x,y,vert,ibas,[0,0]); 
        lamI1 = basP1(x,y,vert,ibas,dind); 
        u = lamI1.*(27/2*lamI0.^2-9*lamI0+1);
    elseif ibas >= 4 && ibas <= 6
        i = ibas - 3; j = (i<3)*(i+1)+(i+1>3)*mod(i+1,3);
        lamI0 = basP1(x,y,vert,i,[0,0]);
        lamI1 = basP1(x,y,vert,i,dind);
        lamJ0 = basP1(x,y,vert,j,[0,0]); 
        lamJ1 = basP1(x,y,vert,j,dind); 
        u = 9/2*(lamI1.*lamJ0.*(3*lamI0-1) + lamI0.*lamJ1.*(3*lamI0-1)...
            +lamI0.*lamJ0.*3.*lamI1);
    elseif ibas >= 7 && ibas <= 9
        i = ibas - 6; j = (i<3)*(i+1)+(i+1>3)*mod(i+1,3);
        lamI0 = basP1(x,y,vert,i,[0,0]); 
        lamI1 = basP1(x,y,vert,i,dind); 
        lamJ0 = basP1(x,y,vert,j,[0,0]); 
        lamJ1 = basP1(x,y,vert,j,dind); 
        u = 9/2*(lamI1.*lamJ0.*(3*lamJ0-1) + lamI0.*lamJ1.*(3*lamJ0-1)...
            +lamI0.*lamJ0.*3.*lamJ1);
    elseif ibas == 10
        lamI0 = basP1(x,y,vert,1,[0,0]); 
        lamI1 = basP1(x,y,vert,1,dind); 
        lamJ0 = basP1(x,y,vert,2,[0,0]); 
        lamJ1 = basP1(x,y,vert,2,dind); 
        lamK0 = basP1(x,y,vert,3,[0,0]); 
        lamK1 = basP1(x,y,vert,3,dind); 
        u = 27*(lamI1.*lamJ0.*lamK0 + lamI0.*lamJ1.*lamK0...
            +lamI0.*lamJ0.*lamK1);
    end
end