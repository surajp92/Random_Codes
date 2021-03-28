function [gw, gx, gy] = gaussQuad2D(vert, ng)
%% USAGE: generate Gaussian weights and nodes on arbitrary line segment 
% 
% INPUTS:
% vert --- the vertice of the integral interval [a,b] 
% ng --- number of Gaussian nodes on a straight line
%             possible values = 1,2,3,4,5,6.
% OUTPUTS:
% gw --- Gaussian weigh 
% gx --- Gaussian nodes

% Last Modified: 06/05/2014 by Xu Zhang
%%
[w,a,b,c] = gaussRef2D(ng);

[x1,y1] = deal(vert(1,1), vert(1,2));
[x2,y2] = deal(vert(2,1), vert(2,2));
[x3,y3] = deal(vert(3,1), vert(3,2));

T = 0.5*abs(x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2);

gw = w*T;
gx = a*x1 + b*x2 + c*x3;
gy = a*y1 + b*y2 + c*y3;
