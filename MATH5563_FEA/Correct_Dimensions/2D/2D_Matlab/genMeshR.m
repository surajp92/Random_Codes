function mesh = genMeshR(domain, nx, ny)
%% Usage: nodes and elements info of a uniform rectangular mesh
%
% INPUTS:
% domain --- cubic domain = [xmin, xmax, ymin, ymax, zmin, zmax].
% nx --- the number of uniform partition in x direction.
% ny --- the number of uniform partition in y direction.

% OUTPUTS:
% p --- np-by-2 vector: (x,y,z) coordinates of each node.
% t --- nt-by-4 vector: eight-node indices for each cube, ordered as follows
%
%  13----14----15----16                            t = [1  2  6  5
%   |     |     |     |                                 2  3  7  6
%   |     |     |     |                                 3  4  8  7
%   9----10----11----12  <--Global Mesh                 5  6 10  9
%   |     |     |     |                                 6  7 11 10
%   |     |     |     |                   A4-----A3     7  8 12 11
%   5-----6-----7-----8                    |      |     9 10 14 13
%   |     |     |     |  Local Index -->   |      |    10 11 15 14
%   |     |     |     |                   A1-----A2    11 12 16 15]
%   1-----2-----3-----4
%
% Last Modified: 08/07/2020 by Xu Zhang

%% 0. Initial Setting
xm = domain(1); xM = domain(2); hx = (xM-xm)/nx;
ym = domain(3); yM = domain(4); hy = (yM-ym)/ny;
%% 1. Form p
np = (nx+1)*(ny+1);
p = zeros(np,2);
k = 1;
for j = 0:ny
    for i = 0:nx
        p(k,1) = xm+i*hx;
        p(k,2) = ym+j*hy;
        k = k+1;
    end
end
%% 2. Form t
t = zeros(nx*ny,4); % 2.1 Form rectangle mesh
k = 1;
for j = 1:ny
    for i = 1:nx
        t(k,1) = (j-1)*(nx+1) + i;
        t(k,2) = (j-1)*(nx+1) + i+1;
        t(k,3) = j*(nx+1) + i+1;
        t(k,4) = j*(nx+1) + i;
        k = k+1;
    end
end
%% Generate mesh struct
mesh = struct('p',p,'t',t);