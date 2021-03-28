function mesh = genMesh(domain, nx, ny)
%% Usage: mesh structure of a uniform triangulation of rectangular domain
% INPUTS:
% domain --- [xmin, xmax, ymin, ymax];
% nx --- the number of uniform partition in x direction.
% ny --- the number of uniform partition in y direction.
%
% OUTPUTS:
% mesh --- a struct data contains very rich mesh information.
% mesh.t --- nx*ny-by-3 matrix, contains the global index. On each element,
%            start labelling the vertex opposite to longest edge.
% mesh.p --- (nx+1)*(ny+1)-by-2 vector, contains the coordinate of the nodes
%
%       9----10---11---12       mesh.t = [1 2 5;   nx = 3
%       | \  | \  | \  |                  6 5 2;   ny = 2
%       |  \ |  \ |  \ |                  2 3 6;
%       5----6----7----8                  7 6 3;
%       | \  | \  | \  |                  3 4 7;
%       |  \ |  \ |  \ |                  8 7 4;
%       1----2----3----4                  5 6 9;
%                                        10 9 6;
%                                         ...
% Last Modified: 02/08/2021 by Xu Zhang

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
tt = zeros(nx*ny,4); % 2.1 Form rectangle mesh
k = 1;
for j = 1:ny
    for i = 1:nx
        tt(k,1) = (j-1)*(nx+1) + i;
        tt(k,2) = (j-1)*(nx+1) + i+1;
        tt(k,3) = j*(nx+1) + i+1;
        tt(k,4) = j*(nx+1) + i;
        k = k+1;
    end
end

t = zeros(2*nx*ny,3); % 2.2 Form triangle mesh
for l = 1:nx*ny
    t(2*l-1,1) = tt(l,1);
    t(2*l-1,2) = tt(l,2);
    t(2*l-1,3) = tt(l,4);
    t(2*l,1) = tt(l,3);
    t(2*l,2) = tt(l,4);
    t(2*l,3) = tt(l,2);
end
%% Generate mesh struct
mesh = struct('p',p,'t',t);