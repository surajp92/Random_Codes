function b = globalVec(fun, mesh, fem, dind, ng)
%% USAGE: generate global vector on a mesh with Lagrange FE.
%                  int_Omega f(x,y) v(x,y)dxdy
% INPUTS:
% fun --- a given function
% mesh --- the mesh data structure
% fem --- the fem data structure contains
% dind --- derivative info for test function
% ng --- number of Gaussian nodes in a triangle
% OUTPUTS:
% b --- global vector.
%
% Last Modified: 01/31/2021 by Xu Zhang
%%
b = zeros(size(fem.p,1),1);
pd = fem.pd;
for k = 1:size(mesh.t,1)   
    k
    vert = mesh.p(mesh.t(k,:),:);
    bK = localVec(fun, vert, pd, dind, ng)
    b(fem.t(k,:)) = b(fem.t(k,:)) + bK;    
end