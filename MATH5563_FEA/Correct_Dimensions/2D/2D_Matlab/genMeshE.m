function [e,t_e] = genMeshE(t)
%% Usage: Form edge info of a triangular or rectangular mesh.
%
% INPUTS:
% t --- nt-by-3(4) array, stores nodes for each triangle (rectangle)
%
% OUTPUTS:
% e --- ne-by-2 array, stores the nodes of each edge.
% t_e --- nt-by-3(4) array, stores the edge indices of each element.
%
% Last Modified: 03/12/2021 by Xu Zhang

%%
nt = size(t,1);
if size(t,2) == 3 % triangular mesh      
    eAll = [t(:,[1,2]);t(:,[2,3]);t(:,[3,1])];
    eAll2 = sort(eAll,2);
    [e,~,ic] = unique(eAll2,'rows','stable');
    t_e = reshape(ic,nt,3);
elseif size(t,2) == 4 % rectangular mesh    
    eAll = [t(:,[1,2]);t(:,[2,3]);t(:,[3,4]);t(:,[4,1])];
    eAll2 = sort(eAll,2);
    [e,~,ic] = unique(eAll2,'rows','stable');
    t_e = reshape(ic,nt,4);
end