function fem = genFEM(mesh,pd)
%% Usage: Form FEM global degrees of freedom on 2D triangular mesh
%
% INPUTS:
% mesh --- a struct data contains rich mesh information.
% pd --- polynomial degree. possible value = 1,2,3.
%
% OUTPUTS:
% fem --- a struct data contains the following fields:
%         fem.p: (x,y) coordinate of each vertex w.r.t a global DoF
%         fem.t: indices of global DoF in each element
%         fem.type: type of finite element methods
%         fem.ldof: number of local DoF on each element
% Last Modified: 03/12/2021 by Xu Zhang
%%
if pd == 1
    pp = mesh.p; tt = mesh.t;
elseif pd == 2
    p = mesh.p; t = mesh.t; e = mesh.e; t_e = mesh.t_e;
    nt = size(t,1); np = size(p,1); ne = size(e,1);
    pp = zeros(np+ne,2);
    pp(1:np,:) = p;
    pp(np+1:end,:) = 1/2*p(e(:,1),:) + 1/2*p(e(:,2),:);
    tt = zeros(nt,6);
    tt(:,1:3) = t;
    tt(:,4:6) = t_e+np;
elseif pd == 3
    p = mesh.p; t = mesh.t; e = mesh.e; t_e = mesh.t_e;
    nt = size(t,1); np = size(p,1); ne = size(e,1);
    pp = zeros(np+2*ne+nt,2);
    pp(1:np,:) = p;
    pp(np+1:np+ne,:) = 2/3*p(e(:,1),:) + 1/3*p(e(:,2),:);
    pp(np+ne+1:np+2*ne,:) = 1/3*p(e(:,1),:) + 2/3*p(e(:,2),:);
    pp(np+2*ne+1:np+2*ne+nt,:) = 1/3*(p(t(:,1),:)+p(t(:,2),:)+p(t(:,3),:));    
    
    eAll = [t(:,[1,2]);t(:,[2,3]);t(:,[3,1])];
    [~,I] = sort(eAll,2);
    ID = find(I(:,1)==2);
    ic1 = t_e; 
    ic1(ID) = ic1(ID) + ne;
    t_e1 = reshape(ic1,nt,3);
    ic2 = t_e+ne; 
    ic2(ID) = ic2(ID) - ne;
    t_e2 = reshape(ic2,nt,3);
    
    tt = zeros(nt,10);
    tt(:,1:3) = t;
    tt(:,4:6) = t_e1+np;
    tt(:,7:9) = t_e2+np;
    tt(:,10) = (1:nt)'+np+2*ne;
end
fem = struct('p',pp, 't',tt, 'pd', pd);