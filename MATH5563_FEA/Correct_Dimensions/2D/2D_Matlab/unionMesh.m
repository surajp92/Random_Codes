%% Union of Three Domains
R1 = [3,4,-1,1,1,-1,-1,-1,1,1]'; % rectangle 1
C1 = [1,1,0,0.5]'; % circle 1
C1 = [C1;zeros(length(R1) - length(C1),1)];
C2 = [1,-1,0,0.5]'; % circle 2
C2 = [C2;zeros(length(R1) - length(C2),1)];
gm = [R1,C1,C2];
sf = 'R1+C1+C2';
ns = char('R1','C1','C2');
ns = ns';
[g,b] = decsg(gm,sf,ns);
[g,b] = csgdel(g,b); % delete the boundary inside the domain
figure(1)
pdegplot(g,'EdgeLabels','on','FaceLabels','on')
axis off

%% Generate Mesh
hmax = 0.3; % mesh size 
[P,E,T] = initmesh(g,'hmax',hmax); 
[P,E,T] = refinemesh(g,P,E,T);
T(4,:) = []; 
figure(2); clf
pdemesh(P,T)
axis equal;  axis off