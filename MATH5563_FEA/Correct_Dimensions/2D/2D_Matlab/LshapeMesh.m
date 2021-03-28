%% L-shape Domain
xmin = -1; xmax = 1; ymin = -1; ymax = 1; 
xmid = 0; ymid = 0;
x1 = xmin; x2 = xmax; x3 = xmax; x4 = xmin;
y1 = ymin; y2 = ymin; y3 = ymax; y4 = ymax;
x5 = xmid; x6 = xmax; x7 = xmax; x8 = xmid;
y5 = ymid; y6 = ymid; y7 = ymax; y8 = ymax;

R1 = [2,4,x1,x2,x3,x4,y1,y2,y3,y4]'; % rectangle 1
R2 = [2,4,x5,x6,x7,x8,y5,y6,y7,y8]'; % rectangle 2
gm = [R1,R2];
sf = 'R1-R2';
ns = char('R1','R2');
ns = ns';
g = decsg(gm,sf,ns);
figure(1)
pdegplot(g,'EdgeLabels','on','FaceLabels','on')
axis off

%% Generate Mesh
hmax = 0.01; % mesh size 
[P,E,T] = initmesh(g,'hmax',hmax); 
%[P,E,T] = refinemesh(g,P,E,T);
T(4,:) = []; 

figure(2)
clf
pdemesh(P,T)
axis equal
axis off