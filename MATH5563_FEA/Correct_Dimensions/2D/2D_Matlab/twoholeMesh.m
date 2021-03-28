%% Two-hole Domain
xmin = -4; xmax = 4; ymin = 0; ymax = 2; 
x1 = xmin; x2 = xmax; x3 = xmax; x4 = xmin;
y1 = ymin; y2 = ymin; y3 = ymax; y4 = ymax;
R1 = [3,4,x1,x2,x3,x4,y1,y2,y3,y4]'; % rectangle 1

x0 = -2; y0 = 1; % center of circle 
r0 = 0.6; % radius of circle
C1 = [1,x0,y0,r0]'; % define the circle
C1 = [C1;zeros(length(R1) - length(C1),1)];

x0 = 2; y0 = 1; % center of ellipse
rx = 0.5; ry = 1; % radius of circle
a = -20;
E1 = [4,x0,y0,rx,ry,a]'; % define the ellipse
E1 = [E1;zeros(length(R1) - length(E1),1)];

gm = [R1,C1,E1];
sf = '(R1-C1)-E1';
ns = char('R1','C1','E1');
ns = ns';
g = decsg(gm,sf,ns);
figure(1)
clf
pdegplot(g,'VertexLabels','on','EdgeLabels','on','FaceLabels','on')
axis off
%% Generate Mesh
hmax = 0.4; % mesh size 
[P,E,T] = initmesh(g,'hmax',hmax); 

T(4,:) = []; 

figure(2)
clf
pdemesh(P,T)
axis equal
axis off