%% Triangle Domain
x1 = -1; x2 = 1; x3 = 0; 
y1 = -1; y2 = -1; y3 = 1; 
T1 = [2,3,x1,x2,x3,y1,y2,y3]'; % triangle 
gm = T1;
g = decsg(gm);
figure(1); clf
pdegplot(g,'EdgeLabels','on','FaceLabels','on');
axis equal
axis([-1.2,1.2,-1.2,1.2])

%% Generate Mesh
hmax = 0.7; % mesh size 
[P,E,T] = initmesh(g,'hmax',hmax); 
T(4,:) = []; 

figure(2); clf
pdemesh(P,T,'ElementLabels','on')
axis equal
axis([-1.2,1.2,-1.2,1.2])

figure(3); clf
pdemesh(P,T,'NodeLabels','on')
axis equal
axis([-1.2,1.2,-1.2,1.2])