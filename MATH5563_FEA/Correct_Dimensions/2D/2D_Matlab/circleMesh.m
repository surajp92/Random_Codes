%% Circular Domain 
x0 = 0; y0 = 0; % center of circle
r0 = pi/4; % radius of circle
C1 = [1,x0,y0,r0]'; % define the circle
gm = C1;
g = decsg(gm);
figure(1)
clf
pdegplot(g,'EdgeLabels','on','FaceLabels','on');
axis equal
axis([-1,1,-1,1])

%% Generate Mesh
hmax = 0.3; % mesh size 
[p,e,t] = initmesh(g,'hmax',hmax); 
%[p,e,t] = refinemesh(g,p,e,t);
t(4,:) = []; 

figure(2)
clf
m = pdemesh(p,t);
axis equal
axis([-1,1,-1,1])

%% Form Mesh Data
mesh.p = p'; 
mesh.t = t';