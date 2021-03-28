fun = @(x,y) 1.0 + sin(4.0.*x) + cos(2.0.*x.*y);

A1 = [0.0,0.0];
A2 = [0.85,0.0];
A3 = [0.45,1.0];
vert = [A1; A2; A3];

ng = [1 3 4 6 7 12 13];
for i = 1:7
    [gw, gx, gy] = gaussQuad2D(vert, ng(i)); 
    f = feval(fun, gx, gy);
    A = sum(gw .* f);
    fprintf('%d-pt Quad = %5.10f\n',ng(i), A)
end