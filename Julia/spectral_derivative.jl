using PyPlot
using FFTW

function jacobiandealiased(nx,ny,dx,dy,wf,k2)
    eps = 1.0e-6
    kx = Array{Float64}(undef,nx)
    ky = Array{Float64}(undef,ny)

    #wave number indexing
    hx = 2.0*pi/(nx*dx)

    for i = 1:Int64(nx/2)
        kx[i] = hx*(i-1.0)
        kx[i+Int64(nx/2)] = hx*(i-Int64(nx/2)-1)
    end
    kx[1] = eps
    ky = transpose(kx)

    j1f = zeros(ComplexF64,nx,ny)
    j2f = zeros(ComplexF64,nx,ny)
    j3f = zeros(ComplexF64,nx,ny)
    j4f = zeros(ComplexF64,nx,ny)

    # x-derivative
    for i = 1:nx for j = 1:ny
        j1f[i,j] = 1.0im*wf[i,j]*kx[i]/k2[i,j]
        j4f[i,j] = 1.0im*wf[i,j]*kx[i]
    end end

    # y-derivative
    for i = 1:nx for j = 1:ny
        j2f[i,j] = 1.0im*wf[i,j]*ky[j]
        j3f[i,j] = 1.0im*wf[i,j]*ky[j]/k2[i,j]
    end end

    nxe = Int64(floor(nx*2/3))
    nye = Int64(floor(ny*2/3))

    for i = Int64(floor(nxe/2)+1):Int64(nx-floor(nxe/2)) for j = 1:ny
        j1f[i,j] = 0.0
        j2f[i,j] = 0.0
        j3f[i,j] = 0.0
        j4f[i,j] = 0.0
    end end

    for i = 1:nx for j = Int64(floor(nye/2)+1):Int64(ny-floor(nye/2))
        j1f[i,j] = 0.0
        j2f[i,j] = 0.0
        j3f[i,j] = 0.0
        j4f[i,j] = 0.0
    end end

    j1 = real(ifft(j1f))
    j2 = real(ifft(j2f))
    j3 = real(ifft(j3f))
    j4 = real(ifft(j4f))
    jacp = zeros(Float64,nx,ny)

    for i = 1:nx for j = 1:ny
        jacp[i,j] = j1[i,j]*j2[i,j] - j3[i,j]*j4[i,j]
    end end

    jacpf = fft(jacp)

    jf = zeros(ComplexF64,nx,ny)

    jf[1:Int64(nx/2),1:Int64(ny/2)] = jacpf[1:Int64(nx/2),1:Int64(ny/2)]
    jf[Int64(nx/2+1):nx,1:Int64(ny/2)] = jacpf[Int64(nxe-nx/2+1):nxe,1:Int64(ny/2)]
    jf[1:Int64(nx/2),Int64(ny/2+1):ny] = jacpf[1:Int64(nx/2),Int64(nye-ny/2+1):nye]
    jf[Int64(nx/2+1):nx,Int64(ny/2+1):ny] =  jacpf[Int64(nxe-nx/2+1):nxe,Int64(nye-ny/2+1):nye]

    jf = jf*(nx*ny)/(nxe*nye)

    return j1f_padded, j1f
end

nx = 8
ny = 8
x_l = 0.0
x_r = 2.0*pi
y_b = 0.0
y_t = 2.0*pi
dx = (x_r-x_l)/nx
dy = (y_t-y_b)/ny

function wavespace(nx,ny,dx,dy)
    eps = 1.0e-6

    kx = Array{Float64}(undef,nx)
    ky = Array{Float64}(undef,ny)

    k2 = Array{Float64}(undef,nx,ny)

    #wave number indexing
    hx = 2.0*pi/(nx*dx)

    for i = 1:Int64(nx/2)
        kx[i] = hx*(i-1.0)
        kx[i+Int64(nx/2)] = hx*(i-Int64(nx/2)-1)
    end
    kx[1] = eps
    ky = kx

    for i = 1:nx for j = 1:ny
        k2[i,j] = kx[i]^2 + ky[j]^2
    end end

    return k2
end

k2 = wavespace(nx,ny,dx,dy)

x = zeros(Float64,nx)
y = zeros(Float64,nx)
u = zeros(Float64,nx,ny)

for i = 1:nx
    x[i] = dx*(i-1)
    y[i] = dx*(i-1)
end

for i = 1:nx for j = 1:ny
    u[i,j] = sin(x[i]) + cos(y[j])
end end

uf = fft(u)
j1fp,j1f = jacobiandealiased(nx,ny,dx,dy,uf,k2)

# ux = real(ifft(uxf))
# uy = real(ifft(uyf))

# fig = figure("An example", figsize=(3,3));
# ax1 = fig[:add_subplot](1,1,1);
# cs = ax1.contourf(x, y, transpose(ux),cmap="YlGnBu",
#                 interpolation="bilinear")
# fig.colorbar(cs, orientation="horizontal")
# fig.tight_layout()
# fig.savefig("trial.pdf")
