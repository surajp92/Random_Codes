static const double x[] = {
    0.00000000000000000000e+00,    5.38469310105683091018e-01,
    9.06179845938663992811e-01
};

static const double A[] = {
    5.68888888888888888883e-01,    4.78628670499366468030e-01,
    2.36926885056189087515e-01
};

float Gauss_Legendre_Integration_5pts(float a, float b, float (*f)(float))
{
   float integral;
   float c = 0.5 * (b - a);
   float d = 0.5 * (b + a);
   float dum;

   dum = c * x[2];
   integral = A[2] * ((*f)(d - dum) + (*f)(d + dum));
   dum = c * x[1];
   integral += A[1] * ((*f)(d - dum) + (*f)(d + dum));
   integral += A[0] * (*f)(d);

   return c * integral;
}

float product(float x, float (*f1)(float))
{
    return (*f1)(x);
}


#include <stdio.h>
#include <math.h>
#define PI 3.1415926
#define PIO2 1.5707963

float ax(float x){return (2 + cos(x));}
float cx(float x){return (1 + pow(x,2));}
float funF(float x){return (exp(x)*(-4-x+pow(x,3)-(2+x)*cos(x)+(1+x)*sin(x)));}

float psi1(float x){return (sin(PI*x));}
float psi2(float x){return (pow(x,2)-x);}
float psi3(float x){return (exp(x)-exp(1));}

float dpsi1(float x){return (PI*cos(PI*x));}
float dpsi2(float x){return (2*x-1);}
float dpsi3(float x){return (exp(x));}

float psid(float x){return (exp(x));}
float dpsid(float x){return (exp(x));}

float S11(float x){return ax(x)*dpsi1(x)*dpsi1(x);}
float S12(float x){return ax(x)*dpsi1(x)*dpsi2(x);}
float S13(float x){return ax(x)*dpsi1(x)*dpsi3(x);}

float S22(float x){return ax(x)*dpsi2(x)*dpsi2(x);}
float S23(float x){return ax(x)*dpsi2(x)*dpsi3(x);}

float S33(float x){return ax(x)*dpsi3(x)*dpsi3(x);}

float M11(float x){return cx(x)*psi1(x)*psi1(x);}
float M12(float x){return cx(x)*psi1(x)*psi2(x);}
float M13(float x){return cx(x)*psi1(x)*psi3(x);}

float M22(float x){return cx(x)*psi2(x)*psi2(x);}
float M23(float x){return cx(x)*psi2(x)*psi3(x);}

float M33(float x){return cx(x)*psi3(x)*psi3(x);}

float f1(float x){return psi1(x)*funF(x);}
float f2(float x){return psi2(x)*funF(x);}
float f3(float x){return psi3(x)*funF(x);}

float bd1(float x){return ax(x)*dpsi1(x)*dpsid(x) + cx(x)*psi1(x)*psid(x);}
float bd2(float x){return ax(x)*dpsi2(x)*dpsid(x) + cx(x)*psi2(x)*psid(x);}
float bd3(float x){return ax(x)*dpsi3(x)*dpsid(x) + cx(x)*psi3(x)*psid(x);}

float bn1(float g, float x){return g*psi1(x);}
float bn2(float g, float x){return g*psi2(x);}
float bn3(float g, float x){return g*psi3(x);}

/* Test function */
float func(float x)
{
    return sin(PIO2*x); //pow(x,2.0)*(pow(x,4.0) - sin(2.0*x));
    // return x*x*(x*x-2.0)*sin(x);
}

float func2(float x)
{
    return func(x);
}

float one(float x)
{
    return 1.0; //pow(x,2.0)*(pow(x,4.0) - sin(2.0*x));
}

/* Integral of test function func, i.e., test result */
float fint(float x)
{
    return -cos(PIO2*x)/PIO2;
    // return 4.0*x*(x*x-7.0)*sin(x)-(pow(x,4.0)-14.0*x*x+28.0)*cos(x);
}

void printmatrix(int N, float *matrix)
{
    int row, columns;

    for (row=0; row<N; row++)
    {
        for(columns=0; columns<N; columns++)
        {
             //printf("%4f     ", *(matrix[row][columns]));
             printf("%4f     ", *((matrix+row*N) + columns));
        }
        printf("\n");
    }
}

int main(void)
{
    float a=0.0,b=1.0,s,t;
    int N = 3;
    int row, columns;
    float S[N][N],M[N][N],A[N][N];
    float f[N][1],bd[N][1],bn[N][1];

    S[0][0] = Gauss_Legendre_Integration_5pts(a,b,S11);
    S[0][1] = Gauss_Legendre_Integration_5pts(a,b,S12);
    S[0][2] = Gauss_Legendre_Integration_5pts(a,b,S13);

    S[1][0] = S[0][1];
    S[1][1] = Gauss_Legendre_Integration_5pts(a,b,S22);
    S[1][2] = Gauss_Legendre_Integration_5pts(a,b,S23);

    S[2][0] = S[0][2];
    S[2][1] = S[1][2];
    S[2][2] = Gauss_Legendre_Integration_5pts(a,b,S33);

    M[0][0] = Gauss_Legendre_Integration_5pts(a,b,M11);
    M[0][1] = Gauss_Legendre_Integration_5pts(a,b,M12);
    M[0][2] = Gauss_Legendre_Integration_5pts(a,b,M13);

    M[1][0] = M[0][1];
    M[1][1] = Gauss_Legendre_Integration_5pts(a,b,M22);
    M[1][2] = Gauss_Legendre_Integration_5pts(a,b,M23);

    M[2][0] = M[0][2];
    M[2][1] = M[1][2];
    M[2][2] = Gauss_Legendre_Integration_5pts(a,b,M33);

    printf("M\n");
    printmatrix(N,(float *)M);

    for (int i = 0; i<N; i++)
    {
        for (int j = 0; j<N; j++)
        {
            A[i][j] = S[i][j] + M[i][j];
        }
    }

    printf("A\n");
    printmatrix(N,(float *)A);

    return 0;
}
