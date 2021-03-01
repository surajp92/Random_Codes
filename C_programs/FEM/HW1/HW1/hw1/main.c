#include <stdio.h>
#include <math.h>
#define PI 3.1415926
#define PIO2 1.5707963

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


int Gaussian_Elimination(float *A, int n, float *B)
{
   int row, i, j, pivot_row;
   float max, dum, *pa, *pA, *A_pivot_row;

      // for each variable find pivot row and perform forward substitution

   pa = A;
   for (row = 0; row < (n - 1); row++, pa += n) {

                       //  find the pivot row

      A_pivot_row = pa;
      max = fabs(*(pa + row));
      pA = pa + n;
      pivot_row = row;
      for (i = row + 1; i < n; pA += n, i++)
         if ((dum = fabs(*(pA + row))) > max) {
            max = dum; A_pivot_row = pA; pivot_row = i;
         }
      if (max == 0.0) return -1;                // the matrix A is singular

        // and if it differs from the current row, interchange the two rows.

      if (pivot_row != row) {
         for (i = row; i < n; i++) {
            dum = *(pa + i);
            *(pa + i) = *(A_pivot_row + i);
            *(A_pivot_row + i) = dum;
         }
         dum = B[row];
         B[row] = B[pivot_row];
         B[pivot_row] = dum;
      }

    // Perform forward substitution

      for (i = row + 1; i < n; i++) {
         pA = A + i * n;
         dum = - *(pA + row) / *(pa + row);
         *(pA + row) = 0.0;
         for (j = row + 1; j < n; j++) *(pA + j) += dum * *(pa + j);
         B[i] += dum * B[row];
      }
   }

    // Perform backward substitution

   pa = A + (n - 1) * n;
   for (row = n - 1; row >= 0; pa -= n, row--) {
      if ( *(pa + row) == 0.0 ) return -1;           // matrix is singular
      dum = 1.0 / *(pa + row);
      for ( i = row + 1; i < n; i++) *(pa + i) *= dum;
      B[row] *= dum;
      for ( i = 0, pA = A; i < row; pA += n, i++) {
         dum = *(pA + row);
         for ( j = row + 1; j < n; j++) *(pA + j) -= dum * *(pa + j);
         B[i] -= dum * B[row];
      }
   }
   return 0;
}

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

//float psid(float x){return (exp(1));}
//float dpsid(float x){return (0.0);}

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

float bd1(float x){return (ax(x)*dpsi1(x)*dpsid(x) + cx(x)*psi1(x)*psid(x));}
float bd2(float x){return (ax(x)*dpsi2(x)*dpsid(x) + cx(x)*psi2(x)*psid(x));}
float bd3(float x){return (ax(x)*dpsi3(x)*dpsid(x) + cx(x)*psi3(x)*psid(x));}

float bn1(float x){return 3.0*psi1(0);}
float bn2(float x){return 3.0*psi2(0);}
float bn3(float x){return 3.0*psi3(0);}


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
    float a=0.0,b=1.0,g=3.0;
    int N = 3;
    int i, j;
    float S[N][N],M[N][N],A[N][N],AC[N][N];
    float f[N],bd[N],bn[N],B[N],BC[N];
    int nx = 100;
    float dx;
    float ug[nx+1],x[nx+1];

    dx = (b-a)/nx;

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

    f[0] = Gauss_Legendre_Integration_5pts(a,b,f1);
    f[1] = Gauss_Legendre_Integration_5pts(a,b,f2);
    f[2] = Gauss_Legendre_Integration_5pts(a,b,f3);

    bd[0] = Gauss_Legendre_Integration_5pts(a,b,bd1);
    bd[1] = Gauss_Legendre_Integration_5pts(a,b,bd2);
    bd[2] = Gauss_Legendre_Integration_5pts(a,b,bd3);

    bn[0] = Gauss_Legendre_Integration_5pts(a,b,bn1);
    bn[1] = Gauss_Legendre_Integration_5pts(a,b,bn2);
    bn[2] = Gauss_Legendre_Integration_5pts(a,b,bn3);

    for (i = 0; i<N; i++){
        B[i] = f[i] - bd[i] - bn[i];
    }

    for (i = 0; i<N; i++){
        for (j = 0; j<N; j++){
            A[i][j] = S[i][j] + M[i][j];
        }
    }

    printf("A\n");
    printmatrix(N,(float *)A);

    for (i = 0; i < N; i++){
        printf("f[%d] = %f\n",i,f[i]);
    }

    for (i = 0; i < N; i++){
        printf("bd[%d] = %f\n",i,bd[i]);
    }

    for (i = 0; i < N; i++){
        printf("bn[%d] = %f\n",i,bn[i]);
    }

    // Save matrices A and vector B, the routine Gaussian_Elimination() //
    // Destroys A and the Solution is returned in B.                    //
    for (i = 0; i < N; i++){
     for (j = 0; j < N; j++){
        AC[i][j] = A[i][j];
     }
     BC[i] = B[i];
    }

    for (i = 0; i < N; i++){
        printf("B[%d] = %f\n",i,B[i]);
    }

    // Call Gaussian_Elimination() solution returned in B. //
    Gaussian_Elimination(&A[0][0], N, (float *)B);

    for (i = 0; i < N; i++){
        printf("B[%d] = %f\n",i,B[i]);
    }

    // compute the Galerkin solution
    for (i = 0; i<nx+1; i++){
        x[i] = a + dx*i;
        ug[i] = B[0]*psi1(x[i]) + B[1]*psi2(x[i]) + B[2]*psi3(x[i]) + psid(x[i]);
        //printf("i = %d %4f %4f\n", i, x[i], ug[i]);
    }

    FILE *fp = fopen("solution_1.txt", "w");
    if (fp == NULL) return -1;
    for (i = 0; i<nx+1; i++) {
    // you might want to check for out-of-disk-space here, too
    fprintf(fp, "%f,%f\n", x[i], ug[i]);
    }
    fclose(fp);

    return 0;
}
