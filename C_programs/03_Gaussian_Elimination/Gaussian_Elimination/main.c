////////////////////////////////////////////////////////////////////////////////
// File: gauss_elimination.c                                                  //
// Routines:                                                                  //
//    Gaussian_Elimination                                                    //
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  int Gaussian_Elimination(double *A, int n, double *B)                     //
//                                                                            //
//     Solve the linear system of equations AX=B where A is an n x n matrix   //
//     B is an n-dimensional column vector (n x 1 matrix) for the             //
//     n-dimensional column vector (n x 1 matrix) X.                          //
//                                                                            //
//     This routine performs partial pivoting and the elements of A are       //
//     modified during computation.  The result X is returned in B.           //
//     If the matrix A is singular, the return value of the function call is  //
//     -1. If the solution was found, the function return value is 0.         //
//                                                                            //
//  Arguments:                                                                //
//     double *A      On input, the pointer to the first element of the       //
//                    matrix A[n][n].  On output, the matrix A is destroyed.  //
//     int     n      The number of rows and columns of the matrix A and the  //
//                    dimension of B.                                         //
//     double *B      On input, the pointer to the first element of the       //
//                    vector B[n].  On output, the vector B is replaced by the//
//                    vector X, the solution of AX = B.                       //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix A is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N], B[N];                                                  //
//                                                                            //
//     (your code to create the matrix A and vector B )                       //
//     err = Gaussian_Elimination((double*)A, NROWS, B);                      //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else { printf(" The Solution is: \n"); ...                             //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
#include <math.h>                                     // required for fabs()

int Gaussian_Elimination(double *A, int n, double *B)
{
   int row, i, j, pivot_row;
   double max, dum, *pa, *pA, *A_pivot_row;

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



////////////////////////////////////////////////////////////////////////////////
// File: testgauss_elimination.c                                              //
// Purpose:                                                                   //
//    Test Gaussian_Elimination() in the file gauss_elimination.c.            //
////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>

int Gaussian_Elimination(double *A, int n, double *B);

static void print_results(void);

#define N 3

double A[N][N], AC[N][N], B[N], BC[N];

int main()
{
   double rows[3][3] = { {5.0, 7.0, 3.0}, {7.0, 11.0, 2.0}, {3.0, 2.0, 6.0} };
   double rhs[3] = {2.0, 3.0, 5.0};
   int matrices[6][3] = { {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}};
   int matrix, row, col;
   int i,j,k,n;

   n = N;
   printf("Prog: testgauss_elimination.c - Gaussian_Elimination\n\n\n");

   for (matrix = 0; matrix < 6; matrix++) {

            // Create the matrix A and vector B, then solve AX = B //

      for (i = 0; i < 3; i++) {
         row = matrices[matrix][i];
         for (col = 0; col < 3; col++) A[i][col] = rows[row][col];
         B[i] = rhs[row];
      }

     // Save matrices A and vector B, the routine Gaussian_Elimination() //
     // Destroys A and the Solution is returned in B.                    //

      for (i = 0; i < n; i++) {
         for (j = 0; j < n; j++) {
            AC[i][j] = A[i][j];
         }
         BC[i] = B[i];
      }

            // Call Gaussian_Elimination() solution returned in B. //

      Gaussian_Elimination(&A[0][0], n, (double *)B);

                              // Print Results //

      print_results();
   }
   return 0;
}

void print_results(void) {
   int i,j;
   double sum;

   printf("******************** Solve Ax = B ********************\n\n");
   printf("where A = \n");
   for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) printf("%6.3f   ", AC[i][j]);
      printf("\n");
   }
   printf("and B = \n");
   for (i = 0; i < N; i++) printf("%6.3f   ", BC[i]);
   printf("\n\n");
   printf("The solution is x = \n");
   for (i = 0; i < N; i++) printf("%6.3f   ", B[i]);
   printf("\n\n");
   printf("Check solution Ax \n");
   for (i = 0; i < N; i++) {
      sum = 0.0;
      for (j = 0; j < N; j++) sum += AC[i][j] * B[j];
      printf("%6.3f   ", sum);
   }
   printf("\n\n\n");
   return;
}
