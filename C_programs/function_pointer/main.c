////////////////////////////////////////////////////////////////////////////////
// File: gauss_legendre_5pts.c                                                //
// Routines:                                                                  //
//    double Gauss_Legendre_Integration_5pts( double a, double b,             //
//                                                     double (*f)(double) )  //
//    void   Gauss_Legendre_Zeros_5pts( double zeros[] )                      //
//    void   Gauss_Legendre_Coefs_5pts( double coef[] )                       //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// The n-th Legendre polynomial is                                            //
//                 Pn(x) = 1/(2^n n!) (d/dx)^n (x^2-1)^n.                     //
// For the n point Gauss-Legendre integral approximation formula the          //
// coefficients are A[i] = 2 (1 - x[i]^2) / (n P(n-1)(x[i])^2 where x[i] is   //
// a zero of the n-th Legendre polynomial Pn(x).                              //
// Note that if x is a zero of Pn(x) then -x is also a zero of Pn(x) and the  //
// coefficients associated with x and -x are equal.                           //
////////////////////////////////////////////////////////////////////////////////

static const double x[] = {
    0.00000000000000000000e+00,    5.38469310105683091018e-01,
    9.06179845938663992811e-01
};

static const double A[] = {
    5.68888888888888888883e-01,    4.78628670499366468030e-01,
    2.36926885056189087515e-01
};

////////////////////////////////////////////////////////////////////////////////
//  double Gauss_Legendre_Integration_5pts( double a, double b,               //
//                                                      double (*f)(double))  //
//                                                                            //
//  Description:                                                              //
//     Approximate the integral of f(x) from a to b using the 5 point Gauss-  //
//     Legendre integral approximation formula.                               //
//                                                                            //
//  Arguments:                                                                //
//     double  a   Lower limit of integration.                                //
//     double  b   Upper limit of integration.                                //
//     double *f   Pointer to function of a single variable of type double.   //
//                                                                            //
//  Return Values:                                                            //
//     The integral of f from a to b.                                         //
//                                                                            //
//  Example:                                                                  //
//     {                                                                      //
//        double f(double);                                                   //
//        double integral, lower_limit, upper_limit;                          //
//                                                                            //
//        (determine lower and upper limits of integration)                   //
//        integral = Gauss_Legendre_Integration_5pts(lower_limit,             //
//                                                          upper_limit, f);  //
//        ...                                                                 //
//     }                                                                      //
//     double f(double x) { define f }                                        //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
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


////////////////////////////////////////////////////////////////////////////////
//  void Gauss_Legendre_Zeros_5pts( double zeros[] )                          //
//                                                                            //
//  Description:                                                              //
//     Returns the zeros of the Legendre polynomial P5.                       //
//                                                                            //
//  Arguments:                                                                //
//     double zeros[] Array in which to store the zeros of P5.  This array    //
//                    should be dimensioned 5 in the caller function.         //
//                    The order is from the minimum zero to the maximum.      //
//                                                                            //
//  Return Values:                                                            //
//     none                                                                   //
//                                                                            //
//  Example:                                                                  //
//     #define N 5                                                            //
//     double z[N];                                                           //
//     int i;                                                                 //
//                                                                            //
//     Gauss_Legendre_Zeros_5pts( z );                                        //
//     printf("The zeros of the Legendre polynomial P5 are:");                //
//     for ( i = 0; i < N; i++) printf("%12.6le\n",z[i]);                    //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
void Gauss_Legendre_Zeros_5pts( double zeros[] ) {

  zeros[0] = -x[2];
  zeros[1] = -x[1];
  zeros[2] = x[0];
  zeros[3] = x[1];
  zeros[4] = x[2];
}


////////////////////////////////////////////////////////////////////////////////
//  void Gauss_Legendre_Coefs_5pts( double coef[] )                           //
//                                                                            //
//  Description:                                                              //
//     Returns the coefficients for the 5 point Gauss-Legendre formula.       //
//                                                                            //
//  Arguments:                                                                //
//     double coef[]  Array in which to store the coefficients of the Gauss-  //
//                    Legendre formula.  The coefficient A[i] is associated   //
//                    with the i-th zero as returned in the function above    //
//                    Gauss_Legendre_Zeros_5pts.                              //
//                                                                            //
//  Return Values:                                                            //
//     none                                                                   //
//                                                                            //
//  Example:                                                                  //
//     #define N 5                                                            //
//     double a[N];                                                           //
//     int i;                                                                 //
//                                                                            //
//     Gauss_Legendre_Coefs_5pts( a );                                        //
//     printf("The coefficients for the Gauss-Legendre formula are :\n");     //
//     for (i = 0; i < N; i++) printf("%12.6lf\n",a[i]);                      //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
void Gauss_Legendre_Coefs_5pts( double coefs[]) {

   coefs[0] = A[2];
   coefs[1] = A[1];
   coefs[2] = A[0];
   coefs[3] = A[1];
   coefs[4] = A[2];
}


#include <stdio.h>
#include <math.h>
#define PIO2 1.5707963


/* Test function */
float func(float x)
{
    return sin(x); //pow(x,2.0)*(pow(x,4.0) - sin(2.0*x));
    // return x*x*(x*x-2.0)*sin(x);
}

/* Integral of test function func, i.e., test result */
float fint(float x)
{
    return -cos(x);
    // return 4.0*x*(x*x-7.0)*sin(x)-(pow(x,4.0)-14.0*x*x+28.0)*cos(x);
}

int main(void)
{
    float a=1.0,b=3.0,s,t;

    double coefs1[5];

    Gauss_Legendre_Coefs_5pts(coefs1);
    /* for loop execution */
    int i;
    for( i = 0; i < 5; i = i + 1 ){
        printf("%4f\n", coefs1[i]);
    }


    printf("Integral of func computed with QROMB and QTRAP\n\n");
    printf("Actual value of integral is %12.6f\n",fint(b)-fint(a));
    // s=qromb(func,a,b);
    // printf("Result from routine QROMB is %11.6f\n",s);
    t=Gauss_Legendre_Integration_5pts(a,b,func);
    printf("Result from routine QTRAP is %11.6f\n",t);
    return 0;
}

