#include <stdio.h>
/* #include <stddef.h> */
#include <stdlib.h>
#define NR_END 1
#define FREE_ARG char*

void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1);
}

#define FUNC(x) ((*func)(x))

float trapzd(float (*func)(float), float a, float b, int n)
{
    float x,tnm,sum,del;
    static float s;
    int it,j;

    if (n == 1) {
            return (s=0.5*(b-a)*(FUNC(a)+FUNC(b)));
    } else {
            for (it=1,j=1;j<n-1;j++) it <<= 1;
            tnm=it;
            del=(b-a)/tnm;              /*This is the spacing of points to be
                                          added. */
            x=a+0.5*del;
            for (sum=0.0,j=1;j<=it;j++,x+=del) sum += FUNC(x);
            s=0.5*(s+(b-a)*sum/tnm);  /*This replaces s by its refined value.*/
            return s;
    }
}
#undef FUNC


#include <math.h>
#define EPS 1.0e-5
#define JMAX 20


float qtrap(float (*func)(float), float a, float b)
{
        float trapzd(float (*func)(float), float a, float b, int n);
        void nrerror(char error_text[]);
        int j;
        float s,olds;

        olds = -1.0e30;
        for (j=1;j<=JMAX;j++) {
                s=trapzd(func,a,b,j);
                if (j > 5)
                        if (fabs(s-olds) < EPS*fabs(olds) ||
                                (s == 0.0 && olds == 0.0)){
                                printf("j = %d\n", j);
                                return s;
                                }
                olds=s;
        }
        printf("s = %11.6f\n", s);
        nrerror("Too many steps in routine qtrap");
        return 0.0;
}
#undef EPS
#undef JMAX

/* Driver for routine qromb */

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

    printf("Integral of func computed with QROMB and QTRAP\n\n");
    printf("Actual value of integral is %12.6f\n",fint(b)-fint(a));
    // s=qromb(func,a,b);
    // printf("Result from routine QROMB is %11.6f\n",s);
    t=qtrap(func,a,b);
    printf("Result from routine QTRAP is %11.6f\n",t);
    return 0;
}
