/*
 * thomas_alg.c
 * 
 * Copyright 2019 Suraj Pawar <supawar@okstate.edu>
 * 
 * This program solves the tridogonal matrix Ax = b using Thomas algorithm. The three basic steps are 
 * 1. LU decomposition
 * 2. Forward substitution
 * 3. Backward substitution
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void ThomasAlgorithm(int N, double *b, double *a, double *c, double *x, double *q)
{
	int i;
	
	double *l = malloc( sizeof(double) * N);
	double *d = malloc( sizeof(double) * N);
	double *u = malloc( sizeof(double) * N);
	double *y = malloc( sizeof(double) * N);
	
	// c++ syntax
	//double *l, *u, *d, *y;
	//l = new double[N];
	//u = new double[N];
	//d = new double[N];
	//y = new double[N];
	
	/* LU decomposition A = LU*/
	d[0] = a[0];
	u[0] = c[0];
	for (i=0;i<N-2;i++)
	{
		l[i] = b[i]/d[i];
		d[i+1] = a[i+1]-l[i]*u[i];
		u[i+1] = c[i+1];
	}
	l[N-2] = b[N-2]/d[N-2];
	d[N-1] = a[N-1] - l[N-2]*u[N-2];
	
	/* Forward substitution Ly = q */
	y[0] = q[0];
	for (i=1;i<N;i++)
	{
		y[i] = q[i]-l[i-1]*y[i-1];
	}
	
	/* Backward substitution Ux = y */
	x[N-1]  = q[N-1]/d[N-1];
	for (i = N-2; i>=0; i--)
	{
		x[i] = (y[i]-u[i]*x[i+1])/d[i];
	}
	
	free( l );
	free( u );
	free( d );
	free( y );
	
	// c++ syntax
	//delete[] l;
	//delete[] d;
	//delete[] u;	
	//delete[] y;
	return;
}

int main()
{
    int N;
    N = 5;
    double a[N], b[N], c[N], x[N], q[N];
    double *p;
    p = &b[0];
    int i;
    a[0] = 1.0;
    c[0] = 0.0;
    q[0] = 100.0; // left boundary
    x[0] = 0.0;
    for(i=0; i<N-2; i++)
    {
        b[i] = 1.0;
        a[i+1] = -2.0;
        c[i+1] = 1.0;
        q[i+1] = 0.0;
        x[i+1] = 0.0;// initial condition
    }
    // printf("pointer = %g", p[1]);
    b[N-2] = 0.0;
    a[N-1] = 1.0;
    q[N-1] = 50.0; // right boundary
    x[N-1] = 0.0;
    
    ThomasAlgorithm(N, a, b, c, x, q);
    // print arrays a,b,c
    for (i = 0;i<N; i++)
    {
		printf("a[%d] = %g\n", i, x[i]);
		printf("b[%d] = %g\n", i, q[i]);
		//printf("c[%d] = %g\n", i, c[i]);
	}
    return 0;
}

