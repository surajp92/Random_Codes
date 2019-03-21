/*
 * matmult.c
 * 
 * Copyright 2019 Suraj <supawar@okstate.edu>
 * Date : 21.03.2019
 * 
 * Program for matrix multiplication. uses nested for loop and 2D arrays
 * link: https://www.youtube.com/watch?v=dDUyEfgbqEE&list=PLNmACol6lYY5YHYV1GXS7Us301CH2n6hd&index=23
 * 
 */

#include<stdio.h>

// pre-processor based constants - datatype automatically set
// also called as macros
#define MAXROW 10
#define MAXCOL 10
// eg: M_PI for pi

int main()
{
	// iterating variables
	int i, j, k;
	
	// default rows and column sizes
	int r1=4, r2= 4, c1 = 4, c2 = 2;
	
	// input matrix A
	// double a[r1][c1] = 		// will not work because these are variables
	// double a[4][4] = 		// will work because these are plain numbers
	double a[MAXROW][MAXCOL] = 	// will work because these are constant
	{
		{5,-3,-1,  6},
		{7,-2, 0,  1},
		{2, 8, 3, -4},
		{4, 9, 10, 11}
	};
	 
	
	return 0;
}
