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
	// input matrix B
	// double b[r2][c2] =
	double b[MAXROW][MAXCOL] = 
	{
		{5,12},
		{7,13},
		{14,-3},
		{-2,2}
	}; 
	
	// output matrix
	double c[MAXROW][MAXCOL];
	
	// temporary sum value
	double s;
	
	// choice variable
	char choice;
	
	// user call to decide between default values or input values
	printf("Proceed with default values? (y/n):");
	scanf("%c", &choice);
	
	if(choice == 'y' || choice == 'Y')
	{
		// User call for matrix 1
		printf("Enter the number of rows in Matric A:");
		scanf("%d", &r1);
		printf("Enter the number of columns in Matrix A:");
		scanf("%d", &c1);
		
		// user input for matrix 1
		for(i = 0; i<r1; i++)
		{
			for(j = 0; j<c1; j++)
			{
				printf("ENter the value of a[%d][%d]:", i, j);
				scanf("%lf", &a[i][j]);
			}
		}
		
		// User call for matrix 2
		printf("Enter the number of rows in Matric B:");
		scanf("%d", &r2);
		printf("Enter the number of columns in Matrix B:");
		scanf("%d", &c2);
		
		// user input for matrix 1
		for(i = 0; i<r2; i++)
		{
			for(j = 0; j<c2; j++)
			{
				printf("ENter the value of b[%d][%d]:", i, j);
				scanf("%lf", &b[i][j]);
			}
		} 
	}
	
	if (c1 == r2)
	{
		// matrix multiplication loop
		for(i=0; i<r1; i++)
		{
			for(j=0; j<c2; j++)
			{
				s = 0.0; // c[i][j] = 0.0 accessing element in array is slowe, hence s = 0.0 is faster
				for(k=0; k<c1; k++)
					s += a[i][k]*b[k][j]; 
				c[i][j] = s;
			}
		}
	}
	printf("The matrix A is :\n");
	// print the matrix
	for(i=0; i<r1; i++)
	{
		printf("|");
		for(j=0; j<c1; j++)
		{
			printf("%g\t", a[i][j]);
		}
		printf("|\n");
	}
	printf("The matrix B is :\n");
	for(i=0; i<r2; i++)
	{
		printf("|");
		for(j=0; j<c2; j++)
		{
			printf("%g\t", b[i][j]);
		}
		printf("|\n");
	}
	printf("The matrix C is :\n");
	for(i=0; i<r1; i++)
	{
		printf("|");
		for(j=0; j<c2; j++)
		{
			printf("%g\t", c[i][j]);
		}
		printf("|\n");
	}
	return 0;
}
