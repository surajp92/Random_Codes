/*
 * fibonacci.c
 * 
 * Program to generate the fibonacci sequence
 * Example of for loop and arrays
 * fn = f(n-1) + f(n-2)
 * Link: https://www.youtube.com/watch?v=6Gy7_dINRBc&index=22&list=PLNmACol6lYY5YHYV1GXS7Us301CH2n6hd
 * Author: Suraj Pawar
 * 
 */

#include<stdio.h>

int main()
{
	// iterating variable and number of terms
	int i , n;
	
	// creating an array to store thesequence
	unsigned int fib_array[100]; // has 100 entries
	
	// user call
	printf("Enter the number of tems in fibonacci series:");
	scanf("%d", &n);
	
	// initializing the first two terms of the array and sequence
	fib_array[0] = 0;	// first term in the assay
	fib_array[1] = 1;	// second term in the array
	
	
	// printing out first two terms
	printf("The fibonacci sequence upto %d terms is: \n", n);
	printf("%u\t%u\t", fib_array[0], fib_array[1]);
	
	// generating and printing the temrs for n>2
	if(n>2)
	{
		for(i = 2; i<n; i++)
		{
			fib_array[i] = fib_array[i-1] + fib_array[i-2];
			printf("%u\t", fib_array[i]);  
		}
	}
	return 0;
}
