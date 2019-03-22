/*
 * areas.c
 * 
 * Copyright 2019 Suraj <supawar@okstate.edu>
 * program to find the arra of square and triangle
 * using functions
 * 
 * Date: 22.03.2019
 */

#include <stdio.h>
#include <math.h>

double Area_Triangle()
{
	// sides of the triangle
	double a, b, c;
	double s; // semi perimeter
	double area; 	// area of the triangle
	
	// get the user input for sides
	printf("Enter the sides of the triangle");
	scanf("%lf %lf %lf", &a, &b, &c);
	
	s = (a+b+c)*0.5;	// semi-perimeter
	area = sqrt(s*(s-a)*(s-b)*(s-c));
	
	// return statement
	return area;
}

double Area_Square()
{
	// sides of the triangle
	double s;		// side of the square
	
	// get the user input for sides
	printf("Enter the side of the square");
	scanf("%lf", &s);
	
	// return statement
	return s*s;
}

int main()
{
	char choice;
	double area;
	
	printf("Enter T for triangle and S for squuare: (t/s): ");
	scanf("%c", &choice);
	
	if(choice == 't' || choice == 'T')
	{
		area = Area_Triangle();
		printf("The area of the triangle is: %g\n", area);
	}
	else if(choice == 's' || choice == 'S')
	{
		area = Area_Square();
		printf("The area of the square is: %g\n", area);
	}
	else 
		printf("unknown option! Exiting... ");
}
