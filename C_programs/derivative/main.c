#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float derivative(float x)
{
    return sin(x);
}

int main()
{
    float a = 1.5707;
    printf("Hello world!\n");
    printf("sin(%f) = %f \n", a, derivative(a));
    return 0;
}
