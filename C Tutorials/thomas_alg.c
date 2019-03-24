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

int main()
{
    int N;
    double a[N], b[N], c[N], x[N], q[N];
    int i;
    N = 5;
    for(i=0;i<N;i++)
    {
        printf(i);
    }
    return 0;
}

