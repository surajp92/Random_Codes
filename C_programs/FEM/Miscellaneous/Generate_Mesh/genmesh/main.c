#include <stdio.h>
#include <string.h>

#define N 10
//const int N = 4;

typedef struct{
    float p[N+1];
    int t[N][2];
} mesh;

mesh generate_mesh(float a, float b)
{
    mesh myMesh;
    float xmin, xmax;
    int i;
    float dx;
    xmin = a;
    xmax = b;
    dx = (xmax - xmin)/N;

    for (i=0; i<N+1; i++){
        myMesh.p[i] = dx*i;
    }

    for (i=0; i<N; i++){
        myMesh.t[i][0] = i;
        myMesh.t[i][1] = i+1;
    }
    return myMesh;
}


int main()
{
    float a,b;
    int i;
    mesh myMesh;
    a = 0.0;
    b = 2.0;

    myMesh = generate_mesh(a,b);

    for (i=0; i<N; i++){
        printf("Edge %d %d %f %f\n", myMesh.t[i][0], myMesh.t[i][1], myMesh.p[i], myMesh.p[i+1]);
    }
    return 0;
}

