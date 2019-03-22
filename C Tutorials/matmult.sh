rm *.o *.exe *.dat *.png

# comilation and building the executable
gcc -c matmult.c
gcc -o matmult.exe matmult.o #- Wall
./matmult.exe


 #General purpose
 #gcc -c [flags] filename1.c
 #gcc -c [flags] filename2.c
 #gcc -c [flags] filename3.c
 #...............
 #gcc -c [flags] filenameN.c
 #gcc -o executable [set of *.o files] [flags]
 #./executable
