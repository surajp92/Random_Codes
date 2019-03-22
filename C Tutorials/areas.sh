rm *.o *.exe *.dat *.png

# comilation and building the executable
gcc -c -Wall areas.c
gcc -o areas.exe areas.o -Wall -lm
./areas.exe


 #General purpose
 #gcc -c [flags] filename1.c
 #gcc -c [flags] filename2.c
 #gcc -c [flags] filename3.c
 #...............
 #gcc -c [flags] filenameN.c
 #gcc -o executable [set of *.o files] [flags]
 #./executable
