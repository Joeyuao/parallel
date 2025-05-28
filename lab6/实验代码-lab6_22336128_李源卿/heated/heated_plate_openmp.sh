#! /bin/bash
#
gcc -c -Wall -fopenmp pthread_heated_plate.c
if [ $? -ne 0 ]; then
  echo "Compile error."
  exit
fi
#
gcc -fopenmp pthread_heated_plate.o -lm
if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi
rm pthread_heated_plate.o
mv a.out $HOME/binc/pthread_heated_plate
#
echo "Normal end of execution."
