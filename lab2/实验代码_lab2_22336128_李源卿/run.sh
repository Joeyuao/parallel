mpicc main.c -o main 
num_process=(1 2 4 8 16)
for i in "${num_process[@]}"; do
    mpirun -np $i ./main
done
