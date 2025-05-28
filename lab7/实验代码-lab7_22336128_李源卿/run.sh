
MPI_PROCS=$1
EXECUTABLE="./bk"

# 静默编译
mpic++  -o bk bk.cpp -lm >/dev/null 2>&1 || exit 1

# 测试数据规模
for MPI_PROCS in 1 2 4 8 16; do
    for size in 65536 262144 1048576; do
        echo "Size: $size"
        mpirun -np $MPI_PROCS $EXECUTABLE $size | grep -E "serial time:|cost:|speedup:"
        echo "-----"
    done
done