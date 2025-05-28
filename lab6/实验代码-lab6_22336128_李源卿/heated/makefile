# 编译器设置
CC = gcc
CFLAGS = -Wall -fPIC

# 目标文件
TARGET = pthread_heated_plate
LIBRARY = libparallel_for.so  # 修改库名为lib前缀

# 默认目标
all: $(LIBRARY) $(TARGET)

# 编译共享库
$(LIBRARY): parallel_for.o
	$(CC) -shared -o $@ $^

# 编译主程序并链接共享库
$(TARGET): pthread_heated_plate.o
	$(CC) -o $@ $< -L. -lparallel_for -fopenmp -Wl,-rpath=.

# 生成.o文件
%.o: %.c
	$(CC) $(CFLAGS) -c $<

# 清理
clean:
	rm -f *.o $(TARGET) $(LIBRARY)

.PHONY: all clean