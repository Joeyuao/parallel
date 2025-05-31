import matplotlib.pyplot as plt
import sys
from collections import defaultdict
Lpoints = []
Lthreads = []
Lefficiency = []
Lerror = []
avg_error = []
serial_times = []
para_times = []
speedups = []
def read_data(file_path):
    """读取数据文件"""
    cnt=0
    t=0.0
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            [points, threads, efficiency, error,
             serial_time, para_time, speedup] = line.strip().split(',')
            if(points == 'points'):
                continue
            Lpoints.append(int(points))
            Lthreads.append(int(threads))
            Lefficiency.append(float(efficiency))
            Lerror.append(float(error))
            serial_times.append(float(serial_time))
            para_times.append(float(para_time))
            speedups.append(float(speedup))
    return

if __name__=="__main__":
    read_data("lab4\pi_results1.txt")
    X_points = Lpoints[::16]
    # print(X_points)
    # print(Lerror)
    plt.plot(X_points,Lerror[0::16],label="1 thread")
    plt.plot(X_points,Lerror[1::16],label="2 thread")
    plt.plot(X_points,Lerror[3::16],label="4 thread")
    plt.plot(X_points,Lerror[7::16],label="8 thread")
    plt.plot(X_points,Lerror[15::16],label="16 thread")

    plt.xlabel('Points')
    plt.ylabel('error')
    plt.title('Error & Point with dif threads_num')
    plt.legend()

    plt.show()

    plt.plot(X_points,Lefficiency[0::16],label="1 thread")
    plt.plot(X_points,Lefficiency[1::16],label="2 thread")
    plt.plot(X_points,Lefficiency[3::16],label="4 thread")
    plt.plot(X_points,Lefficiency[7::16],label="8 thread")
    plt.plot(X_points,Lefficiency[15::16],label="16 thread")

    plt.xlabel('Points')
    plt.ylabel('efficiency')
    plt.title('Efficiency & Point with dif threads_num')
    plt.legend()

    plt.show()

    plt.plot(X_points,speedups[0::16],label="1 thread")
    plt.plot(X_points,speedups[1::16],label="2 thread")
    plt.plot(X_points,speedups[3::16],label="4 thread")
    # plt.plot(X_points,speedups[7::16],label="8 thread")
    # plt.plot(X_points,speedups[15::16],label="16 thread")

    plt.xlabel('Points')
    plt.ylabel('speedup')
    plt.title('Speedup & Point with dif threads_num')
    plt.legend()

    plt.show()

    # plt.plot(X_points,speedups[0::16],label="1 thread")
    # plt.plot(X_points,speedups[1::16],label="2 thread")
    plt.plot(X_points,speedups[3::16],label="4 thread")
    plt.plot(X_points,speedups[7::16],label="8 thread")
    plt.plot(X_points,speedups[15::16],label="16 thread")

    plt.xlabel('Points')
    plt.ylabel('speedup')
    plt.title('Speedup & Point with dif threads_num')
    plt.legend()

    plt.show()