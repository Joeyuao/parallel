#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <tuple>
#include <limits>
#include <iomanip>
#include <omp.h>
using namespace std;
int thread_cnt = 4;
const float INF = numeric_limits<float>::max();

void printMatrix(const vector<vector<float>>& matrix) {
    int n = matrix.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (matrix[i][j] == INF) {
                cout << setw(5) << "INF";
            } else {
                cout << setw(5) << matrix[i][j];
            }
        }
        cout << endl;
    }
}
int main(int argc,char** argv) {
    string filename = "data/my_example2.csv";
    ifstream file(filename);
    if(argc != 2){
        cout<<"./xxx <num_thread>"<<endl;
    }
    thread_cnt = atoi(argv[1]);
    // cout<<argc<<' '<<thread_cnt;
    if (!file.is_open()) {
        cerr << "无法打开文件" << endl;
        return 1;
    }

    string line;
    getline(file, line); // 跳过标题行（如果有）

    int max_node = 0;
    vector<tuple<int, int, float>> edges;

    // 读取所有边并确定最大节点编号
    while (getline(file, line)) {
        // 将逗号替换为空格，以便统一处理分隔符
        replace(line.begin(), line.end(), ',', ' ');
        istringstream iss(line);
        int source, target;
        float distance;
        if (iss >> source >> target >> distance) {
            edges.emplace_back(source, target, distance);
            max_node = max(max_node, source);
            max_node = max(max_node, target);
        }
    }

    int size = max_node + 1; // 邻接矩阵的大小为最大节点编号+1
    vector<vector<float>> adj_matrix(size, vector<float>(size, INF));

    // 初始化对角线为0，其余为INF
    for (int i = 0; i < size; ++i) {
        adj_matrix[i][i] = 0.0;
    }

    // 填充邻接矩阵
    for (const auto& edge : edges) {
        int s = get<0>(edge);
        int t = get<1>(edge);
        float d = get<2>(edge);
        adj_matrix[s][t] = d;
        adj_matrix[t][s] = d; // 无向图需要双向设置
    }

    // cout << "初始邻接矩阵：" << endl;
    // printMatrix(adj_matrix);

    int n = adj_matrix.size();
    
    // Floyd-Warshall算法核心
    double st = omp_get_wtime();
    #pragma omp parallel num_threads(thread_cnt)
    for (int k = 0; k < n; ++k) {
        #pragma omp for
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                // 避免溢出，只有当两边都有有效值时才更新
                if (adj_matrix[i][k] != INF && adj_matrix[k][j] != INF) {
                    if (adj_matrix[i][j] > adj_matrix[i][k] + adj_matrix[k][j]) {
                        adj_matrix[i][j] = adj_matrix[i][k] + adj_matrix[k][j];
                    }
                }
            }
        }
    }
    double time = omp_get_wtime() - st;
    cout<<"用时："<<time<<endl;
    // cout << "\n所有节点对的最短路径矩阵：" << endl;
    // printMatrix(adj_matrix);

    return 0;
}