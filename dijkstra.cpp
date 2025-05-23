#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <tuple>
#include <limits>
#include <iomanip>
using namespace std;
const float MINF = numeric_limits<float>::min();
const float MAXF = numeric_limits<float>::max();
int main(int argc, char** argv) {
    string filename = "data/my_example2.csv";
    ifstream file(filename);
    
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
    vector<vector<float>> adj_matrix(size, vector<float>(size, MAXF));

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
        adj_matrix[t][s] = d;
    }

    vector<vector<float>>ans;
    for (int start = 0; start < size; start++){
        std::vector<float> minDist(size , MAXF);

        // 记录顶点是否被访问过
        std::vector<bool> visited(size , false);
        // int start = 0; 
        minDist[start] = 0;  // 起始点到自身的距离为0

        for (int i = 0; i < size; i++) { // 遍历所有节点

            float minVal = MAXF;
            int cur = 1;

            // 1、选距离源点最近且未访问过的节点
            for (int v = 0; v < size; ++v) {
                if (!visited[v] && minDist[v] < minVal) {
                    minVal = minDist[v];
                    cur = v;
                }
            }

            visited[cur] = true;  // 2、标记该节点已被访问

            // 3、第三步，更新非访问节点到源点的距离（即更新minDist数组）
            for (int v = 0; v < size; v++) {
                if (!visited[v] && adj_matrix[cur][v] != MAXF && minDist[cur] + adj_matrix[cur][v] < minDist[v]) {
                    minDist[v] = minDist[cur] + adj_matrix[cur][v];
                }
            }

        }
        int end = size - 1;
        ans.push_back(minDist);
        for (int i = 0; i < size; i++){
            if(minDist[i] != MAXF)
                cout<< setw(5) << minDist[i];
            else
                cout<< setw(5) <<"INF";
        }
        cout<<endl;
    }

    
}