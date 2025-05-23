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
int thread_cnt = 4;
int main() {
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
    int size = max_node + 1;
    vector<vector<float>>ans;
    for (int start = 0; start < size; start++){
        vector<float>minDist(size, MAXF);
        minDist[start] = 0;
        #pragma omp parallel num_threads(thread_cnt)
        for (int i = 0; i < size - 1; i++){
            int inner_n = edges.size();
            #pragma omp parallel for
            for (int j = 0; j < inner_n; j++){
                int from = get<0>(edges[j]);
                int to = get<1>(edges[j]);
                float dis = get<2>(edges[j]);
                if(minDist[from] != MAXF && minDist[to] > minDist[from] + dis){
                    minDist[to] = minDist[from] + dis;
                }
                else if(minDist[to] != MAXF && minDist[from] > minDist[to] + dis){
                    minDist[from] = minDist[to] + dis;
                }
            }
        }
        ans.push_back(minDist);
        for (int i = 0; i < size; i++){
            if(minDist[i] != MAXF)
                std::cout<< setw(5) << minDist[i];
            else
                std::cout<< setw(5) <<"INF";
        }
        std::cout<<endl;
    }
    
    


}