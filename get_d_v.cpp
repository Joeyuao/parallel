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
    // string filename = "data/updated_flower.csv";
    // string filename = "data/updated_mouse.csv";
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
    int cnt = 0;
    while (getline(file, line)) {
        // 将逗号替换为空格，以便统一处理分隔符
        cnt++;
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
    cout<<"edges_num："<<cnt<<endl;
    cout<<"nodes_num："<<max_node + 1<<endl;
    cout<<"avg:"<<(double)cnt / (double)(max_node + 1)<<endl;
}