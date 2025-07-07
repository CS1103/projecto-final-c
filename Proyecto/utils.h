#pragma once
#include <fstream>
#include <sstream>
#include <vector>

void load_dataset(const std::string& filename,
                  std::vector<std::vector<float>>& X,
                  std::vector<std::vector<float>>& Y) {
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::vector<float> row;
        while (std::getline(ss, val, ',')) {
            row.push_back(std::stof(val));
        }
        if (row.size() != 6) continue;
        X.push_back({ row[0], row[1], row[2], row[3], row[4] });
        std::vector<float> y(3, 0);
        y[(int)row[5]] = 1;
        Y.push_back(y);
    }
}