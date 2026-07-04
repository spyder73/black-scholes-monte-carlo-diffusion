#include "MonteCarlo.cpp"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

struct Run {
    std::string label;
    std::string option_type;
    double S0, K, r, sigma, T;
    int stepSize, N;
};

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) { out.push_back(item);};
    return out;
}

Run loadRun(const std::string& path) {
    std::ifstream in(path);
    std::string header, line;
    std::getline(in, header);
    std::getline(in, line);
    auto c = split(line, ',');
    return {c[0], c[1], std::stod(c[2]), std::stod(c[3]), std::stod(c[4]),
            std::stod(c[5]), std::stod(c[6]), std::stoi(c[7]), std::stoi(c[8])};
}

void writeCSV(const std::string& label, const std::vector<double>& disc) {
    std::ofstream out("results/discounted_" + label + ".csv");
    out << "i,discounted\n";
    for (size_t i = 0; i < disc.size(); ++i)
        out << i + 1 << "," << disc[i] << "\n";
}
void writeResult(const std::string& label, double price, double err) {
    std::ofstream out("results/result_" + label + ".csv");
    out << "price,stderr\n";
    out << price << "," << err << "\n";
}

int main() {
    Run j = loadRun("params.csv");

    BlackScholes mc(j.N, j.K, j.r, j.sigma, j.T/365.0, j.S0, j.stepSize, j.option_type);
    mc.monteCarloWrapper();
    mc.discount();
    auto [price, err] = mc.optionPrice();

    std::cout << j.label << " price=" << price << " stderr=" << err << std::endl;

    writeCSV(j.label, mc.getDiscounted());
    writeResult(j.label, price, err);
    return 0;
}