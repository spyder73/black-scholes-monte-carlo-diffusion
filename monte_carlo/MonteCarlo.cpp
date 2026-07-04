#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <numeric>
#include <algorithm>

std::mt19937_64 generator;

//CALCULATE MEAN VALUE OF A GIVEN ARRAY WITH N VALUES
template<typename T>
std::pair<double,double> average_std(const std::vector<T>& k) {
    double avg = 0.0;
    double sigma = 0.0;
    double n = static_cast<double>(k.size());

    for (auto i = k.begin(); i != k.end(); i++) {
        avg += *i;
    }

    avg /= n; 

    for (auto j = k.begin(); j != k.end(); j++) {
        sigma += pow((*j - avg),2);
    }

    sigma = sqrt(sigma/(n-1));

    // We want the standard error on the mean so:
    sigma /= sqrt(n);

    return std::pair<double,double>(avg, sigma);
}

double normalCDF(double& x, double mean=0.0, double stddev=1.0) {
    return 0.5 * erfc(-(x - mean) / (stddev * sqrt(2)));
}

class BlackScholes{

    private:
        //Parameters
        const int N;
        const int stepSize;
        const std::string type;

        const double X;
        const double r;
        const double sig;
        const double T;
        const double S0;

        std::vector<double> S;
        std::vector<double> payoffs;
        std::vector<double> discounted;

        const double dt;
        double* lastPrice;
    

    public:

        BlackScholes(const int N, const double X, const double r, const double sig, const double T, const double S0, const int stepSize, const std::string type):
            N(N), X(X), r(r), sig(sig), T(T), S0(S0), stepSize(stepSize), type(type), dt(T/stepSize) {
                S.push_back(S0);
                lastPrice = &S[0];
                std::cout << "T:" << T << std::endl;
            };
        
        ~BlackScholes() {std::cout << "Simulation killed." << std::endl;};

        double randomNumber(double a=0.0, double b=1.0) {
            std::normal_distribution<double> distribution(a,b);
            double random_number = distribution(generator);
            return random_number;
        };

        void doStep() {
            double x = randomNumber();
            double step = dt;
            // In that case doStep is only executed once
            if (type == "european_call" || type == "european_put" || type == "binary_call" || type == "binary_put") {
                step = T;
            }
            double price = *lastPrice * exp( (r-0.5*pow(sig,2))*step + sig*sqrt(step)* x);
            S.push_back(price);
            lastPrice = &S.back();
        };

        void monteCarloWrapper() {
            payoffs.clear();
            discounted.clear();
            double currentF;
            for (auto i=0; i < N; i++) {
                // we have to reset the lastPrice to S0 before starting a new trajectory
                // we need to clear the previous paths for each new S
                S.clear();
                S.push_back(S0);
                lastPrice = &S[0];
                if (type != "european_call" && type != "european_put" && type != "binary_call" && type != "binary_put") {
                    for (auto j=0; j < stepSize; j++) {
                        doStep();
                    };
                }
                else {
                    // Do one step to T right away
                    doStep();
                }
                currentF = payoff();
                payoffs.push_back(currentF);
            };
        };

        double payoff() {
            if (type == "european_call") {
                return std::max(*lastPrice - X, 0.0);
            }
            else if (type == "european_put") {
                return std::max(X - *lastPrice, 0.0);
            }
            else if (type == "asian_call") {
                double sum = std::accumulate(S.begin(), S.end(), 0.0);
                double avg = sum / static_cast<double>(S.size());
                return std::max(avg - X, 0.0);
            }
            else if (type == "asian_put") {
                double sum = std::accumulate(S.begin(), S.end(), 0.0);
                double avg = sum / static_cast<double>(S.size());
                return std::max(X - avg, 0.0);
            }
            else if (type == "binary_call") {
                return (*lastPrice > X) ? 1.0 : 0.0;
            }
            else if (type == "binary_put") {
                return (*lastPrice < X) ? 1.0 : 0.0;
            }
            else if (type == "lookback_call") {
                double minS = *std::min_element(S.begin(), S.end());
                return std::max(*lastPrice - minS, 0.0);
            }
            else if (type == "lookback_put") {
                double maxS = *std::max_element(S.begin(), S.end());
                return std::max(maxS - *lastPrice, 0.0);
            }
            else {
                return 0.0;
            }
        }

        double analytical() {
            if (type == "european_call") {
                double d1 = (log(S0/X) + (r + 0.5*pow(sig,2))) * T;
                d1 /= sig * sqrt(T);

                double d2 = d1 - sig*sqrt(T);
                
                return S0 * normalCDF(d1) - X * exp(-r*T) * normalCDF(d2);
            }
            else {
                return 0;
            }
        }

        void discount() {
            for (auto i=payoffs.begin(); i!=payoffs.end(); i++) {
                discounted.push_back(exp(-r*T) * (*i));
            }
        }

        std::pair<double,double> optionPrice() {
            std::pair<double, double> price = average_std(discounted);
            return price;
        }

        const std::vector<double>& getDiscounted() const {
            return discounted;
        }
        

};