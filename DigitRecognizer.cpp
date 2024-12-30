#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>
#include <ctime>
#include <algorithm> 

using namespace std;

// load the training dataset
void loadTrainDataset(const string& filename, vector<vector<int>>& data, vector<int>& labels) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open the training dataset file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    getline(file, line);  // Skip the header line
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<int> row;

        getline(ss, value, ',');
        labels.push_back(stoi(value));

        while (getline(ss, value, ',')) {
            row.push_back(stoi(value));
        }
        data.push_back(row);
    }
    file.close();
}

// load the test dataset
void loadTestDataset(const string& filename, vector<vector<int>>& data) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open the test dataset file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    getline(file, line);  
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<int> row;

        // Extract features (all columns)
        while (getline(ss, value, ',')) {
            row.push_back(stoi(value));
        }
        data.push_back(row);
    }
    file.close();
}

// Function to normalize the dataset
void normalizeData(vector<vector<int>>& data) {
    for (auto& row : data) {
        for (auto& val : row) {
            val = val / 255.0;  // Normalize to range [0, 1]
        }
    }
}

// Sigmoid function for logistic regression
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Predict function for one-vs-rest
int predict(const vector<double>& theta, const vector<int>& x) {
    double z = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        z += theta[i] * x[i];
    }
    return sigmoid(z) >= 0.5 ? 1 : 0;
}

// Train logistic regression for one-vs-rest classification
vector<double> trainLogisticRegression(const vector<vector<int>>& X, const vector<int>& y, double alpha, int iterations) {
    size_t m = X.size();
    size_t n = X[0].size();
    vector<double> theta(n, 0);  // Initialize theta with zeros

    for (int iter = 0; iter < iterations; ++iter) {
        vector<double> gradients(n, 0.0);

        for (size_t i = 0; i < m; ++i) {
            int prediction = predict(theta, X[i]);
            int error = y[i] - prediction;

            for (size_t j = 0; j < n; ++j) {
                gradients[j] += error * X[i][j];
            }
        }

        for (size_t j = 0; j < n; ++j) {
            theta[j] += (alpha / m) * gradients[j];
        }
    }

    return theta;
}

// One-vs-rest training for multi-class classification
vector<vector<double>> trainOneVsRest(const vector<vector<int>>& X, const vector<int>& y, double alpha, int iterations) {
    vector<vector<double>> allThetas(10);  // One theta vector for each digit (0 to 9)

    for (int digit = 0; digit < 10; ++digit) {
        vector<int> binaryLabels(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            binaryLabels[i] = (y[i] == digit) ? 1 : 0;  // Convert labels to binary (1 for this digit, 0 otherwise)
        }
        allThetas[digit] = trainLogisticRegression(X, binaryLabels, alpha, iterations);
    }

    return allThetas;
}

// Predict for multi-class classification
int predictMultiClass(const vector<vector<double>>& allThetas, const vector<int>& x) {
    vector<double> probabilities(10);

    for (int digit = 0; digit < 10; ++digit) {
        double z = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            z += allThetas[digit][i] * x[i];
        }
        probabilities[digit] = sigmoid(z);
    }

    return max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();  // Index of max probability
}

// Save the results to a CSV file
void saveResult(const vector<int>& predictions, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not write to file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    file << "ImageId,Label\n";  
    for (size_t i = 0; i < predictions.size(); ++i) {
        file << i + 1 << "," << predictions[i] << "\n";  
    }
    file.close();
}

// Main function
int main() {
    vector<vector<int>> trainData, testData;
    vector<int> trainLabels;

    // Load datasets
    loadTrainDataset("train.csv", trainData, trainLabels);
    loadTestDataset("test.csv", testData);

    // Normalize datasets
    normalizeData(trainData);
    normalizeData(testData);

    // Add bias term
    for (auto& row : trainData) row.insert(row.begin(), 1);
    for (auto& row : testData) row.insert(row.begin(), 1);

    double alpha = 0.1;  // Smaller learning rate(accuracy drops when changing for some reason?)
    int iterations = 500;  // Fewer iterations for simplicity

    time_t startTime, endTime;
    time(&startTime);

    vector<vector<double>> allThetas = trainOneVsRest(trainData, trainLabels, alpha, iterations);

    vector<int> predictions;
    for (const auto& row : testData) {
        predictions.push_back(predictMultiClass(allThetas, row));
    }

    time(&endTime);
    double totalElapsedTime = difftime(endTime, startTime);

    saveResult(predictions, "result.csv");
    cout << "Total runtime for classification: " << totalElapsedTime << " seconds (" << totalElapsedTime / 60 << " minutes)." << endl;
    cout << "Results saved to result.csv." << endl;

    return 0;
}
