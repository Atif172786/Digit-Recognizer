#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>
#include <ctime>  

using namespace std;

//load the training datasect
void loadTrainDataset(const string& filename, vector<vector<int>>& data, vector<int>& labels) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open the training dataset file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    getline(file, line);  
    int count = 0;
    while (getline(file, line)) { 
        stringstream ss(line);
        string value;
        vector<int> row;
        getline(ss, value, ',');  // First column is the label
        labels.push_back(stoi(value));
        while (getline(ss, value, ',')) {
            row.push_back(stoi(value));
        }
        data.push_back(row);
    }
}

// Load the test dataset
void loadTestDataset(const string& filename, vector<vector<int>>& data) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open the test dataset file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    getline(file, line);  
    int count = 0;
    while (getline(file, line)) {  
        stringstream ss(line);
        string value;
        vector<int> row;
        while (getline(ss, value, ',')) {
            row.push_back(stoi(value));
        }
        data.push_back(row);
    }
}

// Sigmoid function for logistic regression
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Logistic Regression Prediction (Binary classification)
int predict(const vector<double>& theta, const vector<int>& x) {
    double z = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        z += theta[i] * x[i];
    }
    return sigmoid(z) >= 0.5 ? 1 : 0;  // Binary classification
}

// Train logistic regression using gradient descent
vector<double> trainLogisticRegression(const vector<vector<int>>& X, const vector<int>& y, double alpha, int iterations) {
    size_t m = X.size();
    size_t n = X[0].size();
    vector<double> theta(n, 0);  // Initialize theta with zeros

    for (int i = 0; i < iterations; ++i) {
        vector<double> gradients(n, 0.0);
        for (size_t j = 0; j < m; ++j) {
            int prediction = predict(theta, X[j]);
            int error = y[j] - prediction;

            for (size_t k = 0; k < n; ++k) {
                gradients[k] += error * X[j][k];
            }
        }

        // Update the theta values
        for (size_t k = 0; k < n; ++k) {
            theta[k] += (alpha / m) * gradients[k];
        }
    }

    return theta;
}

// Save the result to result.csv
void saveResult(const vector<int>& predictions, const string& filename) {
    ofstream file(filename);
    file << "ImageId,Label\n";
    for (size_t i = 0; i < predictions.size(); ++i) {
        file << i + 1 << "," << predictions[i] << "\n";  // ImageId starts from 1
    }
}

// Main function
int main() {
    vector<vector<int>> trainData, testData;
    vector<int> trainLabels;

    // Load datasets
    loadTrainDataset("train.csv", trainData, trainLabels);
    loadTestDataset("test.csv", testData);

    // Prepare the dataset for Logistic Regression (Add bias term, 1)
    for (auto& row : trainData) {
        row.insert(row.begin(), 1);  
    }

    for (auto& row : testData) {
        row.insert(row.begin(), 1);  
    }

    // Parameters for Logistic Regression
    double alpha = 0.01;  
    int iterations = 1000; 

    
    time_t startTime, endTime;
    time(&startTime);  

    vector<double> theta = trainLogisticRegression(trainData, trainLabels, alpha, iterations);

    vector<int> predictions;

    // Loop through the test data
    for (size_t i = 0; i < testData.size(); ++i) {
        int label = predict(theta, testData[i]);
        predictions.push_back(label);
    }

    // End the clock to calculate total time
    time(&endTime);  
    double totalElapsedTime = difftime(endTime, startTime);

    saveResult(predictions, "result.csv");

    cout << "Total runtime for classification: " << totalElapsedTime << " seconds (" << totalElapsedTime / 60 << " minutes)." << endl;
    cout << "Results saved to result.csv." << endl;

    return 0;
}
