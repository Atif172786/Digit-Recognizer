#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <ctime>

using namespace std;

// Function to load the training dataset 
void loadTrainDataset(const string& filename, vector<vector<int>>& data, vector<int>& labels) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open the training dataset file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    getline(file, line); // Skip the header
    while (getline(file, line)) { 
        stringstream ss(line);
        string value;
        vector<int> row;
        getline(ss, value, ','); // First column is the label
        labels.push_back(stoi(value));
        while (getline(ss, value, ',')) {
            row.push_back(stoi(value));
        }
        data.push_back(row);
    }
}

// Function to load the test dataset 
void loadTestDataset(const string& filename, vector<vector<int>>& data) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open the test dataset file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    getline(file, line); // Skip the header
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

// Function to compute the Euclidean distance between two data points
double euclideanDistance(const vector<int>& a, const vector<int>& b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += pow(a[i] - b[i], 2);
    }
    return sqrt(sum);
}

// KNN algorithm to predict the label for a test image
int knn(const vector<vector<int>>& trainData, const vector<int>& trainLabels, const vector<int>& testData, int k) {
    vector<pair<double, int>> distances;

    // Calculate distance to all training data
    for (size_t i = 0; i < trainData.size(); ++i) {
        double dist = euclideanDistance(trainData[i], testData);
        distances.push_back(make_pair(dist, trainLabels[i]));
    }

    // Sort the distances
    sort(distances.begin(), distances.end());

    // Get the labels of the k nearest neighbors
    vector<int> nearestLabels;
    for (int i = 0; i < k; ++i) {
        nearestLabels.push_back(distances[i].second);
    }

    // Find the most common label among the nearest neighbors
    vector<int> count(10, 0); // There are 10 digits (0-9)
    for (int label : nearestLabels) {
        count[label]++;
    }

    int maxCount = 0;
    int predictedLabel = -1;
    for (int i = 0; i < 10; ++i) {
        if (count[i] > maxCount) {
            maxCount = count[i];
            predictedLabel = i;
        }
    }

    return predictedLabel;
}

// Function to save the result to result.csv
void saveResult(const vector<int>& predictions, const string& filename) {
    ofstream file(filename);
    file << "ImageId,Label\n";
    for (size_t i = 0; i < predictions.size(); ++i) {
        file << i + 1 << "," << predictions[i] << "\n"; // ImageId starts from 1
    }
}

int main() {
    vector<vector<int>> trainData, testData;
    vector<int> trainLabels;

    // Load the first 50 samples from train.csv and test.csv
    loadTrainDataset("train.csv", trainData, trainLabels);
    loadTestDataset("test.csv", testData);

    int k = 3; // Number of nearest neighbors
    vector<int> predictions;

    // Start time
    time_t startTime;
    time(&startTime);

    time_t lastReportTime = startTime;

    // Classify each test image
    for (size_t i = 0; i < testData.size(); ++i) {
        int label = knn(trainData, trainLabels, testData[i], k);
        predictions.push_back(label);

        // Check elapsed time every 10 seconds
        time_t currentTime;
        time(&currentTime);
        if (difftime(currentTime, lastReportTime) >= 10 ) {
            double elapsed = difftime(currentTime, startTime);
            cout << "Elapsed time: " << elapsed << " seconds (" << elapsed / 60 << " minutes)." << endl;
            lastReportTime = currentTime; // Update last report time
        }
    }

    // End time
    time_t endTime;
    time(&endTime);

    // Save the predictions to result.csv
    saveResult(predictions, "result.csv");

    // Print total runtime
    double totalElapsed = difftime(endTime, startTime);
    cout << "Total runtime for classification: " << totalElapsed << " seconds (" << totalElapsed / 60 << " minutes)." << endl;

    cout << "Results saved to result.csv." << endl;

    return 0;
}
