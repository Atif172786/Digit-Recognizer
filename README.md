# Digit-Recognizer

This code uses the K-Nearest Neighbors (KNN) algorithm to classify digits. It reads a training dataset with labeled images and a test dataset with unlabeled images. For each test image, it calculates the Euclidean distance to every training image to measure similarity. After sorting these distances, it picks the `k`(here I took k as 3 ) nearest neighbors and assigns the most frequent label among them as the prediction.

The results are written to `result.csv`, and the code prints updates every 10 seconds to show how much time has passed. It’s a brute-force method which could have been optimized with multithreading or some other algorithm other than knn but due to time contraints i couldn't complete it.
There’s no optimization here—it just runs through all the data, compares everything, and gets the job done. It works but is pretty slow for big datasets.(took 11 hours to run :( )
