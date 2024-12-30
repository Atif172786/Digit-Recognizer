Loading the Data: To load the training and test datasets from files named train.csv and test.csv. The training data has both the features (which are the pixel values of digits) and the labels (the actual digit). For test data, there are only features. To improve the math in training, I added a "bias term," which is just a constant value of 1, to every row of the data.

Understanding Logistic Regression: Logistic regression predicts probabilities for two categories (like 0 or 1). I used sigmoid function to calculate probabilities from weighted sums of the input features. Then, I decided anything above 0.5 should be predicted as 1, and anything below as 0.

Training the Model: To make the model learn, I used  gradient descent. It’s where the program adjusts the weights (called theta in the code) repeatedly to minimize errors. I looped through this process for a set number of iterations(500 yielded the best results).

Making Predictions: After training the model, I used the learned weights to predict the labels for the test data. For each test sample, the program calculates the probability using the sigmoid function and classifies it as 0 or 1.

That’s how I think it works. I’m sure there are mistakes, though. For example:

I didn’t normalize the input data. I read somewhere that it’s important, but I skipped it because I wasn’t sure how to do it.(just did some trial and error but it reduced the accuracy even more).
I forgot to divide the gradient properly, but it doesn’t seem to break anything… yet!
if you look at the history **i did try with knn which improved the accuracy immensely but was very slow**(took almost 11 hours!)
