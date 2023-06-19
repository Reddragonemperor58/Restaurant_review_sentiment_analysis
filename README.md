# Restaurant_review_sentiment_analysis
This project aims to build a model for sentiment analysis using the Restaurant Review dataset. The dataset, named "Restaurant_Reviews.tsv," is obtained from Kaggle and contains 1000 reviews on a restaurant.

## Objective <br />
The goal of this project is to develop a predictive model that can determine whether a restaurant review is positive or negative. To achieve this, we will employ "Random Forest Classifier" from Scikit-learn

## Steps Involved  <br />
#### Importing Dataset: 
We start by importing the Restaurant_Reviews.tsv dataset into our project. This dataset will serve as the basis for training and evaluating our predictive models.<br />

#### Preprocessing Dataset: 
Before training the models, we need to preprocess the dataset. This step involves removing any unnecessary information, such as punctuation and stopwords, and performing tasks like tokenization and stemming to normalize the text data.<br />

#### Vectorization: 
In order to apply machine learning algorithms to text data, we need to convert the textual reviews into numerical feature vectors. This process, known as vectorization, transforms the text data into a format that can be understood by the predictive algorithms.<br />

#### Training and Classification: 
With the preprocessed and vectorized dataset, we can proceed to train our predictive model. We will utilize the RandomForestClassifier for this task. The model will learn from the labeled reviews in the dataset to classify future reviews as positive or negative.<br />

#### Analysis and Conclusion: 
After training the model, we will evaluate our model's performance based on accuracy. This analysis will allow us to optimize the model by changing various hyperparameters to get the best predictive performance for sentiment analysis on the restaurant reviews.


### Saved Model:
The file named "rr_model.sav" is the RandomForestClassifier model that has been saved using joblit after training it😊. So this model can be used directly for predictions as it has already been trained
