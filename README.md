Optimising Sentiment Analysis with Machine Learning and Natural Language Processing A Comparative Study of Algorithms and Techniques



1. **Data Loading and Preprocessing**:
   - Loads a dataset containing reviews and corresponding ratings from a CSV file.
   - Handles missing values by filling them with appropriate values.
   - Combines the review text and summary columns into a single column called "reviews".
   - Performs text preprocessing steps such as converting text to lowercase, removing punctuation, numbers, and hyperlinks.

2. **Exploratory Data Analysis (EDA)**:
   - Analyzes descriptive statistics of the dataset.
   - Visualizes the distribution of ratings and sentiments (positive, neutral, negative) using pie charts and bar plots.
   - Calculates the polarity of reviews using TextBlob and plots a histogram to visualize the polarity distribution.
   - Plots histograms to visualize the distribution of review length and word counts.

3. **N-gram Analysis**:
   - Analyzes uni-gram, bi-gram, and tri-gram frequencies for reviews in each sentiment category (positive, neutral, negative).
   - Visualizes the most common n-grams using bar plots.

4. **Word Clouds**:
   - Generates word clouds to visualize the most common words in each sentiment category (positive, neutral, negative).

5. **Feature Engineering**:
   - Drops irrelevant columns like reviewer ID, product ID, reviewer name, etc., from the dataset.
   - Encodes the sentiment labels using LabelEncoder.

6. **TF-IDF Vectorization**:
   - Transforms the text data into numerical features using TF-IDF vectorization with a maximum of 5000 features and considering bi-grams.

7. **Handling Imbalanced Data**:
   - Uses the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to address class imbalance in the target variable (sentiment).

8. **Model Selection and Evaluation**:
   - Trains and evaluates several classification algorithms using cross-validation:
     - Decision Tree
     - Logistic Regression
     - Support Vector Classifier (SVC)
     - Random Forest
     - Naive Bayes
     - K-Neighbors
   - Evaluates the performance of each algorithm based on accuracy.

9. **Hyperparameter Tuning**:
   - Performs grid search to find the best hyperparameters for Logistic Regression, such as regularization parameter (C) and penalty.

10. **Model Training and Evaluation**:
    - Trains the Logistic Regression model with the best hyperparameters on the training data.
    - Evaluates the model's performance on the test set using accuracy score and a confusion matrix.

Additionally, the code includes sections for Sentiment Analysis Emotions for NLP and training a Random Forest Classifier on test data after cleaning and preprocessing.
