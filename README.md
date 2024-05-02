Optimising Sentiment Analysis with Machine Learning and Natural Language Processing A Comparative Study of Algorithms and Techniques


<h1 align="center">
  <br>
  <a href="http://www.amitmerchant.com/electron-markdownify"><img src="https://raw.githubusercontent.com/amitmerchant1990/electron-markdownify/master/app/img/markdownify.png" alt="Markdownify" width="200"></a>
  <br>
  Naimul Hasan Shadesh
  <br>
</h1>

<h4 align="center">Data Science AI & Machine Learning Deep Learning Natural Language Processing With <a [href="http://electron.atom.io](https://www.kaggle.com/shadesh)" target="_blank">Python</a>.</h4>

<p align="center">
  <a href="https://badge.fury.io/js/electron-markdownify">
    <img src="https://badge.fury.io/js/electron-markdownify.svg"
         alt="Gitter">
  </a>
  <a href="https://gitter.im/amitmerchant1990/electron-markdownify"><img src="https://badges.gitter.im/amitmerchant1990/electron-markdownify.svg"></a>
  <a href="https://saythanks.io/to/bullredeyes@gmail.com">
      <img src="https://img.shields.io/badge/SayThanks.io-%E2%98%BC-1EAEDB.svg">
  </a>
  <a href="https://www.paypal.me/AmitMerchant">
    <img src="https://img.shields.io/badge/$-donate-ff69b4.svg?maxAge=2592000&amp;style=flat">
  </a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
  <a href="#license">License</a>
</p>

![screenshot](https://raw.githubusercontent.com/amitmerchant1990/electron-markdownify/master/app/img/markdownify.gif)

## Key Features

1. Enhancing model interpretability through feature importance analysis.
2. Implementing deep learning models for improved accuracy.
3. Integrating automated model deployment pipelines for seamless production deployment.
4. Incorporating advanced natural language processing techniques like BERT embeddings.
5. Focusing on context and semantics understanding for better performance.
 

## Table of Content

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

## How To Use

To clone and run this application, you'll need [Git](https://git-scm.com) and [Node.js](https://nodejs.org/en/download/) (which comes with [npm](http://npmjs.com)) installed on your computer. From your command line:

```bash
# Clone this repository
$ git clone https://github.com/amitmerchant1990/electron-markdownify

# Go into the repository
$ cd electron-markdownify

# Install dependencies
$ npm install

# Run the app
$ npm start
```

> **Note**
> If you're using Linux Bash for Windows, [see this guide](https://www.howtogeek.com/261575/how-to-run-graphical-linux-desktop-applications-from-windows-10s-bash-shell/) or use `node` from the command prompt.


## Download

You can [download](https://github.com/amitmerchant1990/electron-markdownify/releases/tag/v1.2.0) the latest installable version of Markdownify for Windows, macOS and Linux.

## Emailware

Markdownify is an [emailware](https://en.wiktionary.org/wiki/emailware). Meaning, if you liked using this app or it has helped you in any way, I'd like you send me an email at <bullredeyes@gmail.com> about anything you'd want to say about this software. I'd really appreciate it!

## Credits

This software uses the following open source packages:

- [Electron](http://electron.atom.io/)
- [Node.js](https://nodejs.org/)
- [Marked - a markdown parser](https://github.com/chjj/marked)
- [showdown](http://showdownjs.github.io/showdown/)
- [CodeMirror](http://codemirror.net/)
- Emojis are taken from [here](https://github.com/arvida/emoji-cheat-sheet.com)
- [highlight.js](https://highlightjs.org/)

## Related

[markdownify-web](https://github.com/amitmerchant1990/markdownify-web) - Web version of Markdownify

## Support

<a href="https://www.buymeacoffee.com/5Zn8Xh3l9" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/purple_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

<p>Or</p> 

<a href="https://www.patreon.com/amitmerchant">
	<img src="https://c5.patreon.com/external/logo/become_a_patron_button@2x.png" width="160">
</a>

## You may also like...

- [Pomolectron](https://github.com/shadesh) - A pomodoro app
- [Correo](https://www.kaggle.com/shadesh) - A menubar/taskbar Gmail App for Windows and macOS

## License

MIT

---

> [Naimul Hasan Shadesh]((https://mdshadesh.github.io/Portfolio-NHS-/)) &nbsp;&middot;&nbsp;
> GitHub [@shadesh](https://mdshadesh) &nbsp;&middot;&nbsp;
> Kaggle [@shadesh]((https://www.kaggle.com/shadesh))



