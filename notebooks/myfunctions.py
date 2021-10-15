##COMMENT: JUST HAVE THE MOST IMPORTNAT 

#here
# to IO/manipulate/calculate dataframes
import pandas as pd
import numpy as np

# to do math/statisctics ___________________________________________
import statistics as stat
import math

# to vitualize data ___________________________________________
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme(style="darkgrid")


import datetime, time

# # Regression modeling with sklrean ___________________________________________
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge, RidgeCV
# from sklearn.linear_model import Lasso, LassoCV
# from sklearn.linear_model import LogisticRegression


# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import StandardScaler

# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn import metrics

# # Natural Language Processing _______________________________________________
# from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
# from nltk.stem import WordNetLemmatizer
# from nltk.stem.porter import PorterStemmer
# from nltk.corpus import stopwords
# from nltk.sentiment.vader import SentimentIntensityAnalyzer

# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB

# from textblob import TextBlob

# # NLP: Sentiment Analsyi-----
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# #
# import spacy


# from xgboost import XGBRegressor
# from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc

# from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
# from sklearn.ensemble import AdaBoostClassifier  



# # Imbalance data Handling ___________________________________________________

# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import SMOTE, RandomOverSampler
# from collections import Counter
# from imblearn.pipeline import Pipeline

# # Emoji ________
# import demoji


import string
import re

# # lemmatizer = WordNetLemmatizer()
# # sent = SentimentIntensityAnalyzer()
# # To handel Emoji in NLP ______________________________________________________
# # 


# from sklearn.feature_extraction.text import CountVectorizer

# Instantiate Sentiment Intensity Analyzer


# this setting widens how many characters pandas will display in a column:
pd.options.display.max_colwidth = 400

# imports
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# # Import CountVectorizer and TFIDFVectorizer from feature_extraction.text.
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.ensemble import GradientBoostingClassifier  


# API  ______________________________________________________________________________________
import requests

# Time/date manupulation  ______________________________________________________________________________________
import time
import datetime as dt


# other packages  ______________________________________________________________________________________
import warnings
warnings.filterwarnings(action='ignore')





# Printing Fuctions ------------------------------------------------

# Visualization Fuctions --------------------------------------------

# Modeling Fuctions ------------------------------------------------

# NLP  ------------------------------------------------


# API Fuctions ------------------------------------------------

def api_pushshift_reddit(subreddit, 
                        kind = 'submission',
                        start_date = dt.datetime(2021, 1, 1),
                        end_date   = dt.datetime(2021, 9, 11),
                        day_window = 0.4, 
                        author = None):
    """
    This function read:
    Start date, End date: (year, month, day)
    day_window : any fraction of day, 0.1, 0.5, 1, 2, ... 
    """
    SUBFIELDS = ['title', 'selftext', 'subreddit', 'created_utc', 'author', 'num_comments', 'score', 'is_self']
    # calculate number of submissions 
    days = (end_date - start_date).days
    n_submissions = int(days / day_window); print(n_submissions)

    
    # establish base url and stem
    BASE_URL = f"https://api.pushshift.io/reddit/search/{kind}" # also known as the "API endpoint" 
    stem = f"{BASE_URL}?subreddit={subreddit}&size=100" # always pulling max of 100
    if author != None:
        stem = f"{BASE_URL}?subreddit={subreddit}&size=100&author={author}" # always pulling max of 100
            
    # instantiate empty list for temp storage
    posts = []
    
    start_dt = start_date
    
    # implement for loop with `time.sleep(2)`
    for i in range(1, n_submissions+1):
        end_dt = start_dt + dt.timedelta(day_window)
        URL = "{}&after={}&before={}".format(stem, start_dt, end_dt)
#         URL = "{}&after={}&before={}".format(stem, end_date, start_dt)
#         URL = "{}&after={}d".format(stem, day_window * i) # original comment
        print("Querying from: " + URL)
        response = requests.get(URL)
        assert response.status_code == 200
        mine = response.json()['data']
        df = pd.DataFrame.from_dict(mine)
        posts.append(df)
        time.sleep(2)
#         start_dt = end_date
        start_dt = start_dt + dt.timedelta(day_window)
    
    # pd.concat storage list
    full = pd.concat(posts, sort=False)
    
    # if submission
    if kind == "submission":
        # select desired columns
        full = full[SUBFIELDS]
        # drop duplicates
        full.drop_duplicates(inplace = True)
        # drop nulls
        full.dropna(inplace = True)
        # select `is_self` == True
        full = full.loc[full['is_self'] == True]

    # create `timestamp` column
    full['timestamp'] = full["created_utc"].map(dt.datetime.fromtimestamp)
    
    print("Query Complete!")    
    return full.reset_index(drop=True)


# Evaluation functions _____________________________________________________________________
def model_Evaluate(model, x_train, x_test, y_train, y_test):
    """
    """
    # Print accuracy scores on train and test sets
    print(f'Score on training set: {model.score(x_train, y_train)}')
    print(f'Score on testing set: {model.score(x_test, y_test)}')
    # Predict values for Test dataset
    y_pred = model.predict(x_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'pink',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    ### plot ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.title('ROC CURVE',fontsize=16)
    plt.legend(loc="lower right",fontsize=12)
    plt.show()



def save_model_Evaluate_values(model, x_train, x_test, y_train, y_test, model_name, balanced=True):
    """
    This function take the following inputs: 
    - model e.g. lr (if lr=LogisticRegression(), 
    - x_train, x_test, y_train, y_test : the same datasets you are using to feed your model
    - model_name: the model name you want to be saved
    - balanced=True: to print yes in front of the model
    """
    # Print accuracy scores on train and test sets
    R_train = model.score(x_train, y_train)
    R_test  = model.score(x_test, y_test)
    df_accuracy = pd.DataFrame( np.round( [R_train, R_test], 2 ),  columns=['score'])
    
    df_accuracy['metric'] = ['R_train' , 'R_test']
    df_accuracy['model'] = model_name
    if balanced == True:
        df_accuracy['balanced'] = 'yes'
    if balanced == False:
        df_accuracy['balanced'] = 'no'
        
    # Predict values for Test dataset
    y_pred = model.predict(x_test)
    scores = precision_recall_fscore_support(y_test, model.predict(x_test))
    df_precision_recall = pd.DataFrame( np.round(scores, 2) , columns=['is_pandemicPreps', 'is_covid19positive'] )
    df_precision_recall['metric'] = ['precision' , 'recall' , 'fscore' , 'support']
    df_precision_recall['model'] = model_name
    if balanced == True:
        df_precision_recall['balanced'] = 'yes'
    if balanced == False:
        df_precision_recall['balanced'] = 'no'

    
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = [np.round(value, 2) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    df_cf_matrix = pd.DataFrame( group_percentages ,  columns=['score'] )
    df_cf_matrix['metric'] = group_names
    df_cf_matrix['model'] = model_name
    if balanced == True:
        df_cf_matrix['balanced'] = 'yes'
    if balanced == False:
        df_cf_matrix['balanced'] = 'no'
        
    # save_______________________________ 
        """
        # NOTE: 
        - You should create the CSV file for the first time using the following pd.DatafFrame scripts
        - Change the path of the saved file here 
        """
    path = '../datasets/'
    pd.DataFrame(df_accuracy).to_csv(path+'models_metrics_report_accuracy.csv')
    pd.DataFrame(df_cf_matrix).to_csv(path+'models_metrics_report_confusionMatrix.csv')
    pd.DataFrame(df_precision_recall).to_csv(path+'models_metrics_report_precision_recall.csv')
    # accuracy 
    df1 = pd.read_csv(path+'models_metrics_report_accuracy.csv', index_col=0)
    df2 = df_accuracy
    pd.concat([df1,df2],ignore_index=True).to_csv(path+'models_metrics_report_accuracy.csv')
    # report_precision_recall
    df1 = pd.read_csv(path+'models_metrics_report_precision_recall.csv', index_col=0)
    df2 = df_precision_recall
    pd.concat([df1,df2],ignore_index=True).to_csv(path+'models_metrics_report_precision_recall.csv')
    # confusionMatrix
    df1 = pd.read_csv(path+'models_metrics_report_confusionMatrix.csv', index_col=0)
    df2 = df_cf_matrix
    pd.concat([df1,df2],ignore_index=True).to_csv(path+'models_metrics_report_confusionMatrix.csv')
    
    #     print(classification_report(y_test, y_pred))
    return df_accuracy, df_precision_recall, df_cf_matrix



# Statiscics Functions ----------------------------------------
def replace_null_with_mean(data, feature_list):
    """
    Calculate the mean of each column in a dataset and replace nulls with the column's mean
    """
    [ data[feature].fillna(data[feature].mean(),inplace=True) for feature in feature_list ]
    

# Read Fuctions ------------------------------------------------

# Save Fuctions ------------------------------------------------