# Data Handling
import pandas as pd
import numpy as np

# Misc
import pickle # saving/loading metrics
import argparse

# ML
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

# Text Processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
nltk.download('stopwords')
nltk.download('punkt')

def save_dict(dictionary, filename, verbose=False):
    '''
    Saves dictionary object as a pickle file for reloading and easy viewing
    
    Args:
    - dictionary (dict): data to be saved
    - filename (str): filename for dictionary to be stored in
    - verbose=False (bool): sepcifies if exact filename should be used. if False, .json extension appended to filename if not already present
    Return:
    - filename (str): filename for dictionary to be stored in
    '''
    if (not verbose) and ('.pickle' not in filename):
        filename += '.pickle'
        
    with open(filename, "wb") as outfile:  
        pickle.dump(dictionary, outfile)
        outfile.close()
    
    return filename
        
def load_dict(filename, verbose=False):
    '''
    Loads dictionary of metrics from given filename
    
    Args:
    - filename (str): file to load
    - verbose=False (bool): sepcifies if exact filename should be used. if False, .pickle extension appended to filename if not already present
    Return
    - dictionary (dict): data found in file
    - None (None): return None val in case exception is raised and dictionary file does not exist
    '''
    if (not verbose) and ('.pickle' not in filename):
        filename += '.pickle'

    try:
        with open(filename, 'rb') as pickle_file: 
            dictionary = pickle.load(pickle_file) 
    except FileNotFoundError as e:
        print(e)
        return None
    
    return dictionary

def load_yelp(dataset_dict):
    # Load yelp data sets
    yelp_test_df = pd.read_csv('../data/yelp_review_polarity_csv/test_0.csv', names=['label', 'data']) 
    yelp_train_df = pd.read_csv('../input/data/yelp_review_polarity_csv/train_0.csv', names=['label', 'data']) 

    # Since yelp data set is already split into test and train, recombine
    yelp_df = pd.concat([yelp_test_df, yelp_train_df])

    # Data set is too large to work with in memory since I don't have 2TiB of RAM just lying around, so we're cutting the data down
    yelp_df = yelp_df.sample(n=16000,replace=False,axis='index')

    # Change 1, 2 label to 0, 1 for uniformity with other data sets
    # Data set has 1 for negative and 2 for positive, so we switch 0 to negative and 1 to positive
    yelp_df['label'] = yelp_df['label'].apply(lambda label: 0 if label == 1 else 1)

    #Vectorize
    yelp_df['data'] = vectorizer.fit_transform(yelp_df['data']).toarray()

    # Transform df to np array for easier use & add info to dict
    yelp_data = yelp_df.values
    dataset_dict['yelp'] = yelp_data

    return dataset_dict

def load_subob(dataset_dict):
    # Load data sets
    subjectivity_df = pd.read_csv('../data/subjectobject/subjectivity.txt', sep='\n', encoding='latin-1', names=['data'])
    objectivity_df = pd.read_csv('../data/subjectobject/objectivity.txt', sep='\n', encoding='latin-1', names=['data'])

    # Add labels (subjective is 0, objective is 0)
    subjectivity_df['label'] = 0
    objectivity_df['label'] = 1

    # Combine data sets and rearrange columns for uniformity
    sub_ob_df = pd.concat([subjectivity_df, objectivity_df])
    sub_ob_df = sub_ob_df.reindex(columns=['label', 'data'])

    #Vectorize
    sub_ob_df['data'] = vectorizer.fit_transform(sub_ob_df['data']).toarray()

    #Transform df to np array, and add to dict
    sub_ob_data = sub_ob_df.values
    dataset_dict['sub_ob'] = sub_ob_data

    return dataset_dict

def load_clickbait(dataset_dict):
    # Load data sets
    clickbait_df = pd.read_csv('../data/clickbait/clickbait_data', sep='\n', names=['data'])
    nonclickbait_df = pd.read_csv('../data/clickbait/non_clickbait_data', sep='\n', names=['data'])

    # Add labels (subjective is 0, objective is 0)
    nonclickbait_df['label'] = 0
    clickbait_df['label'] = 1

    # Combine data sets and rearrange columns for uniformity
    clickbait_df = pd.concat([clickbait_df, nonclickbait_df])
    clickbait_df = clickbait_df.reindex(columns=['label', 'data'])

    #Vectorize
    clickbait_df['data'] = vectorizer.fit_transform(clickbait_df['data']).toarray()

    #Transform df to np array, and add to dict
    clickbait_data = clickbait_df.values
    dataset_dict['clickbait'] = clickbait_data

    return dataset_dict

def run_svm():
    # Create metric dict
    svm_metric_dict = {}

    # Create grid of parameters to search over for SVM
    c_vals = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    param_grid_svm = [{'kernel': ['linear'], 'C': c_vals}, {'kernel': ['poly'], 'degree': [2,3], 'C': c_vals}, {'kernel': ['rbf'], 'gamma': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2], 'C': c_vals}]
    # Create model & grid search object
    svc = SVC()
    clf_svc = GridSearchCV(estimator=svc, param_grid=param_grid_svm, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
    for name, dataset in dataset_dict.items():
        # Get data
        X, y = dataset[:, 1:], dataset[:, :1] #Treats first column as label
        for i in range(3):
            print('{} test {}'.format(name, i))
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, shuffle=True)
            
            clf_svc.fit(X_train, y_train.ravel()) # Fit training data to model
            
            # Train set performance
            y_train_pred = clf_svc.predict(X_train)
            acc_train = accuracy_score(y_train, y_train_pred)
            precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_train, y_train_pred)
            
            # Test set performance
            y_test_pred = clf_svc.predict(X_test) # Predict test values using best parameters from classifier
            acc_test = accuracy_score(y_test, y_test_pred) # Get accuracy for predictions
            precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, y_test_pred)
            
            svm_metric_dict[(name, i)] = {'acc_test': acc_test, 'acc_train': acc_train, 'precision_test': precision_test, 'precision_train': precision_train, 'recall_test': recall_test, 'recall_train': recall_train,
                                        'f1_test': f1_test, 'f1_train': f1_train, 'model': clf_svc, 'cv_results': clf_svc.cv_results_} # Add metrics to dict for analysis
            save_dict(svm_metric_dict, '../checkpoints/svm/svm_{}_{}.pickle'.format(name, i)) # Save checkpoint results in case of hardware failure

if __name__ == '__main__':
    # Create dict to store {name: dataset}
    dataset_dict = {}
    # Create vectorizer that turns text samples into token vector 
    vectorizer = CountVectorizer(analyzer='char', tokenizer=word_tokenize, stop_words=stopwords.words('english'))

    # Gather runtime arguments
    parser = argparse.ArgumentParser(description='Gather runtime information')
    parser.add_argument('--yelp', dest='yelp', action='store_true', help='specifies if yelp data should be used')    
    parser.add_argument('--subob', dest='subob', action='store_true', help='specifies if subob data should be used')    
    parser.add_argument('--clickbait', dest='clickbait', action='store_true', help='specifies if clickbait data should be used')    
    parser.add_argument('--svm', dest='svm', action='store_true', help='specifies if svm model should be used')

    args = parser.parse_args()

    # Load data sets
    if args.yelp:
        dataset_dict = load_yelp(dataset_dict)
    if args.subob:
        dataset_dict = load_subob(dataset_dict)
    if args.clickbait:
        dataset_dict = load_clickbait(dataset_dict)    

    # Run models
    if args.svm:
        run_svm()