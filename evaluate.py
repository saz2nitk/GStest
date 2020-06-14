import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import sys
import json
import warnings
warnings.filterwarnings('always') 

def load_data(learner_file, learner_workspace):
    y_learner = pd.read_csv(learner_file)
    y_actual = pd.read_csv(learner_workspace + 'actual.csv')
    return y_learner,y_actual

def validate_submission(y_learner):
    error = 0
    msg = "No error"
    if(list(y_learner.columns) != list(y_actual.columns)):
        msg = "The column names of the submission file do not match the submission format."
        error = 1

    if(y_learner.shape[0] != y_actual.shape[0]):
        msg = "The submission file should contain {} records".format(y_actual.shape[0])
        error = 1

    if(y_learner.shape[1] != y_actual.shape[1]):
        msg = "The submission file should contain {} columns".format(y_actual.shape[1])
        error = 1

    return error,msg

def score_submission(y_learner,y_actual):
    
    score =f1_score(y_actual.Score,y_learner.Score, average="weighted", labels=np.unique(y_actual.Score))
    
    if score > 0.63:
        points=1.0
    elif score <0.63 and score > 0.60 :
        points=0.8
    elif score < 0.6 and score > 0.55:
        points=0.7
    else:
        points=0
    
    
    return score,points

if __name__ == "__main__":
    learner_file = sys.argv[1]
    learner_workspace = sys.argv[2]
    y_learner,y_actual = load_data(learner_file, learner_workspace)
    err,msg = validate_submission(y_learner)
    if(err == 1):
        print(msg, end='')
    else:
        score,points = score_submission(y_learner,y_actual)
        result = json.dumps({'raw_score': score, 'multiplier': points})
        print(result, end='')