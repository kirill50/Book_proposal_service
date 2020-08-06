import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from urllib import request, parse
import urllib.request
import json


def get_prediction(original_title ):
    body = {'original_title ': original_title}

    myurl = "http://localhost:5000/predict"
    req = urllib.request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))
    #print (jsondataasbytes)
    response = urllib.request.urlopen(req, jsondataasbytes)
    return json.loads(response.read())['predictions']

if __name__=='__main__':
    original_title  = '''The Great Gatsby'''
    preds = get_prediction(original_title)
    print(preds)
