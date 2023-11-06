import sys
#import pandas as pd
from config import setting,log,bertlog,bootlog

def LINE():
    return sys._getframe(1).f_lineno
def timestring (unixtime):
    from datetime import datetime
    ts = int(unixtime)
    s=datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return s

def f1_scores(trajectory, predicted):
    # precision,recall,fscore = f1_scores(trajectory, predicted)
    x=set(trajectory)
    y=set(predicted)
    intersecton=x.intersection(y)
    recall = len(intersecton) / len(trajectory)
    precision = len(intersecton) / len(predicted)
    #f1score = 2/( (1/recall + 1/precision) )
    if len(intersecton)==0:
        f1score = 0
    else:
        f1score = 2/( (1/recall + 1/precision) )
    return precision,recall,f1score


if __name__ == "__main__":\
    print( "# f1_scores([1,2,3,4], [2,3,6]) => \n", f1_scores([1,2,3,4], [2,3,6]) )
