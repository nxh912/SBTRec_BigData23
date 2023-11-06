import random,time,os,sys,gc,collections,itertools
#import numpy as np
import pprint


### Global variables

theme2color = dict()
allthemes = ['Amusement', 'Architectural', 'Beach','Building','Cultural','Education','Entertainment','Historical','Museum','Park','Precinct','Religion','Religious','Shopping','Sport','Structure','Transport']
#cm = plt.cm.get_cmap('tab20b')
#for i in range(len(allthemes)):
#    index = i%len(cm.colors)
#    color=cm.colors[index]
#    theme2color[allthemes[i]] = color

setting = {
    ### CHARTING
    "Theme2Color"   : theme2color,
    'DEBUG'         : 1,
    'IN_COLAB'      : ('google.colab' in sys.modules),

    ### POIs Data
    "CITY"          : None,        ## Buda Delh Edin Glas Osak Pert Toro
    "POIs"          : None,        ## POI csv file
    "POIThemes"     : None,        ## themes for each poi
    "Themes"        : None,        ## main themes 

    ### GENSIM
    "COSINE_CALC"   : "STEP_WISE", ### ALL_PAIRS (average) / STEP_WISE (average)
    "Word2Vec"      : "skgr",      ### "skgr" / "cbow" / "fsg" / "fcb"
    "POI_SEARCH"    : "1000",      ### BRUTE_FORCE/ "1000" /...
    "GenSim_Model"  : None,

    #### "Cluster_Size"  : {'Osak':4, 'Toro': 6, 'Pert': 4, 'Buda': 7, 'Glas': 7, 'Edin': 6, 'Delh': 6},
    #"Cluster_Size"  : None,
    #"FOLDS_CROSS_VALIDATION": 5,
    #"FOLD"          : 0,
    #"AStar_Keep"    : 1000,

    ## LSTM
    "acticaton_function" : "relu",
    "loss_function" : 'mse',
    "optimizer" : 'Adam'

}

#np.set_printoptions(precision=3)
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)
#pd.set_option('display.width', 0)
#pd.options.display.width = 0

pp=pprint.PrettyPrinter(indent=4)

import logging,traceback
logging.basicConfig(format='#%(asctime)s |%(levelname)s-%(funcName)s| %(message)s', level=logging.ERROR)

bertlog = logging.getLogger('Bert')
bootlog = logging.getLogger('Bootstrap')
log = logging.getLogger('main')

if ('google.colab' in sys.modules):
    log.setLevel(logging.DEBUG)
    #bertlog.setLevel(logging.DEBUG)
    bootlog.setLevel(logging.INFO )
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.CRITICAL)
else:
    log.setLevel(logging.WARNING)
    log.setLevel(logging.DEBUG)
    bertlog.setLevel(logging.INFO)
    bootlog.setLevel(logging.WARNING )
    #bootlog.setLevel(logging.INFO)
    #bootlog.setLevel(logging.ERROR)
    #handler= logging.StreamHandler(sys.stderr)
    handler= logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    
formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
