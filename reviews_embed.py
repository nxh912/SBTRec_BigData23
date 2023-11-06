import time,os,sys,gc
import argparse
import math
import pandas as pd
import numpy as np

from summarizer import Summarizer
from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs,
    MultiLabelClassificationModel, MultiLabelClassificationArgs)
from transformers import AutoTokenizer, AutoModel

from config import setting,log
from common import f1_scores, LINE, bertlog
from poidata import (load_files, load_dataset, getThemes, getPOIFullNames, getPOIThemes, poi_name_dict)
from Bootstrap import inferPOITimes2,get_distance_matrix

global NEARBY_POIS,MIN_SCORE
NEARBY_POIS = 0
MIN_SCORE = -9999

global theme2num
global num2theme
global poi2themes
global dist_mat
global dist_dict
dist_dict=None

def distance_dict(pois):
    allpois = pois['poiID'].unique()
    poi_dict = dict()
    dist_mat = get_distance_matrix(pois)
    poi_dict_p = dict()
    for p in allpois:
        ### allpois[p] -> [p1:dist1, p2:dist2 ,....] by distance
        for q in allpois:
            if p != q:
                pq_dist = dist_mat[ (p,q) ]
                poi_dict_p[q] = dist_mat[(p,q)]
        sorted_poi_dict_p = dict(sorted(poi_dict_p.items(), key=lambda x: x[1]))
        poi_dict[p] = sorted_poi_dict_p.copy()
    return poi_dict

def get_model_args(bert, city, epochs, pois, train_df):
    model_args = ClassificationArgs()
    # model_args.optimizer = ...
    model_args.no_deprecation_warning=True
    model_args.num_train_epochs = epochs
    model_args.reprocess_input_data = True,
    model_args.overwrite_output_dir = True
    model_args.disable_tqdm = True
    model_args.use_multiprocessing = True

    model_args.use_early_stopping = True
    model_args.early_stopping_delta = 0.001
    model_args.early_stopping_metric = "mcc"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_patience = 5
    model_args.evaluate_during_training_steps = 5
    model_args.use_multiprocessing = True
    model_args.use_multiprocessing_for_evaluation = True
    model_args.use_multiprocessed_decoding = True
    model_args.no_deprecation_warning=True

    ### output / save disk space
    model_args.save_steps = -1
    model_args.save_model_every_epoch = False

    model_args.output_dir = ("/var/tmp/output/output_{}_e{}_{}".format(city, epochs, bert))
    bertlog.debug("LINE {}, FUNC: model_args.output_dir : {}".format(LINE(), model_args.output_dir))
    model_args.disable_tqdm = True

    #model_ args.overwrite_out_put_dir = True
    #model_ args.out_put_dir = "out_put/out_put_{}_e{}".format(city,epochs)

    #### PRINT WHOLE DATA TABLE
    pd.set_option('display.max_rows', None)
    bertlog.info("LINE %d, TRAINING PARAMS: %s".format(LINE(), city, str(model_args)))
    bertlog.info("LINE %d, TRAINING POIs:\n%s\n\n".format(LINE(), city, str(pois)))
    bertlog.info("LINE %d, TRAINING DATA:\n%s\n\n".format(LINE(), city, str(train_df)))

    model_args.early_stopping_delta = 0.00001
    model_args.early_stopping_metric = "mcc"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_patience = 5
    model_args.evaluate_during_training_steps = 10000
    return model_args

def get_bertmodel(model_str):
    LABEL_TYPE=''
    MODEL_NAME=''
    # default BERT model

    if model_str in ['bert', 'BERT']:
        LABEL_TYPE='bert'
        MODEL_NAME='bert-base-uncased'
        #bertlog.info("USING BERT PREDICTION (model_str)")
    elif model_str in ['roberta', 'Roberta']:
        LABEL_TYPE='roberta'
        MODEL_NAME='roberta-base'
        #bertlog.info("USING roberta-base PREDICTION (model_str)")
    elif model_str in ['albert','Alberta']:
        LABEL_TYPE='albert'
        MODEL_NAME='albert-base-v2'
        #bertlog.info("USING albert-base PREDICTION (model_str)")
    elif model_str in ['XLNet','xlnet']:
        LABEL_TYPE='xlnet'
        MODEL_NAME='xlnet-base-cased'
        #bertlog.info("USING XLnet-base PREDICTION (model_str)")
    elif model_str in ['distilbert','DistilBert']:
        LABEL_TYPE='distilbert'
        MODEL_NAME='distilbert-base-cased'
        #bertlog.info("USING distilbert-base PREDICTION (model_str)")
    elif model_str =='xlm':
        LABEL_TYPE='xlm'
        MODEL_NAME='xlm-base'
        #bertlog.info("USING xlmnet PREDICTION")
    elif model_str=='xlmroberta':
        LABEL_TYPE='xlmroberta'
        MODEL_NAME='xlmroberta-base'
        #bertlog.info("USING xlmroberta-base PREDICTION")
    ## XLNet, XLM
    else:
        if bertlog: bertlog.error("Unknown Model:  %s".format(model_str))
        assert(False)
    return LABEL_TYPE, MODEL_NAME

def train_bert_model(city, pois, array, epochs, model_str, use_cuda=True):
    # torch.cuda.is_available()
    log=bertlog
    log=None

    if log: log.info("START MODEL... city: %s, epochs:%d, model_str: %s", city, epochs, model_str)
    npois = pois['poiName'].count()
    theme2num, num2theme, poi2theme = get_themes_ids(pois)
    train_data=[]

    #assert( len(array)==len(nearbys) )
    #for i in range(len(nearbys)):
    #    if log: log.info("line:%d [%d] {} <--> {}".format(LINE(),i,array[i],nearbys[i]))

    ### BERT
    if log: log.info("(1) ### model_str  : %s", model_str)
    NUM_LABELS= 1+ npois+ len(theme2num)
    LABEL_TYPE, MODEL_NAME = get_bertmodel(model_str)

    #bertlog.debug("array : " , items)
    for items in array:
        #print("LINE_152. list_subseqs : ", items)
        bertlog.debug("LINE %d, %s", LINE(), items)
        # [ uid, location, poiid, poitheme, comments (NEW),.. .]
        assert(len(items)>5)
        listA = items[:-5]

        strlistA = [str(i) for i in listA]

        resulta = items[-3]
        resultb = items[-2]
        resultc = items[-1]

        assert(len(strlistA) % 5 == 0)
        trainItem=",".join(strlistA)
        # e.g. 65072601,9,Park,65072601,12,Cultural

        past_pois = listA[1::5]
        cur_poi = past_pois[-1]

        train_data.append( [ trainItem, resulta ] )
        #print( [ trainItem, resulta ] )

    #print("line_170: train_data :\n", train_data)
    
    train_df = pd.DataFrame(train_data)
    train_df.columns = ["text", "labels"]

    bertlog.info("line_183: train_df :\n%s", train_df)

    if log: log.debug("train_df:\n%s\n", str(train_df))
    if log: log.info("(3) ### model_type : %s", LABEL_TYPE)
    if log: log.info("(4) ### model_name : %s", MODEL_NAME)
    assert(LABEL_TYPE)
    assert(MODEL_NAME)
    model_args = get_model_args(model_str, city, epochs, pois, train_df)
    bertlog.debug("LINE %d, train_df <%s> :\n%s", LINE(), str(train_df.shape), str(train_df))
    #bertlog.debug("LINE %d, torch.cuda.is_available: %s", LINE(), ("Yes" if torch.cuda.is_available else "No"))

    cudadev = 0
    if   city in ['Melb','Buda']: cudadev = 1
    elif city in ['Toro','Vien']: cudadev = 2
    else: cudadev=0 #assert(city in ['Osak','Pert','Delh','Edin','Glas'])


    try:
        model = ClassificationModel(model_type= LABEL_TYPE, \
                                    model_name= MODEL_NAME, \
                                    num_labels= NUM_LABELS, \
                                    use_cuda=  (use_cuda and torch.cuda.is_available),   \
                                    cuda_device=cudadev,    \
                                    args=       model_args)
        model.train_model(train_df, no_deprecation_warning=False)
        if log: log.info("(4) line %d : train_ city_bert_model RETURNING : %s", LINE(), str(model))
        if log: log.info("(5) >>> ... train_ city_bert_model")
    except Exception as err:
        print( err)
        bertlog.error("Line %d, ERROR: %s", LINE(), str(err))
        return None,None
    bertlog.debug("train_bert_model... train_df :\n%s\n\n", str(train_df))
    return model,train_df ## train_bert_model

def getTrajectories(pois, userVisits):
    trajectories=dict()
    for userid in userVisits['userID'].unique():
        if userid not in trajectories: trajectories[userid] = []
        userid_visit = userVisits[ userVisits['userID'] ==userid ]

        ### TRAINING DATA
        for seqid in userid_visit['seqID'].unique():
            seqtable = userid_visit[ userid_visit['seqID'] == seqid ]
            seqtable.sort_values(by=['dateTaken'])

            pids = list(seqtable['poiID'])
            # remove duplicate
            pids = list(dict.fromkeys(pids))

            #sentense_list.append(pids)
            trajectories[userid].append(pids)

    return trajectories

def predict_mask_pos(pois, model, seq, maskpos):
    print(f"\n\n\n-- predict_mask_pos / ( <model>, seq= {seq}, maskpos= {maskpos}  )")

    theme2num, num2theme, poi2theme = get_themes_ids(pois)
    numpois=len(poi2theme)
    numthemes=len(theme2num)

    ## GIVEN argmax_theme
    ## next predict POI-ID given theme
    maskedseq=[]

    ## PART-1
    for poi in seq[:maskpos]:
        # theme
        maskedseq.append( poi )
        maskedseq.append( poi2theme[poi] )
    # PART-2
    maskedseq.append('[MASK]')
    # PART-3
    for poi in seq[maskpos:]:
        maskedseq.append( poi )
        maskedseq.append( poi2theme[poi] )

    predict_str=",".join([ str(i) for i in maskedseq ])
    predictions, raw_outputs = model.predict( to_predict=[predict_str] )
    prediction=predictions[0]
    raw_output=raw_outputs[0]

    ### STEP-1  ... SKIP VISITED POIs
    for i in range(len(seq)):
        poi=seq[i]
        raw_output[ int(poi) ] = -999999

    ### STEP-2  ... LOOK UP only POIS in raw_ouput
    poi_raw_output = raw_output[1:numpois]

    ### STEP-3 SCALE near-by POIs
    ## amax starts from poi-0
    amax = 1+int(np.argmax(poi_raw_output))
    amaxval = raw_output[amax]

    if amaxval <= -999999:
        ### when predicted (amax) is already in seq
        ### there is no more POIs to predict,
        return None, None, None

    assert(amax not in seq)
    assert(maskedseq[maskpos*2] == '[MASK]')
    unmasked = maskedseq.copy()
    unmasked[maskpos*2] = amax

    bertlog.debug("line %d,             amax : %s", LINE(), str(amax))
    bertlog.debug("line %d, poi2theme.keys() : %s", LINE(), str(poi2theme.keys()))
    assert(amax in poi2theme.keys())
    amax_theme = poi2theme[amax]
    unmasked.insert(maskpos*2+1, amax_theme)
    unmasked_pois=unmasked[0::2]
    return (amax,amaxval,unmasked_pois)

def predict_mask(pois, model,predseq):
    predseq_str=[str(i) for i in predseq]
    possible_unmasked={}
    for maskpos in range(1,len(predseq)):
        nextpoi, nextval, unmasked_seq = predict_mask_pos(pois, model, predseq, maskpos)
        if maskpos and nextpoi and nextval > 0:
            possible_unmasked[maskpos] = nextpoi, maskpos, nextval, unmasked_seq

    possible_unmasked= dict( sorted(possible_unmasked.items(), key=lambda item: item[1], reverse=True))
    if len(possible_unmasked) > 0:
        assert(len(possible_unmasked) > 0)
        for key in possible_unmasked:
            nextpoi, maskpos, nextval, unmasked_seq = possible_unmasked[key]
            return nextpoi, maskpos, nextval, unmasked_seq
    bertlog.error("LINE %d -- no prediction is found for [%s]", LINE(), predseq)
    return None,None,None,None

def estimate_duration(predseq, durations):
    for p in predseq: assert(str(p) != "")
    for p in predseq:
        print("[estimate_duration] -- predseq -> item: {}, type: {}".format(p,type(p)))
        assert(int(str(p)))     
    assert('[MASK]' not in predseq)

    total_duration = 0
    #poiids = predseq[2::5]
    #bertlog.debug("line %d.. poiids  : %s",   LINE(), poiids)
    #bertlog.debug("duration... line %d.. . predseq : %s",   LINE(), predseq)
    #bertlog.debug("line %d... durations : %s", LINE(), durations)
    
    for p in predseq:
        if type(p) == str: p=int(p)
        intertimes = durations[p] if p in durations else [1, 5 * 60]
        duration = math.ceil(max(intertimes))
        total_duration += duration
    bertlog.debug("[duration] -- line %d..  predseq: %s -> total_duration : %s", LINE(), str(predseq), str(total_duration))
    return total_duration

# def predict_mask(pois, model,predseq):
def predict_seq(pois, model, p1, pz, seqid_duration, boot_duration):
    predseq=[p1,pz]
    num_pois=pois['poiID'].count()

    bertlog.debug(" line %d, predict_seq(pois, model, %d, %d, %d, boot_duration)", LINE(), p1,pz,seqid_duration)
    for iter in range(num_pois):
        bertlog.debug("predict_seq, iter:%d", iter)

        ### INPUT: predseq
        nextpoi, maskpos, nextval, unmasked_seq = predict_mask(pois, model, predseq)
        if not nextpoi: break ### cannot predict next poi
        ## estimate duratiion of new_predseq
        predseq = unmasked_seq
        print("LINE_480: predseq : ", predseq)
        print("LINE_480: unmasked_seq : ", unmasked_seq)
        print("LINE {} unmasked_seq,: {}".format(LINE(), unmasked_seq))
        
        assert( len(predseq) > 2) ## NO prediction
        poi_duration = estimate_duration(unmasked_seq, boot_duration)
        bertlog.debug("predict_seq, iter:%d, predseq:%s", iter, str(predseq))
        bertlog.debug("predict_seq, iter:%d, poi_duration:%d", iter, poi_duration)
        assert(len(predseq)>2) ## NO PREDICTION
        
        if poi_duration > seqid_duration: break
    ### END predict_seq
    bertlog.debug(" line %d, predict_seq(pois, model, %d, %d, %d, boot_duration) : %s", LINE(), p1,pz,seqid_duration, predict_seq)
    return predseq

def getUserLocation():
    bertlog.info("line %d, reading users info: Data/user_hometown.csv", LINE())
    user2city = dict()
    usercity_df = pd.read_csv("Data/user_hometown.csv", \
                              sep=';',
                              keep_default_na=False,
                              na_values='_',
                              dtype={'UserID':str, 'JoinDate':str, 'Occupation':str, 'Hometown':str, 'current_city':str, 'country':str} )

    for i, row in usercity_df.iterrows():
        id           = row['UserID']
        current_city = row['current_city']
        country      = row['country']
        current_city = " ".join([ w.strip() for w in current_city.split(",") ])
        country      = " ".join([ w.strip() for w in country.split(",") ])

        if country=='Unknown' and current_city=='Unknown':
            user2city[ id ] = "Unknown"
        elif country=='':
            user2city[ id ] = current_city
        elif current_city=='':
            user2city[ id ] = country
        else:
            user2city[ id ] = country.strip() + " " + current_city.strip()
        #bertlog.info(f"line %d, ID: %s => ``%s''", LINE(), id, user2city[ id ] )
    setting["User2City"]=user2city
    return user2city

def getUserCity(userid):
    if "User2City" not in setting:
        user2city= getUserLocation()
        for user in user2city:
            print(" | user2city | '{}' -> '{}'".format(user, user2city[user]))
    else:
        user2city=setting["User2City"]
    if userid in user2city:
        return user2city[userid]
    else:
        #bertlog.error(" userid (%s) not found", userid)
        return None

def bert_train_city(city, pois, userVisits, epochs, summary_size, model_str='bert', USE_CUDA=True):
    import torch
    cuda_available = torch.cuda.is_available()
    bertlog.info("LINE %d, cuda_available : %s", LINE(), cuda_available)
    
    theme2num, num2theme, poi2theme = get_themes_ids(pois)
    sentense_list=[]
    list_subseqs = []
    trajectories = getTrajectories(pois, userVisits)

    if "poi_comments" in setting:
        #print(setting["poi_comments"])
        poi_comments = setting["poi_comments"]
        pass
    else:
        ### READ COMMENTS
        poi_comments = read_reviews(city, pois, summary_size=summary_size)
        setting["poi_comments"] = poi_comments

    poi_comments= setting["poi_comments"]
    #for poi in poi_comments: print("\nline {}, reviews( [POI] ) : {}  ==> {}".format(LINE(), poi, poi_comments[poi][:80] ))

    dist_dict = distance_dict(pois)
    #print(trajectories)
    for userid in trajectories:
        trajectory = trajectories[userid]
        for tryj in trajectories[userid]:
            bertlog.info("line %d, userid:%s, trajectory: %s", LINE(), userid, str(tryj))
            n=len(tryj)
            for head in range(0,n-1):
                for seqlen in range(2,n-head+1):
                    subseq=tryj[head:head+seqlen]
                    subseq2=[]
                    for pid in subseq:
                        user_city= getUserCity(userid)
                        if not user_city: user_city = "Unknown"

                        ## starting with USER-ID
                        subseq2.append(userid)
                        ### user's city
                        subseq2.append(user_city)
                        ### POI ID
                        subseq2.append(pid)
                        ### POI theme
                        subseq2.append(poi2theme[pid])
                        ### POI comments
                        subseq2.append(poi_comments[pid])

                    list_subseqs.append(subseq2)
                    #bertlog.info ("LINE %d ==> SubSeq with Themes: %s", LINE(), str(subseq2))

    for sseq in list_subseqs: bertlog.debug ("LINE %d ==> Training Data : %s (%s)",
                                              LINE(), str(sseq), len(sseq))
    cuda_available = torch.cuda.is_available()
    model,train_df = train_bert_model(city, pois, array=list_subseqs, epochs=epochs, model_str=model_str, use_cuda=(USE_CUDA and cuda_available) )
    assert(model)
    assert(not train_df.empty)
    bertlog.info(model)
    bertlog.info(train_df)
    return model,train_df

def get_themes_ids(pois):
    theme2num=dict()
    num2theme=dict()
    poi2theme=dict()
    numpois = pois['poiID'].count()

    allthemes=sorted(pois['theme'].unique())
    for i in range(len(allthemes)) :
        theme2num[allthemes[i]] = i
        num2theme[i] = allthemes[i]

    arr1 = pois['poiID'].array
    arr2 = pois['theme'].array

    for i in range(len(arr1)):
        pid   = arr1[i]
        theme = arr2[i]
        poi2theme[pid] = theme
        if theme not in theme2num.keys():
            num = numpois + len(theme2num.keys())
            theme2num[theme] = num
            num2theme[num] = theme
    return theme2num, num2theme, poi2theme

def predict_user(bertmodel, pois, user, seqid, p0, pn, t0, tn, seconds, path):
    bertlog.debug("LINE:%d ============================================", LINE())
    bertlog.debug("LINE:%d, -- predict_user( %s, %d,%d,%s,%s, seconds allowed:%d, path:%s )",  LINE(), user, p0,pn,t0,tn , seconds, str(path))
    bertlog.debug("LINE:%d = predict_user(<Model>, [pois],", LINE())
    bertlog.debug("LINE:%d =              user:%s, p0:%d, pn:%d, t0:'%s', tn:'%s'", LINE(), user, p0, pn, t0, tn)
    bertlog.debug("LINE:%d ============================================", LINE())

    assert(bertmodel)
    assert(not math.isnan(seconds))
    assert("poi_comments" in setting)
    assert(setting["poi_comments"])

    poi_comments = setting["poi_comments"]
    
    numpois = pois['poiID'].count()
    assert(setting["User2City"])
    user2city=setting["User2City"]
    if user in user2city:
        hometown = user2city[user] or 'Unknown'
    else:
        hometown="Unknown"
    assert(hometown)
    assert(hometown!="")

    poi_embed1="POI_EMBED1"
    poi_embed2="POI_EMBED2"
    if p0 in poi_comments: poi_embed1 = poi_comments[p0]
    if pn in poi_comments: poi_embed2 = poi_comments[pn]

    print(f"LINE {LINE()} POI-1 embed: '{poi_embed1}' ")
    print(f"LINE {LINE()} POI-K embed: '{poi_embed2}' ")
    
    context_arr=[ user,hometown,str(p0),str(t0),poi_embed1,  user,hometown,str(pn),str(tn),poi_embed2 ]
    
    print(f"LINE {LINE()} context_arr : {context_arr}")

    boot_duration = setting['bootstrap_duration']
    assert( boot_duration )

    ### get user HOMETOWN
    user_location = getUserCity(user)
    if not bool(user_location): user_location='Unknown'

    ### predict duration for just []
    predited_arr = [ user, user_location, p0, t0, poi_embed1,
                     user, user_location, pn, tn, poi_embed2 ]
    poiid_seq = predited_arr[2::5]
    #bertlog.debug("LINE:%d INIT predited_arr: '%s'", LINE(), predited_arr)

    ### default var
    ret_predited_arr = predited_arr
    ret_seq_duration = estimate_duration(poiid_seq, boot_duration)

    assert(len(predited_arr) % 5 == 0)
    assert(ret_predited_arr[1]) ### location not None

    # estimate_duration
    assert(ret_seq_duration)

    ### predict next POI, for max. numpoi times
    ### for i in range(numpois):
    i=0
    #bertlog.debug("LINE %d, loop? (%s) ret_seq_duration(%d) < seconds(%d) ? ", LINE(), (ret_seq_duration < seconds), ret_seq_duration, seconds)

    while ret_seq_duration < seconds:
        i = i+1
        #print(f"LINE:{LINE()} --------------------------------------------" )
        #print(f"LINE:{LINE()} ---   ITERATION {i} -- u:{user} -- limit (sec):{seconds}" )
        #print(f"LINE:{LINE()} --------------------------------------------", LINE())

        ### PREDICTION
        print(f"LINE {LINE()} --- context_arr => {context_arr   } ")
        unmask_score, predited_arr = predict_user_iterninary( context_arr, bertmodel, pois)
        ### PREDICTION

        if predited_arr:
            assert('[MASK]' not in predited_arr)
            assert( context_arr[2] == predited_arr[2]) ## POI-1
            assert( context_arr[-3] in  predited_arr) ## POI-N

            ### ESTIMATE SEQUENCE DURATION
            assert(len(predited_arr) % 5 == 0)
            #bertlog.info("line %d, estimate_ duration(predited_arr, boot_duration).. . ", LINE() )
            #bertlog.info("line %d, predited_arr : %s", LINE(), predited_arr )
            assert("[MASK]" not in predited_arr)

            #bertlog.info("line %d, boot_duration : %s", LINE(), boot_duration )
            predseq = predited_arr[2::5]
            #bertlog.info("line %d, predseq : %s", LINE(), predseq )

            poi_duration = estimate_duration(predseq, boot_duration)
            ### next iteration
            context_arr = predited_arr
            context_pois = [int(p) for p in predited_arr[2::5]]
            assert(len(context_pois) == len(set(context_pois)))
            assert(len(predited_arr) % 5 == 0)

            # calc duration from boot_duration
            predseq = predited_arr[2::5]

            assert("[MASK]" not in context_pois)
            seq_duration = estimate_duration(context_pois, boot_duration)
            assert(seq_duration)

            #bertlog.info("LINE:%d, duration:%d (budget:%d)\n\n\n\n", LINE(), seq_duration, seconds)
            ret_predited_arr = predited_arr
            assert(ret_predited_arr[1]) ### location not None
            assert(ret_predited_arr[1]) ### location not None
            
            ### TIME BUDGET OVER??
            if bool(predseq) and bool(seq_duration) and ret_seq_duration > seconds:
                #if seq_duration: ret_seq_duration = seq_duration
                #print(    "LINE:{} ===>  predseq          : {} -- ".format(LINE(), predseq))
                #print(    "LINE:{} ===>  ret_predited_arr : {} -- ".format(LINE(), ret_predited_arr))
                #print(    "LINE:{} ===>  ret_predited_arr : {} -- ".format(LINE(), ret_predited_arr))
                #print(    "LINE:{} ===>  seconds          : {} -- ".format(LINE(), seconds))
                #print( "\n)))\n\n")
                break
        else:
            bertlog.info("LINE:%d, NO MORE RESULT from  %s", LINE(), predited_arr)
            break
        print( "\n)))\n\n")

    print(f"LINE {LINE()}  ret_seq_duration( {ret_seq_duration} ) < seconds ( {seconds} )")

    bertlog.debug("")
    bertlog.debug("LINE %d, AFTER %d INSERTIONS...", LINE(), i)
    bertlog.debug("LINE:%d     FINAL [ %d ,..., %d ] duration estimated: %d secs", LINE(), p0, pn, ret_seq_duration)
    bertlog.debug("LINE:%d *** FINAL ret_predited_arr (with themes) : %s ***", LINE(),  ret_predited_arr)
    bertlog.debug("LINE:%d *** FINAL ret_seq_duration : %s ***", LINE(),  ret_seq_duration)
    bertlog.debug("LINE:%d *** FINAL p0 : '%s', pn : '%s' ***", LINE(), str(p0), str(pn))
    bertlog.debug("LINE:%d *** FINAL p0 : '%s', pn : '%s' ***", LINE(), str(ret_predited_arr[2]), str(ret_predited_arr[-3]))
    bertlog.debug("LINE:%d *** FINAL NUM POIs : '%d' ***", LINE(), len(ret_predited_arr)/5)
    
    assert( str(ret_predited_arr[2] ) == str(p0) )
    assert( str(ret_predited_arr[-3]) == str(pn) )

    assert(ret_predited_arr)
    assert(ret_predited_arr[1]) ### location not None
    assert(ret_seq_duration)

    assert( len(ret_predited_arr) > 2 )
    #assert( ret_predited_arr[0]==user ) ## first-poi user
    #assert( ret_predited_arr[-3]==user ) ## last-poi user
    bertlog.info("LINE:%d   PREDICTED SEQ: %s, DURATION: %s / %d s--(limit)\n\n", \
                 LINE(), ret_predited_arr, ret_seq_duration, seconds)
    assert(len(ret_predited_arr) >= 10)
    return (ret_predited_arr, ret_seq_duration)

def predict_user_insert(context_arr, bertmodel, pois, pos):
    log=bertlog
    log=None

    if log: log.debug("\n\nnLINE %d <<<<<<<<<<<<<<<<< predict_user_insert", LINE())
    assert( '[MASK]' not in context_arr )
    assert( len(context_arr) % 5 == 0)
    assert( bertmodel)

    numpois=pois['poiID'].count()
    theme2num, num2theme, poi2theme = get_themes_ids(pois)

    userid = context_arr[0]
    poiids = context_arr[2::5]
    #bertlog.debug("LINE %d poiids in context : %s", LINE(), ", ".join(poiids))

    context_arr.insert(pos*5, '[MASK]')
    context_arr.insert(pos*5+1, 'theme')
    #context_arr.insert(pos*5+2, context_arr[4]) # COMMENT
    context_arr.insert(pos*5+2, '(poi_comments)') # COMMENT
    context_arr.insert(pos*5, context_arr[1])
    context_arr.insert(pos*5, context_arr[0])

    assert(len(context_arr) % 5 == 0) # addition [userid] [mask] tokens
    for i in range(len(context_arr)):
        if (i % 5 == 2): context_arr[i]='__comment_vec__'
        print(f"  context_arr[ {i} ] --> {context_arr[i]}\n")

    
    maskedstr = ','.join(context_arr)

    predictions, raw_outputs = bertmodel.predict( to_predict=[maskedstr] )


    raw_output = raw_outputs[0]
    ## ignore past pois
    for pastpoi in poiids: raw_output[int(pastpoi)] = MIN_SCORE
    for i in range(numpois, len(raw_output)): raw_output[i] = MIN_SCORE

    poi_output=raw_output[1:]
    amax = int(np.argmax(poi_output))
    predict_poi = amax+1

    ### CHECK POSITIVE PREDICTION
    if amax < 0:
        bertlog.info("LINE %d => NO RESULT: context_arr: {%s} , pos: %d)", LINE(), context_arr, pos)
        assert(False)
        return None,None
    elif amax > numpois:
        bertlog.info("LINE %d => NO RESULT \n  context_arr: {%s} , pos: %d)", LINE(), context_arr, pos)
        ### NO MORE POI TO
        return None,None
    elif str(predict_poi) in context_arr:
        ### NO MORE PREDICTION
        return None,None
    else:
        assert(str(predict_poi) not in context_arr)
        amaxval = raw_output[amax]

        predicted_arr = context_arr.copy()

        ### FILL IN ARG_MAX & THEME
        assert( str(predict_poi) not in context_arr)
        assert('[MASK]' in predicted_arr)
        assert( predicted_arr[pos*5+2]  == '[MASK]')
        assert( predicted_arr[pos*5+3]  == 'theme')
        assert( poi2theme[predict_poi])

        predicted_arr[pos*5+2] = str(predict_poi)
        predicted_arr[pos*5+3] = poi2theme[predict_poi]

        # POST CHECK
        assert( predicted_arr[pos*5+2] != '[MASK]')
        assert( predicted_arr[pos*5+3] != 'theme' )
        assert('[MASK]' not in predicted_arr)

        # AFTER POI PREDICTION
        #print("LINE {}  predicted_arr => {}".format(LINE(), predicted_arr))

        ### CHECK UNIQUE
        poiids = predicted_arr[2::5]
        assert( len(poiids) == len(set(poiids)) )
        assert(context_arr[pos*5+2] == '[MASK]')
        assert(len(predicted_arr) % 5 == 0)

        poiids = predicted_arr[2::5]
        #if log: log.debug("LINE %d >>> predicted_IDS: %s\n", LINE(), poiids)
        assert( len(poiids) == len(set(poiids)) )
        #print(f"LINE {LINE()} >>>>>>>>>>>>>>>>> predict_user_insert\n%s\n{predicted_arr}\n")
        assert('[MASK]' not in predicted_arr)
        return (amaxval, predicted_arr)

def predict_user_iterninary(context, bertmodel, pois):
    print(" context : ", context)
    bertlog.info(" ---> predict_user_iterninary( '%s', bertmodel, pois)", str(context))
    assert('[MASK]' not in context)
    assert(bertmodel)
    assert(len(context) % 5 == 0)

    context_array = [str(el) for el in context[2::5] ]
    context_text = ",".join(context_array)

    #bertlog.debug("BEGIN _iterninary(context: [%s], bertmodel, pois)", context_text)
    #bertlog.debug("BEGIN _iterninary(context: [%s], bertmodel, pois)", context_text)
    #bertlog.debug("BEGIN _iterninary(context: [%s], bertmodel, pois)", context_text)

    n=int(len(context) / 5)
    assert(len(context) % 5 == 0)

    #bertlog.debug("n=%d, context : %s", n, context)
    predicted_prob,predicted_arr = None,None

    poiids = pois['poiID'].unique()
    theme2num, num2theme, poi2theme = get_themes_ids(pois)
    numpois = len(poi2theme.keys())

    # dict : [ pval -> unmasked ]
    # predicted poi in context
    ### PREDICT n x [MASK]
    predicted_arr_dict=dict()
    for idx in range(1,n):
        #print(f"\n\n\n===> idx={idx}, n={n}, context : '{context}'")
        context2 = context.copy()
        #print(f"**************************** (A) idx : {idx}")

        #print(f"line {LINE()}, MASK:{idx}, context2 : \n'{context2}'")
        assert('[MASK]' not in context2)
        assert(len(context2) % 5 == 0)

        #print(f"******* LINE {LINE()} predict_user_insert( idx: {idx}, n: {n}\n")
        assert(len(context2) % 5 == 0)

        pval,context_arr2 = predict_user_insert(context2, bertmodel, pois, idx)

        #print(f"line {LINE()}: context2 -> {context_arr2}")
        assert( len(context_arr2) % 5 == 0)
        assert( context_arr2[1] != '')
        if context_arr2 and pval > MIN_SCORE:
            assert(context_arr2)
            assert('[MASK]' not in context_arr2)
            assert(len(context_arr2) % 5 == 0)
            predicted_arr_dict[pval] = context_arr2
            #print(f"LINE {LINE()} POSSIBLE INSERTION -->  predicted_arr_dict : [ {pval} ] --> context_arr2 : {context_arr2}\n"  )
            assert(len(context_arr2) % 5 == 0 )
            poiids = context_arr2[2::5]
        else:
            #bertlog.debug("line %d, context_arr2 : %s", LINE(), context_arr2)
            #bertlog.debug("line %d, pval : %f", LINE(), pval)
            assert (pval==None or pval <= MIN_SCORE or context_arr2 == None)
            ### no more prediction from BERT model
            break

    ## checking all posible solution in predicted_arr_dict
    if predicted_arr_dict:
        for k in sorted(predicted_arr_dict.keys()): bertlog.info("###### line %d POSSIBLE (%s) -> (%f)", LINE(), predicted_arr_dict[k], k )

        keys=predicted_arr_dict.keys()


        maxkey = max(predicted_arr_dict.keys())
        maxval = predicted_arr_dict[maxkey]


        predicted_arr = maxval
        predicted_prob= maxkey




        assert('[MASK]' not in predicted_arr)

        context_pois=predicted_arr[2::5]
        context_pois=[int(p) for p in context_pois]

        #print ("predicted_prob : ", predicted_prob)
        #print ("predicted_arr  : ", predicted_arr)
        return (predicted_prob,predicted_arr)
    else:
        bertlog.debug("END _iterninary(context: '%s', bertmodel, pois)\n\n",       context)
        bertlog.debug("END _iterninary(context: '%s', bertmodel, pois)\n\n",       context)
        bertlog.debug("END _iterninary(context: '%s', bertmodel, pois)\n\n\n\n\n", context)
        return None,None

def test_user_preference_seqid(city, epochs, u, seqid, boot_duration, pois, user_visit, bertmodel):
    #### USER SEQID -> PREDICTION
    theme2num, num2theme, poi2theme = get_themes_ids(pois)
    user_visit_seqid_dt = user_visit[user_visit['seqID']==seqid].sort_values(by=['dateTaken'])
    hist_pois = user_visit_seqid_dt['poiID'].unique()

    #### GET num2user / user2num
    num2user = dict()
    user2num = dict()
    for u in sorted(set( user_visit['userID'].unique() )):
        n=len(num2user)
        num2user[n] = u
        user2num[u] = n

    ### SKIPPING EMPTY FRAME
    if user_visit_seqid_dt.empty: return None

    bertlog.debug("--> LINE %d, TABLE user_visit_seqid_dt : \n%s", LINE(), user_visit_seqid_dt)
    time1 = user_visit_seqid_dt['dateTaken'].min()
    time2 = user_visit_seqid_dt['dateTaken'].max()
    secs = time2 - time1 + 1
    bertlog.debug("--> LINE %d, secs: %d", LINE(), secs)

    ### ITERNINARY MUST BE AT LEAST 15 mins
    #if secs < 15 * 60: continue
    bertlog.debug("--> LINE %d, mins: %d", LINE(), int(secs/60))

    assert(time1 and time2)

    numpois = len(poi2theme.keys())
    p0,pn = hist_pois[0], hist_pois[-1]
    t0,tn = poi2theme[p0], poi2theme[pn]

    bertlog.debug("--> LINE %d, p0: %d,  t0: %s", LINE(), p0, t0)
    bertlog.debug("--> LINE %d, pn: %d,  tn: %s", LINE(), pn, tn)
    bertlog.debug("--> LINE %d, %d secs,  nan:%s", LINE(), secs, str(math.isnan(secs)))

    assert(not math.isnan(secs))
    assert(bertmodel)
    ### estimate user / PREDICTION
    (predict_user_pid_theme, duration) = predict_user(bertmodel, pois, u, seqid, p0,pn,  t0, tn, secs, hist_pois)

    assert(predict_user_pid_theme[1])

    if duration:
        pass
    else:
        duration=0
    bertlog.debug("LINE %d,  predict_user(... u:'%s', seqid='%d', p0:%d, pn:%d, t0:%s, tn:%s, secs:%d, hist_pois)", LINE(),u,seqid, p0,pn,t0,tn, secs)
    bertlog.debug("line %d, [USER/POI/Theme] => (predict_user_pid_theme, duration) => %s, %s", \
                  LINE(),str(predict_user_pid_theme), duration)
    theme_seq = predict_user_pid_theme[2::3]
    bertlog.debug("line %d, themes : %s", LINE(), theme_seq)
    # print("LINE_1011, predict_user_pid_theme : ", str(predict_user_pid_theme))
    # print("LINE_1012,                        p0  : ", p0)
    # print("LINE_1012, p1 = predict_user_pid_theme[2]  : ", predict_user_pid_theme[2])
    # print("LINE_1012,                        pn  : ", pn)
    # print("LINE_1013, pn = predict_user_pid_theme[-3] : ", predict_user_pid_theme[-3])
    assert( str(predict_user_pid_theme[2]) == str(p0) )
    assert( str(predict_user_pid_theme[-3]) == str(pn) )
    #print("LINE_1035  predict_user_pid_theme : ", predict_user_pid_theme)

    predictstr = predict_user_pid_theme[2::5]
    #print("LINE_1038 predictstr : ", predictstr)
    #bertlog.debug("==> LINE %d, user : %s", LINE(), u)
    #bertlog.debug("==> LINE %d, seqid: %d", LINE(), seqid)
    #bertlog.debug("==> LINE %d, predictstr: %s", LINE(), predictstr)

    #bertlog.debug("==> LINE %d, predicted: %s", LINE(), predictstr)
    #bertlog.debug("==> LINE %d, duration: %s", LINE(), str(duration))
    #bertlog.debug("==> LINE %d, %d secs, nan?..%s", LINE(), secs, str(math.isnan(secs)))
    #bertlog.debug("==> LINE %d, PREDICTED SEQ : %s", LINE(), predictstr)
    #bertlog.debug("==> LINE %d, duration : %d", LINE(), int(duration))

    hist_pois = [ int(pid) for pid in hist_pois ]
    pred_pois = [ int(pid) for pid in predictstr ]
    bertlog.debug("==> LINE %d, TRYJECTORY : %s", LINE(), hist_pois)
    bertlog.debug("==> LINE %d, PREDICTION : %s", LINE(), pred_pois)

    ### CALC F1 SCORES
    # if predictstr and len(predictstr) >= 6:
    if len(hist_pois) and len(pred_pois):
        ### calc f1 score
        # pred_pois = predictstr[1::3]
        # pred_pois = [ int(pid)   for pid   in predictstr ]
        # hist_pois = [ int(poiid) for poiid in hist_pois ]
        assert(hist_pois[ 0]==pred_pois[ 0])
        assert(hist_pois[-1]==pred_pois[-1])

        bertlog.debug("LINE %d: userid:%s, seqid:%d, hist_pois -> %s", LINE(), u, seqid, hist_pois)
        bertlog.debug("LINE %d: userid:%s, seqid:%d, pred_pois -> %s", LINE(), u, seqid, pred_pois)

        p,r,f = f1_scores(hist_pois, pred_pois)

        bertlog.info("LINE %d, SCORES... u:%s, tryj:%s, hist_pois:%s ... p/r/f1 ( %f, %f, %f )", LINE(), u, str(hist_pois), str(hist_pois), 100*p, 100*r, 100*f)
        bertlog.info("LINE %d, SCORES... u:%s, tryj:%s, pred_pois:%s ... p/r/f1 ( %f, %f, %f )", LINE(), u, str(hist_pois), str(pred_pois), 100*p, 100*r, 100*f)
        #scores.append((100*p, 100*r, 100*f))
        return (100*p, 100*r, 100*f)
    else: assert(False)

def test_user_preference(city, epochs,  boot_duration, pois, userVisits, testVisits, bertmodel):
    bertlog.info("LINE %d -- test_user_preference( city='%s', epochs=%d,  boot_duration, pois,userVisits)", LINE(),city,epochs)
    bertlog.info("LINE %d -- bertmodel : %s", LINE(), bertmodel)
    assert(bertmodel)

    scores=[]
    bertlog.debug("LINE %d bertmodel, bertmodel: %s", LINE(), str(bertmodel))

    ### TEST USER
    test_users = sorted(testVisits['userID'].unique())

    for u in reversed(sorted(test_users)):
        user_visit=testVisits[ testVisits['userID'] == u ]
        seqids = sorted( user_visit["seqID"].unique() )
        bertlog.debug("--> LINE %d, START OF USERID => %s", LINE(), u)
        bertlog.info ("==> LINE %d, - TEST/EVAL USER: %s", LINE(), u)
        bertlog.debug('==> LINE %d, userid: %s, seqids: %s', LINE(), u, str(seqids))

        #bertlog.debug("==> LINE %d, START OF USERID => %s", LINE(), u)
        for seqid in seqids:
            assert(bertmodel)
            seq_f1scores = test_user_preference_seqid(city, epochs, u, seqid, boot_duration, pois, testVisits, bertmodel)
            assert(bertmodel)

            if seq_f1scores:
                #bertlog.debug("LINE %d, seq_scores : %s", LINE(), str(seq_f1scores))
                p,r,f = seq_f1scores
                scores.append( seq_f1scores )
            ### END FOR SEQID
        ### END FOR USER 
        bertlog.debug("--> LINE %d, END OF USERID => %s\n\n", LINE(), u)

    #bertlog.debug("LINE %d, scores : => %s", LINE(), str(scores))
    #for i in range(len(scores)):
    #    print("LINE {}, scores [ {} ] => {}".format(LINE(), i, str(scores[i])))
    ### TEST ALL USERs
    # scores[ (precision,recall,f1),...]
    # return scores
    # eval_scores = scores[ (precision,recall,f1),...]
    bertlog.debug("line %d, eval_scores : count: %d f1/recall/precision score(s)", LINE(), len(scores))
    bertlog.debug("line %d, eval_scores : %s", LINE(), str(scores))

    for (p,r,f1) in scores: bertlog.debug( "line %d, tryjectory score: ( f1:%0.5f, recall:%0.5f, prec:%0.5f )", LINE(), f1, r, p )

    arr = dict()
    arr['LM']            = str(bertmodel)
    arr['City']          = str(city)
    arr['Epochs']        = str(epochs)
    arr['F1']            = np.mean([f1 for (_,_,f1) in scores])
    arr['F1_std']        = np.std( [f1 for (_,_,f1) in scores])
    arr['Recall']        = np.mean([ r for (_,r,_)  in scores])
    arr['Recall_std']    = np.std( [ r for (_,r,_)  in scores])
    arr['Precision']     = np.mean([ p for (p,_,_)  in scores])
    arr['Precision_std'] = np.std( [ p for (p,_,_)  in scores])
    return arr

def train_user_bert(epochs, model_str, subseqs, use_cuda=True):
    model_args = ClassificationArgs()
    assert(0)
    ### MAKE CLASSIFCATION
    usermodel = ClassificationModel(
        model_type=  LABEL_TYPE, \
        model_name=  MODEL_NAME, \
        num_labels=  NUM_LABELS, \
        use_cuda=    use_cuda,   \
        cuda_device= 2,          \
        args=        model_args)
    usermodel.train_model(train_df, no_deprecation_warning=True)
    return user2num, num2user, usermodel

def mean_pooling(model_output, attention_mask):
    import torch
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def reviews_sentiments(city, pois):
    #poiid_comments = read_reviews(city, pois)
    #for poiid in poiid_comments:
    #    comments = poiid_comments[poiid]
    #    print (comments)
    #    assert(0)

    import torch
    from transformers import pipeline
    import statistics

    sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

    ### LOAD COMMENTS
    reviews_path="Data/Review-{}.csv".format(city)
    city_reviews_df= pd.read_csv(reviews_path, na_values="None", sep=';', keep_default_na=False, dtype={'poiID':int, 'title':str, 'text':str})

    poiid_sentiment = dict()
    for pid in sorted(pois['poiID'].unique()):
        pidcomment=""
        city_reviews_poi=city_reviews_df[ city_reviews_df['poiID']==pid ]
        arr = city_reviews_poi['text'].array
        arr2=[]
        for a in arr:
            if a.endswith('.') or a.endswith('!'): pass
            else: a+='.'
            #print ( " --> ", a)
            #print("POI-{} : '{}'".format(pid, a))
            arr2.append(a)
        sentences = arr2

        sentresults = sentiment_analysis(sentences)
        #[{'label': 'POSITIVE', 'score': 0.9989388585090637},
        # {'label': 'POSITIVE', 'score': 0.9983086585998535},
        # {'label': 'POSITIVE', 'score': 0.998842179775238}]
        sentiment_score=[]
        for d in sentresults:
            label = d['label']
            pve = 1 if label=='POSITIVE' else -1
            print(f">> {d['label']} ({pve} x {d['score']})")
            sentiment_score.append( pve * d['score'])
        #print(sentiment_score)
        #print("POI-{} -> score: {}".format(pid, np.mean(sentiment_score)))
        poiid_sentiment[ pid ] = np.mean(sentiment_score)
    return poiid_sentiment

def read_reviews(city, pois, summary_size=0):
    import torch
    ### LOAD COMMENTS
    reviews_path="Data/Review-{}.csv".format(city)
    city_reviews_df= pd.read_csv(reviews_path, na_values="None", sep=';', keep_default_na=False, dtype={'poiID':int, 'title':str, 'text':str})
    #print(city_reviews_df)
    poiid_comment = dict()

    for pid in sorted(pois['poiID'].unique()):

        pidcomment=""
        city_reviews_poi=city_reviews_df[ city_reviews_df['poiID']==pid ]

        arr = city_reviews_poi['text'].array
        arr2=[]
        for a in arr:
            if a.endswith('.') or a.endswith('!'): pass
            else: a+='.'
            #print ( " --> ", a)
            #print("POI-{} : '{}'".format(pid, a))
            arr2.append(a)

        sentences = arr2

        #Load AutoModel from huggingface model repository
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        print(sentences)
        if sentences:
            #Tokenize sentences
            encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
        
            #Compute token embeddings
            with torch.no_grad(): model_output = model(**encoded_input)
            
            #Perform pooling. In this case, mean pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            #print(sentence_embeddings)
            #print(sentence_embeddings.shape)
            # assign embedding to dict
            poiid_comment[ pid ] = sentence_embeddings

    #for p in poiid_comment:  print("LINE {}, POI {} -> {}".format(LINE(),p,poiid_comment[p]))
    # RETURN COMMENTS  for lengths [ min_length=1, max_length ]
    return poiid_comment

    # NOT USED FROM HERE
    # NOT USED FROM HERE
    # NOT USED FROM HERE

    # summodel = Summarizer()
    poiid_comment=dict()

    ### FULL REVIEWS
    if summary_size == 0:
        reviews_path="Data/Review-{}.csv".format(city)
        for pid in sorted(pois['poiID'].unique()):

            pidcomment=""
            city_reviews_poi= city_reviews_df[ city_reviews_df['poiID']==pid ]
            print("line 422: city_reviews_poi.. .")

            # combine 2 columns
            city_reviews_poi['comment'] = city_reviews_poi['title'].astype("string") + " " +  city_reviews_poi['text'].astype("string")

            print(f"LINE_429_city_reviews_poi['comment'] : \n{city_reviews_poi['comment']}")
            print(f"LINE_429_city_reviews_poi[''] : \n{city_reviews_poi}")
            for c in city_reviews_poi['comment'] : pidcomment += c

            ## pidcomment is long...
            # summary
            summtxt = summodel( pidcomment, min_length=25, max_length=50)
            poiid_comment[ pid ] = summtxt
    else:
        reviews_path="Data/Review-{}-summary-1-{}.csv".format(city,summary_size)
        bertlog.info("READING reviews from users: %s",reviews_path)
        ### LOAD COMMENTS

        reviews_df= pd.read_csv(reviews_path, na_values="None", sep=',', keep_default_na=False, dtype={'poiID':int, 'title':str, 'text':str})
        city_reviews_df = reviews_df[reviews_df['city']==city]
        print(city_reviews_df)
        for pid in sorted(pois['poiID'].unique()):
            pidcomment=""
            city_reviews_poi=city_reviews_df[ city_reviews_df['poiID']==pid ]
            # print("line 422: city_reviews_poi.. .")
            # combine 2 columns
            keywords = " ".join(city_reviews_poi['keywords'].astype("string"))
            poiid_comment[ pid ] = keywords

    for p in poiid_comment:  print("LINE {}, POI {} -> \"{}\"".format(LINE(),p, poiid_comment[p]))
    # RETURN COMMENTS  for lengths [ min_length=1, max_length ]
    return poiid_comment

def main(bert, city, epochs, USE_CUDA=True, args=None):    
    # read in from  spmf.sh /
    ### for city in ['Buda','Delh','Edin','Glas','Osak','Pert','Toro']
    #e = args.epochs
    (pois,  userVisits, evalVisits, testVisits) = load_dataset( city, DEBUG=1 )

    bertlog.info( "  [[[[[ RUN PARAMETERS : %s ]]]]]", str(args))
    bertlog.info( "  [[[[[ userVisits.shape : %s ]]]]]", str(userVisits.shape))
    bertlog.info( "  [[[[[ evalVisits.shape : %s ]]]]]", str(evalVisits.shape))
    bertlog.info( "  [[[[[ testVisits.shape : %s ]]]]]", str(testVisits.shape))

    reviews_sentiments( city, pois)
    assert(0)

    # combine all 3
    all_visits = pd.concat([userVisits, evalVisits, testVisits], axis=0)

    boot_duration = inferPOITimes2(pois, all_visits, alpha_pct=90)
    bertlog.info("boot_duration : %s", boot_duration)
    setting['bootstrap_duration'] = boot_duration
    assert(setting['bootstrap_duration'])

    summary_size=args.summary_size

    dist_dict = distance_dict(pois)
    theme2num, num2theme, poi2theme = get_themes_ids(pois)

    poi_reviews = read_reviews(city, pois, summary_size=0)
    setting["poi_comments"] = poi_reviews

    assert( 'poi_comments' in setting )
    assert( setting["poi_comments"] )

    ## TRAINING MAIN MODEL WITH ALL USERS
    ## TRAINING MAIN MODEL WITH ALL USERS
    ## TRAINING MAIN MODEL WITH ALL USERS
    bertmodel, train_df = bert_train_city(city, pois, userVisits, 1, summary_size=summary_size, model_str=bert, USE_CUDA=USE_CUDA)
    assert(bertmodel)
    for iter in range(epochs+1):
        if iter in [1,5,10,15,20,25,30,40,50,60,70,80,90,100]:
            # eval
            ## STEP-C EVALUATION
            print( "  PART [3] EVALUATION")
            print( "  PART [3] EVALUATION")
            print( "  PART [3] EVALUATION")
            assert(bertmodel)

            ### EVAL DATASET
            eval_scores = test_user_preference( city, epochs,  boot_duration, pois, userVisits, evalVisits, bertmodel)
            assert(0)

            assert(bertmodel)
            bertlog.info("\nEVAL_Scores, LM:\t%s\t city: \t%s\t, epochs: \t%d\t => precision: \t%0.5f\t %0.5f\t, recall: \t\t%0.5f\t %f\t, f1: \t%0.5f\t %0.5f", \
                         bert, city, iter,
                         eval_scores['Precision'], eval_scores['Precision_std'],
                         eval_scores['Recall'],    eval_scores['Recall_std'],
                         eval_scores['F1'],        eval_scores['F1_std'])

            ## STEP-D TESTING
            # TESTING
            print( "  PART [4] TESTING")
            print( "  PART [4] TESTING")
            print( "  PART [4] TESTING")
            bertlog.debug( "LINE %d, PART [4] main TESTINGN -->  bert_test_ city(city, pois, model, testVisits) --> %s".format(LINE(), str(bertmodel)))
            
            ### EVAL DATASET
            test_scores = test_user_preference( city, epochs,  boot_duration, pois, userVisits, testVisits, bertmodel)

            bertlog.info("\nTEST_Scores, LM:\t%s\t city: \t%s\t, epochs: \t%d\t => precision: \t%0.5f\t%0.5f\t, recall: \t\t%0.5f\t%f\t, f1: \t%0.5f\t%0.5f", \
                         bert, city, iter,
                         test_scores['Precision'], test_scores['Precision_std'],
                         test_scores['Recall'],    test_scores['Recall_std'],
                         test_scores['F1'],        test_scores['F1_std'])
        # retrain anothr iteration
        bertmodel.train_model(train_df, no_deprecation_warning=False)

    bertlog.info("LINE %d, END bertmodel : %s", LINE(), str(bertmodel))

    if False:
        try:
            output_dir = ("/var/tmp/output/output_{}_e{}_{}".format(city, epochs, bert))
            bertlog.info("... removing all output files: %s", output_dir)
            os.remove(output_dir)
        except FileNotFoundError as e:
            bertlog.warning("Cannot remove folder: '%s'", output_dir)

    #return (eval_scores,test_scores)
    quit(0)
    return


if __name__ == '__main__':
    ## default action for no arguments
    import random
    random.seed(1)
    #print( sys.argv )
    if len(sys.argv) <= 1:
        summary = main( bert="bert", city="Pert", epochs=1)
        print("Testing : \t", summary)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--city',         '-c', type=str, required=True)
        parser.add_argument('--epochs',       '-e', type=int, required=True)
        parser.add_argument('--model',        '-m', type=str, required=False, default='bert', help='LM module (bert/albert/roberta)')
        parser.add_argument('--summary_size', '-s', type=int, required=False, default=10,     help='max length of user reviews(POI)')
        parser.add_argument('--cuda', default=False, action=argparse.BooleanOptionalAction,   help='Using CUDA GPU')
        
        args = parser.parse_args()
        e = args.epochs + 1
        bertlog.info("RUN: %s", args)
        summary_eval,summary_test = main( bert=args.model, city=args.city, epochs=args.epochs, USE_CUDA=args.cuda, args=args)
        print("Evaluation : \t", summary_eval)
        print("Testing : \t", summary_test)
    quit(0)


