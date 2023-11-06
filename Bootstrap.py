import math
import common
from   common import LINE
from config import setting,log,bootlog
import statistics
from statistics import mean,median,pstdev,pvariance,stdev,variance

import random,time,os,sys,gc,collections,itertools
#import numpy as np
#import matplotlib.pyplot as plt
#from tqdm import tqdm

#import pandas as pd
#from   numpy   import diag
from   pathlib import Path
from   scipy   import linalg
import logging,traceback

### HELPER FUNCTIONS
def sorted_by_value(xdict):
  import operator
  sorted_x = sorted(xdict.items(), key=operator.itemgetter(1))
  return sorted_x

def timestring (unixtime):
  from datetime import datetime
  ts = int(unixtime)
  s=datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
  return s

def time_travel(distance):
  # distance: in  metres
  # return num of hours
  speed = 60.0 ## km/h
  return (distance / 1000 / speed)

def json_similar_list(list):
  data_set = {"POI_Prob": []}
  #data_set = {}
  for (poi,prob) in list:
    data_set["POI_Prob"].append( {"POI":poi, "Prob":prob} )
    #data_set.append( {"POI":poi, "Prob":prob} )
  return data_set["POI_Prob"]

def get_distance(gps1, gps2, method="manhattan") :
  import math
  def haversine(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))

  [lat1,lng1] = gps1
  [lat2,lng2] = gps2
  if method=="manhattan":
    return ((haversine([lat1,lng1],[lat1,lng2])) + (haversine([lat1,lng2],[lat2,lng2])))
  else:
    return (haversine([lat1,lng1],[lat2,lng2]))

def get_distance_matrix(data):
  poi_distances = dict()
  for i in range(0,len(data)):
    id_i = data.iloc[i]['poiID']
    i_lat = data.iloc[i]['lat']
    i_lng = data.iloc[i]['long']
    for j in range(i,len(data)):
      id_j = data.iloc[j]['poiID']
      j_lat = data.iloc[j]['lat']
      j_lng = data.iloc[j]['long']
      distance = get_distance([i_lat,i_lng], [j_lat,j_lng], "euclidean")
      poi_distances[ (id_i,id_j) ] = distance
      poi_distances[ (id_j,id_i) ] = distance
  return poi_distances

def bootstrapping(weight_pop, default=15*60, alpha_pct=90, B=5000, iteration=10000, seed=None):
  ## Part 1. Bootstrap and Confidence Interval
  import numpy as np
  import matplotlib as plt
  from   datetime import datetime

  if seed==None:
    random.seed(datetime.now())
    np.random.seed(int(time.time()))
  else:
    np.random.seed(int(seed))

  # remove samples with value 0 or 1
  weight_pop2=[]
  for w in weight_pop:
    if w >= default:
      weight_pop2.append(w)

  weight_pop = weight_pop2
  #print("\n\nweight_pop  ===> ", weight_pop)
  if len(weight_pop) == 0: return [default, default]


  #step 2
  weight_sample = np.random.choice(weight_pop, size=len(weight_pop)*5)
  sample_mean   = np.mean(weight_sample)  # sample mean
  sample_std    = np.std(weight_sample)   # sample std

  #step 3: bootstrap for 10,000 times
  boot_means = []
  for _ in range(iteration):
    # take a random sample each iteration
    boot_sample = np.random.choice(weight_sample,replace = True, size=B)
    # calculate the mean for each iteration
    boot_mean = np.mean(boot_sample)
    # append the mean to boot_means
    boot_means.append(boot_mean)
  # transform it into a numpy array for calculation
  boot_means_np = np.array(boot_means)

  # step 4: analysis and interpretation
  boot_means = np.mean(boot_means_np)# bootstrapped sample means

  #np.mean(weight_pop)# recall: true population mean
  boot_std = np.std(boot_means_np) # bootstrapped std
  # boot_std
  # alpha_pct=2.5  (100-alpha_pct)/2
  ci = np.percentile(boot_means_np, [ (100-alpha_pct)/2, 100-(100-alpha_pct)/2])
  # print( ci )
  # plt.hist(boot_means_np, alpha = 1)
  # plt.axvline(np.percentile(boot_means_np,2.5),color = 'red',linewidth=2)
  # plt.axvline(np.percentile(boot_means_np,97.5),color = 'red',linewidth=2)
  return ci

### FUNCTIONS
def inferPopularThemes(poiThemes,pois,userVisits):
  logging.info("LINE %d Bootstrap.py -- inferPopularThemes()",LINE())
  all_users = userVisits['userID'].unique()
  num_all_users = len(all_users)

  for theme in userVisits['poiTheme'].unique():
    for poiid in poiThemes:
      poithemes = userVisits[ userVisits['poiTheme']==theme]
      poivisits = poithemes[ poithemes['poiID'] == poiid ]

      if poivisits.empty:
        pass
      else:
        uniq_users = poivisits['userID'].unique()
        num_uniq_users = len(uniq_users)

        ## photos
        visits = userVisits[ userVisits['poiID']==poiid]
        #print(visits)

        users=visits['userID'].unique()
        user_visit_photocounts = []
        for user in users:
          user_visit = visits[ visits['userID']==user]
          user_visit_photocount = (len(user_visit))
          user_visit_photocounts.append(user_visit_photocount)
        avg_photo_ci = bootstrapping(user_visit_photocounts)
  quit(0)

def inferPOITimes(pois, userVisits, alpha_pct=90):
  logging.info("LINE %d Bootstrap.py -- inferPopularThemes()",LINE())
  all_users = userVisits['userID'].unique()
  num_all_users = len(all_users)
  poitimes = dict()

  for poiid in sorted(pois['poiID']):
    poivisits = userVisits[ userVisits['poiID'] == poiid ]
    if poivisits.empty:
      continue
    user_visit_second=[]

    users = (poivisits['userID'].unique())
    for userid in users:
      t = poivisits[ poivisits['userID']==userid ]

      # by sequence id
      seqids = (t['seqID'].unique())
      for seqid in seqids:
        tt = t[ t['seqID']==seqid ]
        timemin = tt['dateTaken'].min()
        timemax = tt['dateTaken'].max()
        #poivisits[ poivisits['dateTaken'] < timemin ]['dateTaken'].max()
        timediff = timemax - timemin + 1
        user_visit_second.append(timediff)

    avg_time_ci = bootstrapping(user_visit_second, alpha_pct=alpha_pct)
    logging.debug("POI_ID %2d -> time_visits:  %d%% C.I.: [ %0.3f %0.3f ]", poiid, alpha_pct, avg_time_ci[0], avg_time_ci[1])
    poitimes[poiid] = (avg_time_ci[0], avg_time_ci[1])

  ### default to 5 mins
  for poiid in pois['poiID']:
    if poiid in poitimes:
      pass
    elif not userVisits[ userVisits['poiID']==poiid ].empty:
      smalldb = userVisits[ userVisits['poiID']==poiid ]
      print('### 185.. poiID == ', poiid)
      quit(0)
      print(smalldb)
      ### intepolate times
      quit(0)
    else:
      log.warn("line %d, Bootstrap... no checkin records for POI id : %d", LINE(), poiid)
      #print(userVisits[ userVisits['poiID']==poiid ])
      ##get all seqids with poiid
      seqtable = userVisits[ userVisits['poiID']==poiid ]
      #print(seqtable)
      #print(seqtable['seqID'].unique())
      poitimes[poiid] = ( 5*60, 5*60 )

  return poitimes

def getUserPOITimes(seqid, userVisits, BEFORE=True, AFTER=True):
    #print(f"-- getUserPOITimes( seqid:{seqid}, userVisits)")
    #print(userVisits)
    if 'dateTaken' not in userVisits.columns: userVisits['dateTaken'] = userVisits['datetaken']
    userdf = userVisits[ userVisits['seqID'] == seqid]
    #poi_duration=[]
    poi_times=dict()

    for poiid in userdf['poiID'].unique():
      if poiid not in poi_times: poi_times[poiid]=[]
      #print(f"\n-----------------\nseqid:{seqid} poiid:{poiid}")
      #print(f"userdf => \n{userdf} ")
      user_poi_df = userdf[ userdf['poiID']==poiid ]
      #print("user_poi_df : \n", user_poi_df)
      user_poi_mintime = user_poi_df['dateTaken'].min()
      user_poi_maxtime = user_poi_df['dateTaken'].max()
      #if user_poi_mintime==user_poi_maxtime: user_poi_mintime = user_poi_maxtime-1

      if BEFORE:
        beforedf = userdf[userdf['dateTaken'] < user_poi_mintime]

        if beforedf.empty:
          beforetime = user_poi_mintime
          assert(beforetime)

          #poi_duration.append(user_poi_maxtime-beforetime)
          #poi_times[poiid].append(user_poi_maxtime-beforetime)
        else:
          beforetime = beforedf['dateTaken'].max()

        #poi_duration   .append(user_poi_maxtime-beforetime)
        if beforetime<user_poi_maxtime and user_poi_maxtime-beforetime<2*60*60:
          poi_times[poiid].append(user_poi_maxtime-beforetime)
          assert(0 != user_poi_maxtime-beforetime)
          #print(f"  C4 poiid:{poiid}  BEFORE TIME: -> {user_poi_maxtime-beforetime}")
          assert(beforetime <= user_poi_mintime )
          assert(              user_poi_mintime <= user_poi_maxtime )

      if AFTER:
        afterdf = userdf[userdf['dateTaken'] > user_poi_maxtime]
        if afterdf.empty:
          aftertime=user_poi_maxtime
        else:
          aftertime=afterdf['dateTaken'].min()

        if user_poi_mintime <= user_poi_maxtime   and \
           user_poi_mintime <  aftertime          and \
           aftertime - user_poi_mintime < 2*60*60 :
          # last poi must be within 2 hrs from all visit
          assert(user_poi_mintime <= user_poi_maxtime )
          assert(                    user_poi_maxtime <= aftertime)
          #print(f"  C5 poiid:{poiid}  AFTER TIME: -> {aftertime - user_poi_mintime}")
          #poi_duration.append(aftertime-user_poi_mintime)
          assert(0 != aftertime-user_poi_mintime)
          poi_times[poiid].append(aftertime-user_poi_mintime)

    #assert (len( poi_times[poiid] ) > 0)
    return (poi_times)

def getPOITimes(__poiid__, userVisits):
    #print("poiid => ", poiid)
    #poivisits = userVisits[ userVisits['poiID'] == poiid ]
    #assert(not poivisits.empty)

    userids=userVisits['userID'].unique()
    seqids=userVisits['seqID'].unique()
    #print("line 213: userids : ", userids)

    poitime_dict = dict()

    ### FOR EACH SEQ_ID
    for seqid in seqids:
      #print("\nLINE_276  seqid --> ", seqid)
      poitimes = getUserPOITimes(seqid, userVisits) 
      for poiid in poitimes:
        assert(0 not in poitimes[poiid])
        if poiid in poitime_dict:
          poitime_dict[poiid].extend(poitimes[poiid])
        else:
          poitime_dict[poiid] = poitimes[poiid]
    if bootlog:
        bootlog.debug("poitime_dict.keys() : %s", str(poitime_dict.keys()))
        for poiid in sorted(poitime_dict.keys()):
            bootlog.info(" -> poitime_dict[ %s ] ==> %s", poiid, str(poitime_dict[poiid]))
    return poitime_dict

def inferPOITimes2(pois, userVisits, alpha_pct=90):
  logging.info("LINE %d Bootstrap.py -- inferPOITimes2()",LINE())
  all_users = userVisits['userID'].unique()
  num_all_users = len(all_users)
  poitimes = dict()

  if True: #for poiid in sorted(pois['poiID']):
    poitimes = getPOITimes( 0, userVisits)
    ci_pid= dict()
    for poiid in sorted( poitimes.keys() ):
      avg_time_ci_poi = bootstrapping( poitimes[ poiid ], alpha_pct=alpha_pct)
      #print(f"poitimes[ {poiid} ] --->  {poitimes[ poiid ]}")
      #print(f"poitimes[ {poiid} ] ===>  {avg_time_ci_poi}")
      #print(f"poitimes[ {poiid} ] ===>  {avg_time_ci_poi[0]/60} .. {avg_time_ci_poi[1]/60} mins")
      ci_pid[poiid] = avg_time_ci_poi

    return ci_pid
    assert(0)

    avg_time_ci = bootstrapping(user_visit_second, alpha_pct=alpha_pct)
    logging.debug("POI_ID %2d -> time_visits:  %d%% C.I.: [ %0.3f %0.3f ]", poiid, alpha_pct, avg_time_ci[0], avg_time_ci[1])
    poitimes[poiid] = (avg_time_ci[0], avg_time_ci[1])
    print(f"line_271: poitimes[ {poiid} ] ==> { poitimes[poiid][0]/60 } .. {poitimes[poiid][1]/60} mins")
    assert(0)

  ### default to 5 mins
  for poiid in pois['poiID']:
    if poiid in poitimes:
      pass
    elif not userVisits[ userVisits['poiID']==poiid ].empty:
      smalldb = userVisits[ userVisits['poiID']==poiid ]
      print('### 185.. poiID == ', poiid)
      quit(0)
      print(smalldb)
      ### intepolate times
      quit(0)
    else:
      log.warn("line %d, Bootstrap... no checkin records for POI id : %d", LINE(), poiid)
      #print(userVisits[ userVisits['poiID']==poiid ])
      ##get all seqids with poiid
      seqtable = userVisits[ userVisits['poiID']==poiid ]
      #print(seqtable)
      #print(seqtable['seqID'].unique())
      poitimes[poiid] = ( 5*60, 5*60 )

  return poitimes

def infer2POIsTimes(pois,userVisits):
  logging.info("LINE %d -- infer2POIsTimes()",LINE())
  all_users = userVisits['userID'].unique()
  num_all_users = len(all_users)
  poitimes = dict()

  poi2poi_CI = dict()

  #for theme in userVisits['poiTheme'].unique():
  for poiid1 in sorted(pois['poiID']):
    for poiid2 in sorted(pois['poiID']):
      if poiid1==poiid2: continue

      poivisits1 = userVisits[ userVisits['poiID'] == poiid1 ]
      poivisits2 = userVisits[ userVisits['poiID'] == poiid2 ]
      users1 = poivisits1['userID'].unique()
      users2 = poivisits2['userID'].unique()

      commonUsers = set(users1).intersection(set(users2))

      timespans = []

      for user in commonUsers:
        table1 = userVisits[ userVisits['userID']==user ]
        table1 = table1[ table1['poiID']==poiid1 ]
        table2 = userVisits[ userVisits['userID']==user ]
        table2 = table2[ table2['poiID']==poiid2 ]

        if table1.shape[0]>2 and table2.shape[0]>2:
          combined = pd.concat([table1, table2])
          combined.sort_values(by=['dateTaken'])
          #print(combined)
          for seqid2 in table2['seqID'].unique():
            seqtable2 = table2[table2['seqID']==seqid2]
            timespan = seqtable2['dateTaken'].max() - seqtable2['dateTaken'].min() + 1
            if timespan > 0: timespans.append(timespan)
      #print( "pid1:{} pid2:[ {} ] TIMESPANs: {}".format(poiid1,poiid2,timespans) )
      if len(timespans)>4 :
        ci = bootstrapping(timespans, alpha_pct=90)
        print("{} -> {} CI: {}".format(poiid1,poiid2,ci))
        poi2poi_CI[(poiid1,poiid2)] = ci
      else:
        print("    estimate from single poi: {} -> {}".format(poiid1,poiid2))
  return poi2poi_CI

def main ():
    print("inferPOITimes2(..)")
    from poidata import load_files
    (pois, userVisits, testVisits, costProfat) = load_files( "Buda" )
    print("inferPOITimes2(...)  ")

    times = inferPOITimes2(pois, userVisits, alpha_pct=95)
    print("inferPOITimes2(....)  ")
    print(times)

if __name__ == "__main__":
  main()
