import sys
import logging
import pandas as pd
from config import setting,log


def corpus_text(text):
  ### only CAPTITAL LETTER, no a/e/i/o/u/ no symbol
  text = text.upper()  \
    .replace(",","")   \
    .replace(",","")   \
    .replace("[","")   \
    .replace("]","")   \
    .replace("(","")   \
    .replace(")","")   \
    .replace("_","_")   \
    .replace("-","_")
  return text

def poi_name_dict(pois):
  p2n=dict()
  n2p=dict()
  poiNames = getPOINames(pois)
  poiFullNames = getPOIFullNames(pois)
  for i in poiNames:
    ## i <-> poiNames[i]
    name = poiNames[i]
    p2n[i]    = name
    n2p[name] = i
  return (p2n,n2p)

def load_dataset(city, DEBUG):
  pd.set_option('mode.chained_assignment', None)
  EVALPCT=20
  TESTPCT=10
  POI_file = "Data/POI-{}.csv".format(city)
  pois = pd.read_csv(POI_file, sep=';', dtype={'poiID':int, 'poiName':str, 'lat':float, 'long':float, 'theme':str} )    
  pois['poiLongName'] = pois['poiName']
  pois['poiName'] = pois['poiLongName'].apply(lambda x: corpus_text(x))
  pois.style.set_properties(**{'text-align': 'left'})
  
  allVisits_file = "Data/userVisits-{}-allPOI.csv".format(city)

  allVisits = pd.read_csv( allVisits_file, sep=';',
                           dtype={'photoID':str, 'userID':str, 'dateTaken':int, 'poiID':str, 'poiTheme':str, 'poiFreq':str, 'seqID':int} )
  #                         dtype={'photoID':str, 'userID':str, 'dateTaken':int, 'poiID':int, 'poiTheme':str, 'poiFreq':str, 'seqID':int} )
  
  assert("userID" in allVisits.columns)
  logging.info("poidata line L44")

  if 'userid' in allVisits.columns and 'userID' not in allVisits.columns: allVisits['userID'] = allVisits['userid']
    
  assert("userID" in allVisits.columns)

  #allVisits['userID2'] = allVisits['userID'].str.split('@').str[0]
  #allVisits = allVisits.drop('userID', axis=1)
  allVisits = allVisits.rename(columns={'userID2': 'userID'})
  assert("userID" in allVisits.columns)
  
  train_index_list, train_list, index_list=[],[],[]  
  eval_index_list, eval_list, index_list=[],[],[]
  test_index_list, test_list, index_list=[],[],[]

  all_seqid_set = allVisits['seqID'].unique()
  finish_time = dict()
  finish_time_to_seqid = dict()

  ### shrink allVisits to only 3 or more
  for seqid in all_seqid_set:
    userVisits_seqid = allVisits[ allVisits['seqID'] == seqid ]
    if 'datetaken' in  userVisits_seqid.columns and  'dateTaken' not in  userVisits_seqid.columns :
      userVisits_seqid.rename(columns = {'datetaken':'dateTaken'}, inplace = True)
       
    poiids = userVisits_seqid['poiID'].unique()
    #print(poiids)
    if len(poiids) >= 3:
      #print("userVisits_seqid:",userVisits_seqid)
      time = userVisits_seqid['dateTaken'].max()
      finish_time[ seqid ] = time
      finish_time_to_seqid[ time ] = seqid
    else:
      # drop
      #print( "  drop seq_id : ", seqid)
      remove_index = allVisits[ allVisits['seqID'] == seqid ].index
      #print(f"seqid => {seqid} : before => {allVisits.shape}")
      allVisits.drop(remove_index, inplace = True)
      #print(f"seqid => {seqid} : after => {allVisits.shape}")

  ### shrink allVisits to only 3 or more
  #print( "REMOVED short tryhectories : ", allVisits.shape)
  #print("### finish_time:\n", finish_time)
  n=len(finish_time)

  all_finish_times = (sorted( finish_time.values()))
  #print(all_finish_times)

  lastindex=int(n * (100-EVALPCT-TESTPCT) / 100)
  lastindex2=int(n * (100-TESTPCT) / 100)

  #print(f"n:{n} lastindex= [ 0, {lastindex}, {lastindex2}")
  
  #print("all_finish_times : ", all_finish_times)
  #print(all_finish_times)
  max_training_time = max( all_finish_times[ 0 : lastindex ] ) or 0
  max_eval_time = max( all_finish_times[ 0 : lastindex2 ] )
  
  i = 0
  for t in finish_time:
    endtime=finish_time[t]
    seqid = finish_time_to_seqid[endtime]

    if i <= lastindex:
      train_list.append(seqid)
      #print(f"[TRAIN] #{i} seqid:{seqid} finish_time[ {t} ]  --> {endtime}")
    elif i <= lastindex2:
      eval_list.append(seqid)
      #print(f"[VALID] #{i} seqid:{seqid} finish_time[ {t} ]  --> {endtime}")
    else:
      test_list.append(seqid)
      #print(f"[TEST] #{i} seqid:{seqid} finish_time[ {t} ]  --> {endtime}")
    i = i+1

  train_df = allVisits.copy()
  eval_df = allVisits.copy()
  test_df = allVisits.copy()

  assert("userID" in allVisits.columns)

  for i, row in train_df.iterrows():
    if row['seqID'] not in train_list:
      train_df.drop(i, inplace=True)

  for i, row in eval_df.iterrows():
    if row['seqID'] not in eval_list:
      eval_df.drop(i, inplace=True)

  for i, row in test_df.iterrows():
    if row['seqID'] not in test_list:
      test_df.drop(i, inplace=True)

  if "CITY" in setting: setting["CITY"]=city
  
  assert("userID" in train_df.columns)
  assert("userID" in eval_df.columns)
  assert("userID" in test_df.columns)
  return pois, train_df, eval_df, test_df

def load_files(city, fold=1, subset=1, DEBUG=0):
  def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]

  IN_COLAB = 'google.colab' in sys.modules
  logging.info("LINE 37, IN_COLAB: {}".format(IN_COLAB))

  Themes = ["Amusement", "Beach", "Cultural", "Shopping", "Sport", "Structure"]
  POI_file=''

  if IN_COLAB:
    drive_path="/content"
    POI_file         = drive_path + "/Data/POI-"         + city+".csv"
    userVisits_file  = drive_path + "/Data/userVisits-"  + city+"-allPOI.csv"
    #costProfCat_file = drive_path + "/Data/costProfCat-" + city+"POI-all.csv"
  else:
    POI_file = "Data/POI-{}.csv".format(city)
    userVisits_file = "Data/userVisits-{}-allPOI.csv".format(city)
    #costProfCat_file = "Data/costProfCat-"+city+"POI-all.csv"

  ### ADD IN LONG NAMES
  #print("L53: reading poi file: ", POI_file)
  pois = pd.read_csv(POI_file, sep=';', dtype={'poiID':int, 'poiName':str, 'lat':float, 'long':float, 'theme':str} )
  #print(pois)
  pois['poiLongName'] = pois['poiName']
  pois.style.set_properties(**{'text-align': 'left'})
  pois['poiName'] = pois['poiLongName'].apply(lambda x: corpus_text(x))

  ### USE SHORT NAMES
  ### pois['poiName'] = pois['poiLongName'].apply(lambda x: corpus_text(x))
  ### pois.drop(columns=['poiName2'], axis=1)
  # costProfCat= pd.read_csv(costProfCat_file, sep=';', dtype={'photoID':int, 'userID':str, 'dateTaken':int, 'poiID':int, 'poiTheme':str, 'poiFreq':int, 'seqID':int} )
  userVisits = pd.read_csv(userVisits_file, sep=';', dtype={'photoID':int, 'userID':str, 'dateTaken':int, 'poiID':int, 'poiTheme':str, 'poiFreq':int, 'seqID':int} )
  assert(userVisits)
  
  ### remove seqid with only a few photos
  remove_index_list,remove_list,index_list=[],[],[]
  all_seqid_set = userVisits['seqID'].unique()
  finish_time=dict()

  ### bootstrapping
  from Bootstrap import inferPOITimes,infer2POIsTimes
  boottable=userVisits.copy()

  ### shrink userVisits to only 3 or more
  for seqid in all_seqid_set:
    userVisits_seqid = userVisits[ userVisits['seqID'] == seqid ]
    poiids = userVisits_seqid['poiID'].unique()
    #print(poiids)
    if len(poiids) < 4:
      t = userVisits[ userVisits['seqID'] != seqid ]
      userVisits = t
    else:
      # record finish time
      finish_time[seqid] = userVisits_seqid['dateTaken'].max()

  n=len(finish_time)
  all_finish_times = (sorted( finish_time.values()))
  lastindex=int(n * 80 / 100)
  max_training_time = max( all_finish_times[ 0 : lastindex ] )

  ### bootstrapping
  #from Bootstrap import inferPOITimes,infer2POIsTimes
  #boottable=userVisits.copy()
  boottable=boottable[ boottable['dateTaken'] <= max_training_time]
  #print(boottable)
  boot_times = inferPOITimes(pois,boottable)
  setting['bootstrap_duration'] = boot_times

  drop_seqids,keep_seqids=[],[]

  for seqid in all_seqid_set:
    userVisits_seqid_table = userVisits[ userVisits['seqID'] == seqid ]
    if userVisits_seqid_table.empty:
      drop_seqids.append(seqid)
    else:
      #print("==> userVisits_seqid_table : \n", userVisits_seqid_table)
      if userVisits_seqid_table['dateTaken'].max() <= max_training_time:
        keep_seqids.append(seqid)
      else:
        drop_seqids.append(seqid)

  userVisits_training = filter_rows_by_values(userVisits, "seqID", drop_seqids)
  userVisits_testing = filter_rows_by_values(userVisits, "seqID", keep_seqids)
  #print("{} userVisits.shape: {}".format(city,userVisits.shape))
  #print("{} userVisits_training.shape: {}".format(city, userVisits_training.shape))
  #print("{} userVisits_testing.shape:  {}".format(city, userVisits_training.shape))
  #logging.info("### poidata.py ### load_files LOADED")

  if "CITY" in setting: setting["CITY"]=city
  ## CROSS VALIDATION SET ?
  '''
  if "FOLDS_CROSS_VALIDATION" in setting and setting["FOLDS_CROSS_VALIDATION"]:
    logging.info("poidata: LINE 85, using cross validation.." )
    quit(-1)
    setting_FOLD = setting["FOLD"]
    setting_FOLDS = setting["FOLDS_CROSS_VALIDATION"]
    logging.info("  Crossing Valudation: %d / %d", setting["FOLD"], setting["FOLDS_CROSS_VALIDATION"] )

    users = ( userVisits["userID"].unique() )
    remove_users = []
    keep_users = []
    for i in range(len(users)):
      if setting_FOLD == i % setting_FOLDS:
        remove_users.append(users[i])
      else:
        keep_users.append(users[i])

    logging.debug("poidata.py:   dropped %d rows, from %d-FOLDS_CROSS_VALIDATION", len(remove_users), setting["FOLDS_CROSS_VALIDATION"])
    logging.debug("poidata.py: userVisits.shape : %s", str(userVisits.shape))

    testVisits = userVisits.copy()
    for ruser in keep_users:
      remove_index = testVisits[ testVisits['userID'] == ruser ].index
      testVisits.drop(remove_index, inplace = True)
    logging.debug("testVisits.shape : %s", str(testVisits.shape))

    for ruser in remove_users:
      remove_index = userVisits[ userVisits['userID'] == ruser ].index
      userVisits.drop(remove_index, inplace = True)
    logging.debug("userVisits.shape : %s", str(userVisits.shape))
    #print("RETURN: {} pois       : {}".format(city,pois.shape))
    #print("RETURN: {} userVisits : {}".format(city,userVisits_training.shape))
    #print("RETURN: {} testVisits : {}".format(city,userVisits_testing.shape))
  '''
  #return(pois, userVisits_training, userVisits_testing, costProfCat)
  return(pois, userVisits_training, userVisits_testing, None)

def getThemes(pois):
  sorted_themes = pois.sort_values("theme")
  return sorted_themes['theme'].unique()

def getAllUsers(userVisits):
  users = userVisits[['userID']].drop_duplicates(subset=['userID'])
  return users

def getUser2User(users,pois):
  user2userDF = users.join(users, lsuffix='_a', rsuffix='_b')
  #print(user2userDF.head())
  jr = user2userDF.shape[1]

  if False:
    d = []
    for i in range(jr):
      u1 = user2userDF['userID_a'][i]
      u2 = user2userDF['userID_b'][i]

      if u1 == u2:
        d.append(i)
        #print("3 user2userDF['userID_a'][i] : ", user2userDF['userID_a'][i])
    #print("### drop : ", d)
    user2userDF = user2userDF.drop(d)

  data = np.array(user2userDF)
  data2 = []
  for i in data.tolist():
    for p in pois["poiID"]:
      data2.append(  i + [p] )
      #print("  poiId : ", p)
  #print(data2)
  data3 = pd.DataFrame(columns=['userID1','userID2','poiID'],data=data2)
  return data3

def getPOIIDs(df):
  poiIDs=[]
  for i in df["poiID"]:
    poiIDs.append(i)
  return poiIDs

def getPOIThemes(df):
  poiIDs=dict()
  for index, row in df.iterrows():
    id = row['poiID']
    poiIDs[id] = row['theme']
  return poiIDs

def getPOINames(df):
  poiIDs=dict()
  for index, row in df.iterrows():
    id = row['poiID']
    name = row['poiName']
    poiIDs[id] = name
  return poiIDs

def getPOIFullNames(df):
  poiIDs=dict()
  for index, row in df.iterrows():
    id = row['poiID']
    poiIDs[id] = row['poiLongName']
  return poiIDs

def getPOILongNames(df):
  poiIDs=dict()
  for index, row in df.iterrows():
    id = row['poiID']
    name = row['poiLongName']
    poiIDs[id] = name
  return poiIDs

def unittest():
  import numpy as np
  if True:
    # check users
    for city in ["Buda", "Delh", "Edin", "Glas", "Osak", "Pert", "Toro", "Vien" ]:
      print("READING CSV FILES: ", city)
      pois, userVisits, evalVisits, testVisits = load_dataset(city, DEBUG=1)
      #print("{} userVisits: timestamps [ {}, {} ] ".format(city, str(userVisits['dateTaken'].min()), str(userVisits['dateTaken'].max())))
      train_users = set(sorted(userVisits['userID'].unique()))
      eval_users = set(sorted(evalVisits['userID'].unique()))
      test_users = set(sorted(testVisits['userID'].unique()))
      print("  users in train. set: ", len(train_users))

      print("  users in eval.  set: ", len( eval_users ))
      train_and_eval = eval_users.intersection(train_users )
      print("        in train. set: ", len( train_and_eval ))
      print("    not in train. set: ", len( eval_users - train_users ))

      print("  users in test.  set: ", len( test_users ))
      train_and_test = test_users.intersection(train_users)
      print("        in train. set: ", len( train_and_test ))
      print("    not in train: set: ", len( test_users - train_users ))
      print("")
  #assert(0)

  ### POIs / Tryjectorys per city
  for city in ["Buda", "Delh", "Edin", "Glas", "Osak", "Pert", 'Melb', "Toro", "Vien" ]:
    #pois, userVisits, testVisits, costProfCat = load_files(city, fold=1, subset=1, DEBUG=1)
    pois, userVisits, evalVisits, testVisits = load_dataset(city, DEBUG=1)


    print("---------------------\ncity: {}".format(city))
    print("train unique users: ", userVisits['userID'].unique().shape)
    print("train        photos: ", userVisits['userID'].shape)
    print("train avg    photos: ", (userVisits['userID'].shape[0] / userVisits['userID'].unique().shape[0]) )
    print("evalVisita unique users: ", evalVisits['userID'].unique().shape)
    print("testVisits unique users: ", testVisits['userID'].unique().shape)
    #print("testing        photos: ", testVisits['userID'].shape)

    num_pois_vec = []
    num_photos_vec = []
    for seqid in userVisits['seqID'].unique():
      #print("  seqid : ", seqid)
      seqid_table = userVisits[ userVisits['seqID']==seqid ]
      #print(seqid_table)
      numpois = len( seqid_table['poiID'].unique())
      numcheckins = seqid_table['poiID'].count()

      num_pois_vec.append( numpois )
      num_photos_vec.append( numcheckins )

    print(f"{city} (train) num_test_tryj :         {len(num_pois_vec)}")
    print(f"{city} (train) num_pois_vec / tryj :   {np.mean(num_pois_vec)}")
    print(f"{city} (train) num_photos_vec / tryj : {np.mean(num_photos_vec) / np.mean(num_pois_vec)}" )

    num_pois_vec = []
    num_photos_vec = []
    for seqid in evalVisits['seqID'].unique():
      seqid_table = evalVisits[ evalVisits['seqID']==seqid ]
      numpois = len( seqid_table['poiID'].unique())
      numcheckins = seqid_table['poiID'].count()
      num_pois_vec.append( numpois )
      num_photos_vec.append( numcheckins )

    #print(f"{city} (eval) num_tryj :              {len(num_pois_vec)}")
    print(f"{city} (eval) num_test_tryj :         {len(num_pois_vec)}")
    print(f"{city} (eval) num_pois_vec / tryj :   {np.mean(num_pois_vec)}")
    print(f"{city} (eval) num_photos_vec / tryj : {np.mean(num_photos_vec) / np.mean(num_pois_vec)}" )

    
    num_pois_vec = []
    num_photos_vec = []
    for seqid in testVisits['seqID'].unique():
      seqid_table = testVisits[ testVisits['seqID']==seqid ]
      numpois = len( seqid_table['poiID'].unique())
      numcheckins = seqid_table['poiID'].count()
      num_pois_vec.append( numpois )
      num_photos_vec.append( numcheckins )

    #print(f"{city} (test) num_tryj :              {len(num_pois_vec)}")
    print(f"{city} (test) num_test_tryj :         {len(num_pois_vec)}")
    print(f"{city} (test) num_pois_vec / tryj :   {np.mean(num_pois_vec)}")
    print(f"{city} (test) num_photos_vec / tryj : {np.mean(num_photos_vec) / np.mean(num_pois_vec)}" )
    
def latex_table():
  import numpy as np
  allcities = sorted(["Buda", "Delh", "Edin", "Glas", "Osak", "Pert", 'Melb', "Toro", "Vien" ])
  all_excaped_cities = [ "\\" + c for c in allcities]
  ### POIs / Tryjectorys per city
  print("\\begin{tabular}")
  colc = ''.join( [ 'c' for c in allcities ] )
  print("  {{l{} }}".format(colc))
  print("  \hline\hline".format(colc))
  print("    City & {}".format( " & ".join( all_excaped_cities ) ))
  print("         \\\\")
  print("         \\hline\\hline")


  ### No. of \POIs
  #         {No. of~\POIs} & 39 & 26 & 29 & 29 & 242 & 28 & 25 & 30 & 29 \\
  #           \hline
  numpois = dict()
  #dict = {'Pert': 111, "Toro":222, "Buda":111, "Delh":222, "Edin":333, "Glas":444, "Osak":555, "Pert":666, 'Melb':777, "Toro":888, "Vien":999 } 
  all_tryjs      = dict()
  train_tryjs    = dict()
  eval_tryjs     = dict()
  test_tryjs     = dict()
  all_checkins   = dict()
  train_checkins = dict()
  eval_checkins  = dict()
  test_checkins  = dict()
  for city in allcities:
    pois, userVisits, evalVisits, testVisits = load_dataset(city, DEBUG=1)
    assert("userID" in userVisits.columns)
    assert("userID" in evalVisits.columns)
    assert("userID" in testVisits.columns)


    userVisits.to_csv("Data/train_{}.csv".format(city), sep=';')
    evalVisits.to_csv("Data/valid_{}.csv".format(city), sep=';')
    testVisits.to_csv("Data/test_{}.csv".format(city), sep=';')

    numpois[city] = int(pois['poiID'].count())
    train_tryjs[city] = int( len(userVisits['seqID'].unique()) )
    eval_tryjs[city] = int( len(evalVisits['seqID'].unique()) )
    test_tryjs[city] = int( len(testVisits['seqID'].unique()) )

    train_checkins[city] = userVisits['photoID'].count()
    eval_checkins[city] = evalVisits['photoID'].count()
    test_checkins[city] = testVisits['photoID'].count()
    all_checkins[city] = train_checkins[city] + eval_checkins[city] + test_checkins[city]
    
  strnumpois= [ str(numpois[c]) for c in  allcities]
  all_strnumpois = " & ".join( strnumpois )


  for c in allcities:
    all_tryjs[c] = train_tryjs[c] + eval_tryjs[c] + test_tryjs[c]
  
  print("  No. of~POIs & {} \\\\".format(  " & ".join(strnumpois)))
  print("  \\hline")

  all_tryjs_arr=[]
  train_tryjs_arr=[]
  eval_tryjs_arr=[]
  test_tryjs_arr=[]

  
  for c in allcities:
    train_tryjs_arr.append( str(train_tryjs[c]) )
    eval_tryjs_arr.append( str(eval_tryjs[c]) )
    test_tryjs_arr.append( str(test_tryjs[c]) )
  
  for k in all_tryjs.keys():
    all_tryjs_arr.append( str(all_tryjs[k]) )
    #print( "|||  all_tryjs[{}] => {}".format(k,all_tryjs[k]))
  
  print("  No. of~tryjectories & {} \\\\".format(" & ".join(  all_tryjs_arr)))
  print("  .. for training~~~ & {} \\\\".format( " & ".join(train_tryjs_arr)))
  print("  .. for evalidation & {} \\\\".format( " & ".join( eval_tryjs_arr)))
  print("  .. for testing     & {} \\\\".format( " & ".join( test_tryjs_arr)))

  print("  \\hline")

  ## No. of check-ins~~~~

  all_checkins_arr=[]
  train_checkins_arr=[]
  eval_checkins_arr=[]
  test_checkins_arr=[]
  for c in allcities:
    #train_checkins[city] = userVisits['photoID'].count()
    #eval_checkins[city] = evalVisits['photoID'].count()
    #test_checkins[city] = testVisits['photoID'].count()
    #all_checkins[city] = train_checkins[city] + eval_checkins[city] + test_checkins[city]

    train_checkins_arr.append( str(int(train_checkins[c])) )
    eval_checkins_arr.append(  str(int(eval_checkins[c]))  )
    test_checkins_arr.append(  str(int(test_checkins[c]))  )

  print("  No. of check-ins~~~~ & {} \\\\".format(" & ".join(  all_checkins_arr)))
  print("  training~~~~ & {} \\\\        ".format(" & ".join(train_checkins_arr)))
  print("  validation~~~~ & {} \\\\"      .format(" & ".join( eval_checkins_arr)))
  print("  testing~~~~ & {} \\\\"         .format(" & ".join( test_checkins_arr)))

  # train_checkins

  
  for city in allcities:
    # {lccccccccc }
    #   \hline\hline
    #     City & \Buda & \Delh & \Edin & \Glas & Melbourne & \Osak &  \Pert & \Toro & \Vien
    #          \\
    #print("  \\begin{tabular}")


    print("---------------------\ncity: {}".format(city))
    print("train unique users: ", userVisits['userID'].unique().shape)
    print("train        photos: ", userVisits['userID'].shape)
    print("train avg    photos: ", (userVisits['userID'].shape[0] / userVisits['userID'].unique().shape[0]) )
    print("evalVisita unique users: ", evalVisits['userID'].unique().shape)
    print("testVisits unique users: ", testVisits['userID'].unique().shape)
    #print("testing        photos: ", testVisits['userID'].shape)

    num_pois_vec = []
    num_photos_vec = []
    for seqid in userVisits['seqID'].unique():
      #print("  seqid : ", seqid)
      seqid_table = userVisits[ userVisits['seqID']==seqid ]
      #print(seqid_table)
      numpois = len( seqid_table['poiID'].unique())
      numcheckins = seqid_table['poiID'].count()

      num_pois_vec.append( numpois )
      num_photos_vec.append( numcheckins )

    print(f"{city} (train) num_test_tryj :         {len(num_pois_vec)}")
    print(f"{city} (train) num_pois_vec / tryj :   {np.mean(num_pois_vec)}")
    print(f"{city} (train) num_photos_vec / tryj : {np.mean(num_photos_vec) / np.mean(num_pois_vec)}" )

    num_pois_vec = []
    num_photos_vec = []
    for seqid in evalVisits['seqID'].unique():
      seqid_table = evalVisits[ evalVisits['seqID']==seqid ]
      numpois = len( seqid_table['poiID'].unique())
      numcheckins = seqid_table['poiID'].count()
      num_pois_vec.append( numpois )
      num_photos_vec.append( numcheckins )

    #print(f"{city} (eval) num_tryj :              {len(num_pois_vec)}")
    print(f"{city} (eval) num_test_tryj :         {len(num_pois_vec)}")
    print(f"{city} (eval) num_pois_vec / tryj :   {np.mean(num_pois_vec)}")
    print(f"{city} (eval) num_photos_vec / tryj : {np.mean(num_photos_vec) / np.mean(num_pois_vec)}" )

    
    num_pois_vec = []
    num_photos_vec = []
    for seqid in testVisits['seqID'].unique():
      seqid_table = testVisits[ testVisits['seqID']==seqid ]
      numpois = len( seqid_table['poiID'].unique())
      numcheckins = seqid_table['poiID'].count()
      num_pois_vec.append( numpois )
      num_photos_vec.append( numcheckins )

    #print(f"{city} (test) num_tryj :              {len(num_pois_vec)}")
    print(f"{city} (test) num_test_tryj :         {len(num_pois_vec)}")
    print(f"{city} (test) num_pois_vec / tryj :   {np.mean(num_pois_vec)}")
    print(f"{city} (test) num_photos_vec / tryj : {np.mean(num_photos_vec) / np.mean(num_pois_vec)}" )
  print("\n\end{tabular}\n\n")
  
def users_stat():
  def df_stat(city, usercity_df, users):
    # checking users with known countrys
    #usercity_df
    user2city=dict()
    for u in users:
      '''
        UserID        oinDate                                         Occupation                           Hometown current_city   country
0     10002536      July 2007                                                                             Innsbruck    Satu Mare   Romania
1     10007579  December 2006                Web Developer and Corporate Trainer                      Dubai, Mumbai      Alberta    Canada
      '''

      userdf2 = usercity_df[usercity_df['UserID']==u]
      #print(" user: {} in usercity_df ? => {}".format( u, not(userdf2.empty)))
      if (not(userdf2.empty)):
        assert(userdf2.shape == (1, 6))
        for index, row in userdf2.iterrows():
          UserID      = row['UserID']
          JoinDate    = row['JoinDate']
          Occupation  = row['Occupation']
          Hometown    = row['Hometown']
          current_city= row['current_city']
          country     = row['country']
          
          user_homecity = " ".join([ country, current_city ]).strip()
          if user_homecity: user2city[ u ] = user_homecity
          #print(">>> user: {}  ==> '{}'".format( u, user_homecity))
          break
      #else:
      #  print(">>> user: {}  --> NONE".format( u ))
    keys = list(user2city.keys())
    num_cityusers=0
    for ui in range(len(keys)):
      u = keys[ui]
      #print(">>> i: {}, user: {}  --> '{}'".format( ui, u, user2city[u] ))
      num_cityusers = num_cityusers + 1
    #print("city: {} >>--> total users with location : {} / {}".format( city, num_cityusers, len(users) ))
    # func: df_stat
    return num_cityusers

  ### READ USER INFO
  usercity_df = pd.read_csv("Data/user_hometown.csv", \
                            sep=';',
                            keep_default_na=False,
                            na_values='_', 
                            dtype={'UserID':str, 'JoinDate':str, 'Occupation':str, 'Hometown':str, 'current_city':str, 'country':str} )

  for city in ["Buda", "Delh", "Edin", "Glas", "Melb", "Osak", "Pert", "Toro", "Vien" ]:
    ### READ CTY INFO
    print("READING CSV FILES: ", city)
    pois, userVisits, evalVisits, testVisits = load_dataset(city, DEBUG=1)
    #print("{} userVisits: timestamps [ {}, {} ] ".format(city, str(userVisits['dateTaken'].min()), str(userVisits['dateTaken'].max())))
    train_users = set(sorted(userVisits['userID'].unique()))
    eval_users = set(sorted(evalVisits['userID'].unique()))
    test_users = set(sorted(testVisits['userID'].unique()))

    users_with_info = df_stat(city, usercity_df, train_users)
    print("  users with info (train.) {} / {} = {} %%".format( users_with_info, len(train_users), 100*users_with_info/len(train_users)))
    users_with_info = df_stat(city, usercity_df, eval_users)
    print("  users with info (eval.) {} / {} = {} %%".format( users_with_info, len(eval_users), 100*users_with_info/len(eval_users)))
    users_with_info = df_stat(city, usercity_df, test_users)
    print("  users with info (test.) {} / {} = {} %%".format( users_with_info, len(test_users), 100*users_with_info/len(test_users)))



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


if __name__ == "__main__":
  #unittest()
  #latex_table()
  load_dataset('Florence',DEBUG=1)
  users_stat()
