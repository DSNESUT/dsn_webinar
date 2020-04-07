import os
import json
import pickle
from datetime import datetime
from time import sleep

from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from werkzeug.utils import secure_filename

from pipeline import getProcessedData


app=Flask(__name__)

        
def check_correct(x):
    '''
    function to check if model prediction is correct ,
    using company expertsystem as the standard.
    '''
    model_pred=x['modelPrediction']
    company_pred=x['company_system']
    
    if company_pred and model_pred==1:
        model_status=1
    elif company_pred and model_pred==0:
        model_status=0
    elif not company_pred and model_pred==1:
        model_status=0
    elif not company_pred and model_pred==0:
        model_status=1
    
    return model_status


@app.route('/Model',methods=['POST','GET'])
def Model():
    
    if request.method=='POST':
        #collect data from request
        data=dict(request.get_json())
    
        #parse data as dataframe object
        data=pd.DataFrame(data)
        
        #load model
        model = pickle.load(open('model.pkl','rb'))

        #preprocess data
        processed_data=getProcessedData(data.drop('idFinding',axis=1))

        #make prediction
        prediction=model.predict_proba(processed_data)[:,1]
        
        #create dictionary of IdFinding and their corresponding prediction
        output=dict(zip(list(data.idFinding),list(prediction)))
        print(output)

        #return json response
        return jsonify(str(output))

    else:
        return jsonify('Access Denied')
        #Unauthorized access



sched = BackgroundScheduler(daemon=True)
sched.add_job(lambda : scoring_job(),trigger='interval',hours=24)
sched.start()
def batch_job():
    '''function to query the table for recently added Issues  ,
     score this Issues, check if the predictions are correct using company 
     expert system as the standard, push the results to postgre.
     '''

    #load the  model
    model = pickle.load(open('model.pkl','rb'))
    print('Starting Scoring Job')
    
    # create database connection
    DB_URL='db string'
    engine = create_engine(DB_URL)
    
    #query database for recently added findings (last 24hours)
    sql_query='''
        query string
         '''
    data_without_machine_tag= pd.read_sql_query(sql_query,con=engine')

    #query engine name
    machine_tag = pd.read_sql_query('query string',con=engine)
    machine_tag.columns=['idEngine','machine_tag']

    #merge data_without_machine_tag and machine_tag
    data_with_machine_tag=pd.merge(data_without_machine_tag,machine_tag,how='left',left_on='fkMachine',right_on='idEngine')#join the engine name to data
    data=data_with_machine_tag
    data.drop(['fkMachine','idMachine'],axis=1,inplace= True)

    if  data.shape[0]>50:
        start=datetime.now()

        #Extracting  company prediction from the new  data (labeling each row as either True or False)
        data['company_system']=data.apply(lambda x:True if x['fkIssues']!=None else False,axis=1)
        
        #passing the Issues through the Data Machineering processing pipeline  
        processed_Issues = getProcessedData(data)
        
        
        #scoring the processed Issues
        score=model.predict(processed_Issues)
        
        #insert the model predictions into output data
        data.insert(11,'modelPrediction',score,True)
        
        #check if models prediction was correct
        models_correct=pd.DataFrame({'status':data.apply(lambda x : check_correct(x),axis=1)})
        data.insert(11,'status',models_correct,True)

        #add model version and drop unwanted data from output data
        data.drop(['StringProperties','description','type'],axis=1,inplace=True)
        data.insert(2,'modelVersion',[model_version]*len(data),True)
        

        #push results to  table
        finish=datetime.now()
        print('Job Status: {} predictions Completed Succesfully in : {}  seconds'.format(data.shape[0],finish-start) )
        start=datetime.now()

        ##comment out the line below if you do not want to push to modelscores table
        data.to_sql('table',engine,if_exists='append',index=False,method='multi',chunksize=1000)
        print('output data:',data.columns)
        print(data.info())
        finish=datetime.now()
        
        print('Job Status: {} Uploads Completed Succesfully in : {}  seconds'.format(data.shape[0],finish-start) )
        atexit.register(lambda: sched.shutdown(wait=False))
    else:
        print('Data from previous day was less than 50')
        pass
    
# Shutdown your cron thread if the web process is stopped
atexit.register(lambda: sched.shutdown(wait=False))

if __name__ == "__main__":
    app.run(debug=False,use_reloader=False)