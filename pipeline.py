#code for the complete pipeline for the Python,Elixir,Go and Solidty language

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import pickle
from sklearn.preprocessing import PolynomialFeatures

def Serious_uniform(x):
  #this corrects the spelling inconsistensies in the Serious column
  try:
    if x !=None:
      return str(x)
    elif x==None:
      return 'None'
  except:
    return 'None'

##creating a custom transformer to clean the Serious column
class SeriousCleaner(BaseEstimator,TransformerMixin):
  def __init__(self):
    pass
  
  
  def fit(self,df,y=None):
    self.dataframe=df
    self.dataframe.fkSerious=self.dataframe.fkSerious.apply(lambda x:Serious_uniform(x))
    
    return self
    
  def transform(self,df):
    
    print ('Stage 1 (SeriousCleaner)--------completed. Output data:' ,self.dataframe.shape[0],' rows',self.dataframe.shape[1],' columns',)
    return self.dataframe

def get_avg(row):
  #gets the average word lenght of a a string
    arr=[]
    for word in list(row.split()):
      arr.append(len(word))
    if len(arr) > 0:
      return np.mean(arr)
    else:
      return 0

class Descriptionstringproperties(BaseEstimator,TransformerMixin):
  #Transformer for the description and line content
  def __init__(self):
    pass

  def fit(self,df,y=None):
    return self
    
  def transform(self,df):
    df_copy=df
    stringproperties_wordcount = df_copy.stringproperties.apply(lambda x: len(str(x).split()) if x!=None  else 0)
    description_wordcount = df_copy.description.apply(lambda x: len(str(x).split()) if x!=None  else 0)
    stringproperties_word_avg=df_copy.stringproperties.apply(lambda x: get_avg(str(x)) if x!=None  else 0)
    description_word_avg=df_copy.description.apply(lambda x: get_avg(str(x)) if x!=None  else 0)
    dataframe=pd.DataFrame({'stringproperties_wordcount':stringproperties_wordcount,
                                 'description_wordcount':description_wordcount,
                                 'stringproperties_word_avg':stringproperties_word_avg,
                                 'description_word_avg':description_word_avg})
    df_copy=pd.concat([df_copy,dataframe],axis=1)
    df_copy.drop(['stringproperties','description' ],axis=1,inplace=True)
    print ('Stage 2 (Descriptionstringproperties)--------completed. Output data:' ,df_copy.shape[0],' rows',df_copy.shape[1],' columns',)
    return df_copy


def check_feature(df):
  model_feature_names='feature array'
  
  #function to check all the features used to train the model are present in new data 
  col_checker=model_feature_names

  dataframe=pd.DataFrame({'filler_column':df.description_word_avg})
  
  for col in col_checker:
    try:
      dataframe=pd.concat([dataframe,df[col]],axis=1)
    except KeyError:
      #column not in dataset
      filler=pd.DataFrame({col:[0]*len(df)})
      filler.index=dataframe.index
      dataframe=pd.concat([dataframe,filler],axis=1)
  dataframe.drop('filler_column',axis=1,inplace=True)
  
  return dataframe
  

class FeatureComplete(BaseEstimator,TransformerMixin):
  def __init__(self):
    pass
  def fit(self,X,y=None):
    self.dataframe=check_feature(X)
    return self
  def transform(self,X):
    X_copy=self.dataframe
    print('Last Stage (FeatureComplete)------------------------------completed. Output data:' ,' rows=',X_copy.shape[0],' columns=',X_copy.shape[1])
    return X_copy


class PolyFit(BaseEstimator,TransformerMixin):
  def __init__(self):
    pass
  def fit(self,df,y=None):
    
    return self
  def transform(self,df):
    df_copy=df.copy()
    poly=PolynomialFeatures(2)
    
    poly.fit(df[['stringproperties_wordcount','description_wordcount', 'stringproperties_word_avg','description_word_avg']])
    dataframe=pd.DataFrame(poly.transform(df[['stringproperties_wordcount',
                                                'description_wordcount', 'stringproperties_word_avg','description_word_avg']]))
    for name in list(dataframe.columns):
      df_copy[name]=dataframe[name].values
    print ('Stage 3 (PolyFeatures)--------completed. Output data:' ,df_copy.shape[0],' rows',df_copy.shape[1],' columns',)
    
    return df_copy

class OneHot(BaseEstimator,TransformerMixin):
  def __init__(self):
    pass
  def fit (self,df,y=None):
    return self
  def transform(self,df):
    df_copy=df.copy()
    dataframe=pd.get_dummies(df_copy[['fkSerious','language','type']])
    dataframe.index=df_copy.index
    dataframe=pd.concat([df_copy,dataframe],axis=1)
    print ('Stage 4 (OneHot Encoding)--------completed. Output data:' ,dataframe.shape[0],' rows',dataframe.shape[1],' columns')
    return dataframe


def getProcessedData(df):
  pipe=Pipeline([
  ('Serious_cleaner',SeriousCleaner()),
  ('Nlp',Descriptionstringproperties()),
  ('polyfit',PolyFit()),
  ('OneHot',OneHot()),
  ('features',FeatureComplete())
  ])
  return pipe.fit_transform(df)
    
    


