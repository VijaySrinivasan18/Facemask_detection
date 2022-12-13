import pandas as pd 
import numpy as np 
import os 
from sklearn.model_selection import train_test_split
import shutil


# preprocessing mask_nomask data
df=pd.DataFrame(columns=["file_name","location","label"])
temp=pd.DataFrame([[0,0,0]],columns=["file_name","location","label"])
for root,folder,files in os.walk(r"dataset/mask_nomask"):
    for file in files:
        temp["file_name"]=file
        temp["location"]=os.path.join(root,file)
        if root.split('\\')[-1]=="with_mask":
            temp["label"]="masked"
        else:
            temp["label"]="no mask"
        df=pd.concat((df,temp),axis=0).reset_index(drop=True).sample(frac=1)
x=df.iloc[:,:-1]
y=df.iloc[:,[-1]]

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,stratify=y,random_state=1)

print(ytrain["label"].value_counts(normalize=True))
print(ytest["label"].value_counts(normalize=True))


#  Creating Image folders for Train and Test
for location,label in zip(xtrain["location"],ytrain["label"]):
    shutil.copy(location,f"dataset/mask_nomask/train/{label}")

for location,label in zip(xtest["location"],ytest["label"]):
    shutil.copy(location,f"dataset/mask_nomask/test/{label}")


# preprocessing human no-hum data

dfh=pd.DataFrame(columns=["file_name","location","label"])
temph=pd.DataFrame([[0,0,0]],columns=["file_name","location","label"])
for root,folder,files in os.walk(r"dataset\human_detection"):
    for file in files:
        temph["file_name"]=file
        temph["location"]=os.path.join(root,file)
        if root.split('\\')[-1]=="no_person":
            temph["label"]="no_person"
        else:
            temph["label"]="person"
        dfh=pd.concat((dfh,temph),axis=0).reset_index(drop=True).sample(frac=1)
xh=dfh.iloc[:,:-1]
yh=dfh.iloc[:,[-1]]

hn_xtrain,hn_xtest,hn_ytrain,hn_ytest=train_test_split(xh,yh,test_size=0.3,stratify=yh,random_state=1)

print(hn_ytrain["label"].value_counts(normalize=True))
print(hn_ytest["label"].value_counts(normalize=True))


#  Creating Image folders for Train and Test
for location,label in zip(hn_xtrain["location"],hn_ytrain["label"]):
    shutil.copy(location,f"dataset/human_detection/train/{label}/")

for location,label in zip(hn_xtest["location"],hn_ytest["label"]):
    shutil.copy(location,f"dataset/human_detection/test/{label}/")
    
