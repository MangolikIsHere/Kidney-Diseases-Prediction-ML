import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv(r'kidney_disease.csv')

df = df.replace(['?', '\t?', 'NA', 'na', ' '], np.nan)
df=df.drop(columns=['id', 'pcc', 'ba', 'bgr', 'sc','sg' ,'sod', 'pot', 'pcv', 'wc', 'cad', 'pe', 'ane'])
df = df.dropna()

label=LabelEncoder()
for col in ['rbc','pc','htn','appet','dm']:
    df[col]=label.fit_transform(df[col])

X=df.drop(columns=['classification'])
y = df['classification']

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42,stratify=y)

rf_class=RandomForestClassifier(n_estimators=100,random_state=42)
rf_class.fit(X_train,y_train)

y_pred=rf_class.predict(X_test)
score=accuracy_score(y_test,y_pred)
print(f"The models accuracy score is : {score*100:.2f}%")

l=[]
def check():
    while True:
        try:
            n=float(input())
            return n
        except Exception:
            print("Please enter a valid input :", end="")
l2=[
    "Enter your age",
    "Enter your Blood Pressure",
    "What's the level of Urine Albumin in your body ?(0-4)",
    "Enter your urine sugar level (0-5)",
    "How's your Red Blood Cells ? (normal: 1 , abnormal:0)",
    "How's your Puc Cells ? (normal: 1 , abnormal:0)",
    "What's your blood urea level ?(mg/dl)",
    "What's your Hemoglobin level (g/dL)",
    "Enter your Red Blood Cell Count",
    "Do you have high blood pressure? (yes: 1 , no: 0)",
    "Do you have Diabetis-(Melitus) ? (yes: 1 , no: 0)",
    "How's your apetite ? (Good: 0 , poor : 1)"
    ]
for i in range(len(l2)):
    print(l2[i]," : ",end="")
    l.append(check())
l=np.array(l).reshape(1,-1)
l=scaler.transform(l)
pred_score=rf_class.predict(l)[0]
conf=rf_class.predict_proba(l)[0]

if pred_score=="ckd":
    print(f"You have Chronic Kidney Disease...!\n Please visit a doctor soon..  \nConfidence : {conf[0]*100:.2f}%")
else:
    print(f"You don't have Chronic Kidney Disease...!\n Stay safe and healthy.. \nConfidence : {conf[1]*100:.2f}%")