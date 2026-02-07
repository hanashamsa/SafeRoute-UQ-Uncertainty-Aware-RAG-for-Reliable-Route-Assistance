import json, os
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

# load fused and labels
fused = [json.loads(l) for l in open("data/generations_fused.jsonl","r",encoding="utf-8").read().splitlines()]
labels = {json.loads(l)["query_id"]: json.loads(l)["label"] for l in open("data/labels.jsonl","r",encoding="utf-8").read().splitlines()}

X=[]
y=[]
for obj in fused:
    qid = obj["query_id"]
    if qid not in labels: continue
    X.append(obj["trust"])
    y.append(1 if labels[qid]=="correct" else 0)
X=np.array(X).reshape(-1,1)
y=np.array(y)

# split small if many data, else use all
if len(X) >= 8:
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.33, random_state=42)
else:
    Xtr,Xte,ytr,yte = X,X,y,y

# isotonic regression
iso = IsotonicRegression(out_of_bounds='clip').fit(Xtr.ravel(), ytr)
y_pred_iso = iso.predict(Xte.ravel())
brier_iso = brier_score_loss(yte, y_pred_iso)

# platt (logistic)
lr = LogisticRegression().fit(Xtr, ytr)
y_pred_lr = lr.predict_proba(Xte)[:,1]
brier_lr = brier_score_loss(yte, y_pred_lr)

# choose better
best = ("isotonic" if brier_iso <= brier_lr else "platt")
model = iso if best=="isotonic" else lr

# save model 
import pickle
with open("data/calibrator.pkl","wb") as f:
    pickle.dump((best, model), f)

print("Calibrator:", best, "brier_iso", brier_iso, "brier_lr", brier_lr)
