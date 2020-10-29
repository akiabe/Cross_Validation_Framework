import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics

def run(fold):
    df = pd.read_csv("../input/train_folds.csv")
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    xtrain = df_train.drop(["id", "target", "kfold"], axis=1)
    xvalid = df_valid.drop(["id", "target", "kfold"], axis=1)

    xvalid = xvalid[xtrain.columns]

    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    label_encoders = []
    for c in xtrain.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(xtrain[c].values.tolist() + xvalid[c].values.tolist())
        xtrain.loc[:, c] = lbl.transform(xtrain[c].values.tolist())
        xvalid.loc[:, c] = lbl.transform(xvalid[c].values.tolist())
        label_encoders.append((c, lbl))

    clf = ensemble.RandomForestClassifier(n_jobs=-1, verbose=2)
    clf.fit(xtrain, ytrain)
    pred = clf.predict_proba(xvalid)[:, 1]
    #auc = metrics.roc_auc_score(yvalid, preds)

    print(pred)

if __name__ == "__main__":
    for j in range(5):
        run(j)