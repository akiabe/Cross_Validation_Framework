import pandas as pd

import lightgbm as lgb
from sklearn import preprocessing
from sklearn import metrics

def run(fold):
    df = pd.read_csv("../input/train_folds.csv")

    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    xtrain = df_train[features].values
    xvalid = df_valid[features].values

    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    clf = lgb.LGBMClassifier()
    clf.fit(xtrain, ytrain)
    pred = clf.predict_proba(xvalid)[:, 1]
    auc = metrics.roc_auc_score(yvalid, pred)

    print(f"fold={fold}, auc={auc}")
    #df_valid.loc[:, "lgb_pred"] = pred

    #return df_valid[["id", "target", "kfold", "lgb_pred"]]

if __name__ == "__main__":
    #dfs = []

    for j in range(5):
        run(j)
    #    temp_df = run(j)
    #    dfs.append(temp_df)

    #fin_valid_df = pd.concat(dfs)
    #print(fin_valid_df.shape)
    #fin_valid_df.to_csv("../model_preds/lgb.csv", index=False)