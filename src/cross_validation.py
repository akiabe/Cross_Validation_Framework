import pandas as pd
from sklearn import model_selection

"""
- binary classification
- multi class classification
- single column regression
- multi column regression
- hold-out (time series data, a lot of sample)
- multi label classification
"""

class CrossValidation:
    def __init__(
            self,
            df,
            target_cols,
            shuffle,
            problem_type="binary_classification",
            multilabel_delimiter=",",
            num_folds=5
    ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.multilabel_delimiter = multilabel_delimiter

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        self.dataframe["kfold"] = -1

    def split(self):
        if self.problem_type in ["binary_classification", "multiclass_classification"]:
            #if self.num_targets != 1:
            #    raise Exception("Invalid number of targets for this problem type")

            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, "kfold"] = fold

        elif self.problem_type in ["single_col_regression", "multi_col_regression"]:
            #if self.num_targets != 1 and self.problem_type == "single_col_regression":
            #    raise Exception("Invalid number of targets for this problem type")
            #if self.num_targets < 2 and self.problem_type == "multi_col_regression":
            #    raise Exception("Invalid number of targets for this problem type")

            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, "kfold"] = fold

        elif self.problem_type.startswith("holdout_"):
            #holdout_5, holdfout_10, holdfout_15...
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, "kfold"] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, "kfold"] = 1

        elif self.problem_type == "multilabel_classification":
            #if self.num_targets != 1:
            #    raise Exception("Invalid number of targets for this problem type")

            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)):
                self.dataframe.loc[val_idx, "kfold"] = fold

        else:
            raise Exception("Problem type not understood!")

        return self.dataframe

if __name__ == "__main__":
    #df = pd.read_csv("../input/train.csv")
    #cv = CrossValidation(df, target_cols=["target"], shuffle=True)

    #df = pd.read_csv("../input/train_reg.csv")
    #cv = CrossValidation(df, target_cols=["SalePrice"], problem_type="single_col_regression", shuffle=True)

    #df = pd.read_csv("../input/train.csv")
    #cv = CrossValidation(df, target_cols=["target"], problem_type="holdout_10", shuffle=False)

    df = pd.read_csv("../input/train_multilabel.csv")
    cv = CrossValidation(df, target_cols=["attribute_ids"],
                         problem_type="multilabel_classification",
                         multilabel_delimiter=" ", shuffle=True)

    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())