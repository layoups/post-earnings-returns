from typing import Dict, List, Any

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from hyperopt import tpe, hp, fmin, STATUS_OK, Trials, space_eval
from sklearn.datasets import fetch_openml

import pycatch22 as tsfe

class TSFEA:
    
    def __init__(
        self,
        models: Dict[str, BaseEstimator] = {},
        hyparam_space: Dict[str, Dict[str, Any]] = {}
    ) -> None:
        self.models = models
        self.hyparam_space = hyparam_space

    
    def homogenize_earnings_dates(
        self,
        path_to_earnings: str
    ) -> pd.DataFrame:
        earnings = pd.read_csv(
            path_to_earnings
        ).sort_values(
            ['TICKER', 'ANNDATS'], 
            ascending=[True, False]
        ).set_index(
            'TICKER'
        )

        earnings['ERNUM'] = earnings.groupby(
            'TICKER', 
            group_keys = False
        ).apply(
            lambda x: (x.ANNDATS != x.ANNDATS.shift(1)).cumsum()[::-1]
        )
        max_ernum = earnings.ERNUM.max()
        earnings['ERNUM'] = earnings.ERNUM.groupby(
            'TICKER', 
            group_keys=False
        ).apply(
            lambda x: x + (max_ernum - x.max())
        )

        return earnings.set_index('ANNDATS', append=True)


    def format_estimates_data(
        self,
        path_to_estimates: str,
        homogenized_earnings: pd.DataFrame
    ) -> pd.DataFrame:
        estimates = pd.read_csv(path_to_estimates)
        estimates = estimates[estimates.FISCALP == 'QTR']
        estimates.FPEDATS = estimates.FPEDATS.astype('int')
        estimates = estimates.set_index(['TICKER', 'FPEDATS'])

        estimates = estimates.loc[
            estimates.index.get_level_values(
                0
            ).isin(
                homogenized_earnings.index.get_level_values(0)
            )
        ].sort_index()
        estimates['ERNUM'] = 0

        return estimates


    def reconcile_dataset_with_earnings(
        self,
        dataset: pd.DataFrame,
        homogenized_earnings: pd.DataFrame
    ) -> pd.DataFrame:
        for idx, row in homogenized_earnings.iterrows():
            try:
                dataset.loc[
                    (
                        idx[0], 
                        slice(idx[1])
                    ),
                    'ERNUM'
                ] = row.ERNUM
            except:
                pass
        
        return dataset


    def reconcile_estimates_with_earnings(
        self,
        homogenized_earnings: pd.DataFrame,
        formatted_estimates: pd.DataFrame
    ) -> pd.DataFrame:
        homogenized_estimates = self.reconcile_dataset_with_earnings(
            formatted_estimates,
            homogenized_earnings
        )
        return homogenized_estimates.droplevel(1).set_index('ERNUM', append=True).sort_index()


    def extract_features_from_column(
        self,
        dataset: pd.DataFrame,
        column_name: str
    ) -> pd.DataFrame:
        ret, col = [], column_name
        for ticker, ernum in dataset.index.unique():
            ftrs = pd.DataFrame(
                tsfe.catch22_all(
                    dataset.loc[
                        (ticker, ernum), 
                        col
                    ]
                )
            ).set_index(
                'names'
            ).T

            ftrs.columns.name = None
            ftrs['ticker'] = ticker
            ftrs['ernum'] = ernum
            ftrs.set_index(['ticker', 'ernum'], inplace=True)
            ret.append(ftrs)

        all_ftrs = pd.concat(ret).add_prefix(col)
        return all_ftrs


    def extract_features_from_dataset(
        self,
        dataset: pd.DataFrame
    ) -> pd.DataFrame:
        ret = []
        for col in dataset.columns:
            ret.append(
                self.extract_features_from_column(
                    dataset,
                    col
                )
            )

        all_ftrs = pd.concat(
            ret, 
            axis = 1
        ).sort_index()

        return all_ftrs


    @ignore_warnings(category=ConvergenceWarning)
    def tune_train_test(
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        algo, 
    ):
        trials = Trials()
        thePredictionDict = []
        thePredictionEvalDict = {}
        theFeatureImportanceDict = {}
        
        def objective(params):
            model.set_params(**params)
            
            score = cross_val_score(
                model, X_train, y_train, cv=3, n_jobs=-1, error_score=0.99, 
                scoring=metrics.make_scorer(custom_scoring, greater_is_better=False)
                # scoring='neg_mean_absolute_error'
            )
            return {'loss':  -np.mean(score), 'status': STATUS_OK}

        best_classifier = fmin(
            objective, params, algo=tpe.suggest, max_evals=10, trials=trials, show_progressbar=False
        )
        best_params = space_eval(params, best_classifier)

        opti = model
        opti.set_params(**best_params)

        opti_model = opti.fit(
            X_train,
            y_train
        )
        y_pred = opti_model.predict(X_test)
        
        new_y_pred = scale_predicted_returns(pd.Series(y_pred, index=index))

        thePredictionEvalDict["MODEL"] = algo
        thePredictionEvalDict["DATE"] = date
        
        thePredictionEvalDict["IC"], thePredictionEvalDict["T"] =\
            information_coefficient_t_statistic(y_test.div(100), new_y_pred)

        for i in index:
            thePredictionDict += [
                {"MODEL": algo, "DATE": date, "SEDOL": i, "RETURN": new_y_pred.loc[i]}
            ]

        if algo == "LinearRegression":
            coef_sig = opti_model.coef_
            theFeatureImportanceDict["DATE"] = date
            theFeatureImportanceDict["MODEL"] = algo
            for i in range(len(numerical_columns)):
                theFeatureImportanceDict[numerical_columns[i]] = coef_sig[i]
        if algo == "AdaBoost":
            coef_sig = opti_model.feature_importances_
            theFeatureImportanceDict["DATE"] = date
            theFeatureImportanceDict["MODEL"] = algo
            for i in range(len(numerical_columns)):
                theFeatureImportanceDict[numerical_columns[i]] = coef_sig[i]

        return thePredictionDict, thePredictionEvalDict, theFeatureImportanceDict