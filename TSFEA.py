from typing import Dict, List, Any

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor 
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn import metrics 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

        return earnings


    def homogenize_estmates_dates(
        self,
        path_to_estimates: str,
        homogenized_earnings: pd.DataFrame
    ) -> pd.DataFrame:
        pass



    def get_earnings_number(
        self,
        path_to_estimates: str,
        path_to_earnings: str
    ) -> pd.DataFrame:
        pass