from json import load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder
from chi_square import ChiSquare
import scipy.stats as stats

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#  https://www.kaggle.com/code/nehapawar/churn-prediction-using-logistic-regression/notebook
class Main:

    def __init__(self):
        self.load_data()
        self.prepare_data()
        #self.plot_1()
        #self.categorical_churn()
        self.chi_square()
        self.anova_test()
        self.convert_categorical_to_numerical()
        self.feature_extraction()
        self.train_linear_model()

    def load_data(self):
        self.data: DataFrame = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
        print(self.data.head())
        print(self.data['Churn'].value_counts())

    def prepare_data(self):
        self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')
        print(self.data.isnull().sum())
        self.data['TotalCharges'].fillna(self.data['TotalCharges'].mean(), inplace=True)  # fill null with sum
        self.categorical_var = list(self.data.dtypes.loc[self.data.dtypes == 'object'].index)
        print(len(self.categorical_var))
        print(self.categorical_var)
        self.categorical_var.remove('customerID')
        self.continuous_var = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        print(self.data.describe())

    def plot_1(self):
        fig, ax = plt.subplots(6, 3, figsize=(12, 20))

        sns.countplot(self.data['gender'], ax=ax[0][0])
        sns.countplot(self.data['Partner'], ax=ax[0][1])
        sns.countplot(self.data['Dependents'], ax=ax[0][2])

        sns.countplot(self.data['PhoneService'], ax=ax[1][0])
        sns.countplot(self.data['MultipleLines'], ax=ax[1][1])
        sns.countplot(self.data['InternetService'], ax=ax[1][2])

        sns.countplot(self.data['OnlineSecurity'], ax=ax[2][0])
        sns.countplot(self.data['OnlineBackup'], ax=ax[2][1])
        sns.countplot(self.data['DeviceProtection'], ax=ax[2][2])

        sns.countplot(self.data['TechSupport'], ax=ax[3][0])
        sns.countplot(self.data['StreamingTV'], ax=ax[3][1])
        sns.countplot(self.data['StreamingMovies'], ax=ax[3][2])

        sns.countplot(self.data['Contract'], ax=ax[4][0])
        sns.countplot(self.data['PaperlessBilling'], ax=ax[4][1])
        sns.countplot(self.data['PaymentMethod'], ax=ax[4][2])

        sns.countplot(self.data['Churn'], ax=ax[5][0])

        #--- Continuous variables representation
        nd = pd.melt(self.data, value_vars=self.continuous_var)
        n1 = sns.FacetGrid(nd, col='variable', col_wrap=2, sharex=False, sharey=False)
        n1 = n1.map(sns.distplot, 'value')

        #--- Correlation between continuous
        corr = self.data[self.continuous_var].corr()
        sns.heatmap(corr)
        #fig.show()
        plt.show()

    def categorical_churn(self):
        # Как категориальные влияют на отток
        for var in self.categorical_var:
            if var != 'Churn':
                test = self.data.groupby([var, 'Churn'])
                print(test.size(), '\n\n')


    def chi_square(self):
        df = self.data
        cT = ChiSquare(df)
        # Feature Selection
        for var in self.categorical_var:
            cT.TestIndependence(colX=var, colY="Churn")


    def anova_test(self):
        # ANOVA test
       for var in self.continuous_var:
            result = stats.f_oneway(self.data[var][self.data['Churn'] == 'Yes'],
                                    self.data[var][self.data['Churn'] == 'No'])
            print(var)
            print(result)

    def convert_categorical_to_numerical(self):
        # convert all the categorical to numerical
        for var in self.categorical_var:
            self.data[var] = self.data[var].astype('category')
        self.data[self.categorical_var] = self.data[self.categorical_var].apply(lambda x: x.cat.codes)
        target = self.data['Churn']
        self.data = self.data.drop('customerID', axis=1)
        self.all_columns = list(self.data.columns)
        self.all_columns.remove('Churn')

    def feature_extraction(self):
        # feature extraction
        self.X = self.data[self.all_columns]  # Features
        y = self.data['Churn']  # Target variable
        # Feature extraction
        model = LogisticRegression()
        rfe = RFE(model)
        fit = rfe.fit(self.X, y)
        print("Num Features: %s" % (fit.n_features_))
        print("Selected Features: %s" % (fit.support_))
        print("Feature Ranking: %s" % (fit.ranking_))
        selected_features_rfe = list(fit.support_)

        self.final_features_rfe = []
        for status, var in zip(selected_features_rfe, self.all_columns):
            if status == True:
                self.final_features_rfe.append(var)

        print(self.final_features_rfe)

    def train_linear_model(self):
        #  train LogisticRegression model
        X_rfe_lr = self.data[self.final_features_rfe]
        y = self.data['Churn']

        X_train_rfe_lr, X_test_rfe_lr, y_train_rfe_lr, y_test_rfe_lr = train_test_split(X_rfe_lr, y, test_size=0.25,
                                                                                        random_state=0)

        self.lr_model = LogisticRegression()

        # fit the model with data
        self.lr_model.fit(X_train_rfe_lr, y_train_rfe_lr)
        y_pred_rfe_lr = self.lr_model.predict(X_test_rfe_lr)

        acc_rfe_lr = metrics.accuracy_score(y_test_rfe_lr, y_pred_rfe_lr)
        print("Accuracy: ", acc_rfe_lr)

        X_train, X_test, y_train, y_test = train_test_split(self.X, y, test_size=0.25, random_state=0)
        # instantiate the model (using the default parameters)
        self.lr_model_single = LogisticRegression()

        # fit the model with data
        self.lr_model_single.fit(X_train, y_train)
        y_pred = self.lr_model_single.predict(X_test)
        lr_acc = metrics.accuracy_score(y_test, y_pred)
        print("Accuracy: ", lr_acc)

        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        print(cnf_matrix)

        from sklearn.metrics import roc_curve, auc
        fpr_1, tpr_1, thresholds = roc_curve(y_test, y_pred_rfe_lr)
        fpr_2, tpr_2, thresholds = roc_curve(y_test, y_pred)
        roc_auc_1 = auc(fpr_1, tpr_1)
        roc_auc_2 = auc(fpr_2, tpr_2)




    def churn_pred(self):
        # prediction
        pass


if __name__ == '__main__':
    m = Main()

