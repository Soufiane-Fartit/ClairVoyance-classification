import streamlit as st
from session_state import SessionState

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import os


class ml_page():

    """

    This class does the machine learning in classification problems

    """

    def __init__(self,session_state):
        self.session_state = session_state
        self. raw_data = session_state.raw_data
        self.out_col = session_state.out_col
        self.ml_list_names = ['Support Vector Machines', 'KNN', 'Random Forest Classifier', 'Adaboost', 'XGBoost']
        self.chosen_models_names = []
        self.chosen_models = []
        self.meta_ml_list_names = ["Logistic Regression", "Support Vector Machines"]
        self.algorithms = []
        self.percent_train = 80
        self.cv_check = False
        self.k_cv = 3
        self.meta_model_check = False
        self.meta_model = ""
        self.meta_model_type = ""


    def get_ml_algs(self):

        """

        ASKS THE USER FOR THE ML ALGORITHMS TO USE FOR TRAINING

        """

        self.algorithms = st.multiselect('ML algorithms?', self.ml_list_names)

    def get_train_ratio(self):

        """

        ASKS THE USER FOR THE TRAINING SET RATIO TO USE

        """

        self.percent_train = st.slider('train set ratio', 0, 100, 80)

    def get_cv_check(self):

        """

        ASKS THE USER IF THEY WANT TO USE CROSS VALIDATION

        """

        self.cv_check= st.checkbox("use cross validation ?")

    def get_cv_k(self):

        """

        ASKS THE USER FOR THE NUMBER OF K - FOLD (USED IN CASE THEY CHOSE TO USE CROSS VALIDATION)

        """

        self.k_cv = st.slider('K fold cross validation', 2, 10, 3)

    def get_meta_model_check(self):

        """

        ASKS THE USER IF THEY WANT TO USE A META MODEL ON TOP OF THE FIRST ONE

        """

        self.meta_model_check = st.checkbox("use a meta model ?")

    def get_meta_model(self):

        """

        ASKS THE USER ABOUT THE TYPE OF META MODEL THEY WANT TO USE

        """

        self.meta_model_type = st.selectbox("meta model type :", ["voting", "stacking"])
        self.meta_model = st.radio("choose a meta model (if meta model type is stacking)", self.meta_ml_list_names)


    def train_algs(self):

        """

        TRAIN WlTHOUT CROSS VALIDATION

        """

        st.subheader("Results")
        self.chosen_models_names = []
        self.chosen_models = []

        if len(self.algorithms)==0:
            st.warning('You should select at least one algorithm')
            return

        X = self.raw_data.drop(self.out_col, axis = 1)
        y = self.raw_data[self.out_col]
        msk = np.random.rand(len(X)) < self.percent_train/100
        X_train = X[msk]
        X_test = X[~msk]
        Y_train = y[msk]
        Y_test = y[~msk]

        for alg in self.algorithms:

            if alg == 'Support Vector Machines':
                from sklearn.svm import SVC
                svc = SVC(probability=True)
                svc.fit(X_train, Y_train)
                svc_predictions = svc.predict(X_test)
                st.write('svc accuracy on test set',accuracy_score(svc_predictions, Y_test))

                self.chosen_models_names.append('Support Vector Machines')
                self.chosen_models.append(svc)

            elif alg == 'KNN':
                from sklearn.neighbors import KNeighborsClassifier
                knn = KNeighborsClassifier()
                knn.fit(X_train, Y_train)
                knn_predictions = knn.predict(X_test)
                st.write('knn accuracy on test set',accuracy_score(knn_predictions, Y_test))

                self.chosen_models_names.append('KNN')
                self.chosen_models.append(knn)

            elif alg == 'Random Forest Classifier':
                from sklearn.ensemble import RandomForestClassifier
                rfc = RandomForestClassifier()
                rfc.fit(X_train, Y_train)
                rfc_predictions = rfc.predict(X_test)
                st.write('rfc accuracy on test set',accuracy_score(rfc_predictions, Y_test))

                self.chosen_models_names.append('Random Forest Classifier')
                self.chosen_models.append(rfc)

            elif alg == 'Adaboost':
                from sklearn.ensemble import AdaBoostClassifier
                ada = AdaBoostClassifier()
                ada.fit(X_train, Y_train)
                ada_predictions = ada.predict(X_test)
                st.write('ada accuracy on test set',accuracy_score(ada_predictions, Y_test))

                self.chosen_models_names.append('Adaboost')
                self.chosen_models.append(ada)

            elif alg == 'XGBoost':
                import xgboost as xgb
                xgb = xgb.XGBClassifier(n_estimators=300)
                xgb.fit(X_train, Y_train, verbose =0)
                xgb_predictions = xgb.predict(X_test)
                st.write('xgb accuracy on test set',accuracy_score(xgb_predictions, Y_test))

                self.chosen_models_names.append('XGBoost')
                self.chosen_models.append(xgb)

        if self.meta_model_check:
            if self.meta_model_type == "voting":
                from sklearn.ensemble import VotingClassifier
                from sklearn.linear_model import LogisticRegression
                stack = VotingClassifier(estimators=list(zip(self.chosen_models_names,self.chosen_models)), voting='hard')
                stack.fit(X_train, Y_train)
                stack_predictions = stack.predict(X_test)
                st.write('voting accuracy',accuracy_score(stack_predictions, Y_test))


            else:
                from sklearn.ensemble import StackingClassifier

                if self.meta_model == "Logistic Regression":
                    from sklearn.linear_model import LogisticRegression
                    stack = StackingClassifier(estimators=list(zip(self.chosen_models_names,self.chosen_models)), final_estimator=LogisticRegression())

                elif self.meta_model == "Support Vector Machines":
                    from sklearn.svm import SVC
                    stack = StackingClassifier(estimators=list(zip(self.chosen_models_names,self.chosen_models)), final_estimator=SVC(probability=True))

                stack.fit(X_train, Y_train)
                stack_predictions = stack.predict(X_test)
                st.write(self.meta_model+' stack score',accuracy_score(stack_predictions, Y_test))

    def train_algs_cv(self):

        """

        TRAIN USING CROSS VALIDATION

        """

        st.subheader("Results using cross validation")
        self.chosen_models_names = []
        self.chosen_models = []

        if len(self.algorithms)==0:
            st.warning('You should select at least one algorithm')
            return

        X_train = self.raw_data.drop(self.out_col, axis = 1)
        Y_train = self.raw_data[self.out_col]

        for alg in self.algorithms:

            if alg == 'Support Vector Machines':
                from sklearn.svm import SVC
                svc = SVC(probability=True)
                svc_scores = cross_val_score(svc, X_train, Y_train, cv = self.k_cv)
                st.write('svc score :', svc_scores.mean())

                self.chosen_models_names.append('Support Vector Machines')
                self.chosen_models.append(svc)

            elif alg == 'KNN':
                from sklearn.neighbors import KNeighborsClassifier
                knn = KNeighborsClassifier()
                knn_scores = cross_val_score(knn, X_train, Y_train, cv = self.k_cv)
                st.write('knn score :', knn_scores.mean())

                self.chosen_models_names.append('KNN')
                self.chosen_models.append(knn)

            elif alg == 'Random Forest Classifier':
                from sklearn.ensemble import RandomForestClassifier
                rfc = RandomForestClassifier()
                rfc_scores = cross_val_score(rfc, X_train, Y_train, cv = self.k_cv)
                st.write('rfc score :', rfc_scores.mean())

                self.chosen_models_names.append('Random Forest Classifier')
                self.chosen_models.append(rfc)

            elif alg == 'Adaboost':
                from sklearn.ensemble import AdaBoostClassifier
                ada = AdaBoostClassifier()
                ada_scores = cross_val_score(ada, X_train, Y_train, cv = self.k_cv)
                st.write('ada score :', ada_scores.mean())

                self.chosen_models_names.append('Adaboost')
                self.chosen_models.append(ada)

            elif alg == 'XGBoost':
                import xgboost as xgb
                xgb = xgb.XGBClassifier(n_estimators=300)
                xgb_scores = cross_val_score(xgb, X_train, Y_train, cv = self.k_cv)
                st.write('xgb score :', xgb_scores.mean())

                self.chosen_models_names.append('XGBoost')
                self.chosen_models.append(xgb)

        if self.meta_model_check:
            if self.meta_model_type == "voting":
                from sklearn.ensemble import VotingClassifier
                from sklearn.linear_model import LogisticRegression
                stack = VotingClassifier(estimators=list(zip(self.chosen_models_names,self.chosen_models)), voting='hard')
                stack_scores = cross_val_score(stack, X_train, Y_train, cv = self.k_cv)
                st.write('voting score :', stack_scores.mean())

            else:
                from sklearn.ensemble import StackingClassifier

                if self.meta_model == "Logistic Regression":
                    from sklearn.linear_model import LogisticRegression
                    stack = StackingClassifier(estimators=list(zip(self.chosen_models_names,self.chosen_models)), final_estimator=LogisticRegression())

                elif self.meta_model == "Support Vector Machines":
                    from sklearn.svm import SVC
                    stack = StackingClassifier(estimators=list(zip(self.chosen_models_names,self.chosen_models)), final_estimator=SVC(probability=True))

                stack_scores = cross_val_score(stack, X_train, Y_train, cv = self.k_cv)
                st.write(self.meta_model+' stack score using cv :', stack_scores.mean())


    def train(self):

        """

        LAUNCH THE TRAINING
        EITHER USING CROSS VALIDATION OR NOT

        """

        if not self.cv_check:
            self.train_algs()
        else:
            self.train_algs_cv()

    def routine(self):

        """

        THE LOOP THAT STREAMLIT WILL BE EXECUTING

        """

        self.get_ml_algs()
        self.get_train_ratio()
        self.get_cv_check()
        self.get_cv_k()
        self.get_meta_model_check()
        self.get_meta_model()
        self.train()
