import streamlit as st
from session_state import SessionState

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import os


class ml_reg_page():


    def __init__(self,session_state):
        self.session_state = session_state
        self. raw_data = session_state.raw_data
        self.out_col = session_state.out_col
        self.ml_list_names = ['LinearSVR', 'RidgeCV', 'Random Forest Regressor', 'Adaboost', 'XGBoost']
        self.chosen_models_names = []
        self.chosen_models = []
        self.meta_ml_list_names = ["GradientBoostingRegressor", "RandomForestRegressor"]
        self.algorithms = []
        self.percent_train = 80
        self.cv_check = False
        self.k_cv = 3
        self.meta_model_check = False
        self.meta_model = ""
        self.meta_model_type = ""


    def get_ml_algs(self):
        self.algorithms = st.multiselect('ML algorithms?', self.ml_list_names)

    def get_train_ratio(self):
        self.percent_train = st.slider('train set ratio', 0, 100, 80)

    def get_cv_check(self):
        self.cv_check= st.checkbox("use cross validation ?")

    def get_cv_k(self):
        self.k_cv = st.slider('K fold cross validation', 2, 10, 3)

    def get_meta_model_check(self):
        self.meta_model_check = st.checkbox("use a meta model ?")

    def get_meta_model(self):
        self.meta_model_type = st.selectbox("meta model type :", ["voting", "stacking"])
        self.meta_model = st.radio("choose a meta model (if meta model type is stacking)", self.meta_ml_list_names)


    def train_algs(self):

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

            if alg == 'LinearSVR':
                from sklearn.svm import LinearSVR
                svc = LinearSVR()
                svc.fit(X_train, Y_train)
                st.write("LinearSVR score", svc.score(X_test, Y_test))

                self.chosen_models_names.append('LinearSVR')
                self.chosen_models.append(svc)

            elif alg == 'RidgeCV':
                from sklearn.linear_model import RidgeCV
                rid = RidgeCV()
                rid.fit(X_train, Y_train)
                st.write("RidgeCV score", rid.score(X_test, Y_test))

                self.chosen_models_names.append('RidgeCV')
                self.chosen_models.append(rid)


            elif alg == 'Random Forest Regressor':
                from sklearn.ensemble import RandomForestRegressor
                rfc = RandomForestRegressor()
                rfc.fit(X_train, Y_train)
                st.write("rfc score", rfc.score(X_test, Y_test))

                self.chosen_models_names.append('Random Forest Regressor')
                self.chosen_models.append(rfc)


            elif alg == 'Adaboost':
                from sklearn.ensemble import AdaBoostRegressor
                ada = AdaBoostRegressor()
                ada.fit(X_train, Y_train)
                st.write("ada score", ada.score(X_test, Y_test))

                self.chosen_models_names.append('Adaboost')
                self.chosen_models.append(ada)


            elif alg == 'XGBoost':
                import xgboost as xgb
                xgb = xgb.XGBRegressor(n_estimators=300)
                xgb.fit(X_train, Y_train, verbose =0)
                st.write("xgb score", xgb.score(X_test, Y_test))

                self.chosen_models_names.append('XGBoost')
                self.chosen_models.append(xgb)

        if self.meta_model_check:
            if self.meta_model_type == "voting":
                from sklearn.ensemble import VotingRegressor
                stack = VotingRegressor(estimators=list(zip(self.chosen_models_names,self.chosen_models)))
                stack.fit(X_train, Y_train)
                st.write("stack score", stack.score(X_test, Y_test))

            else:
                from sklearn.ensemble import StackingRegressor

                if self.meta_model == "GradientBoostingRegressor":
                    from sklearn.ensemble import GradientBoostingRegressor
                    stack = StackingRegressor(estimators=list(zip(self.chosen_models_names,self.chosen_models)), final_estimator=GradientBoostingRegressor())

                elif self.meta_model == "RandomForestRegressor":
                    from sklearn.ensemble import RandomForestRegressor
                    stack = StackingRegressor(estimators=list(zip(self.chosen_models_names,self.chosen_models)), final_estimator=RandomForestRegressor())

                stack.fit(X_train, Y_train)
                st.write("stack score", stack.score(X_test, Y_test))
                

    def train_algs_cv(self):
        st.subheader("Results using cross validation")
        self.chosen_models_names = []
        self.chosen_models = []

        if len(self.algorithms)==0:
            st.warning('You should select at least one algorithm')
            return

        X_train = self.raw_data.drop(self.out_col, axis = 1)
        Y_train = self.raw_data[self.out_col]

        for alg in self.algorithms:

            if alg == 'LinearSVR':
                from sklearn.svm import LinearSVR
                svc = LinearSVR()
                svc_scores = cross_val_score(svc, X_train, Y_train, scoring='r2', cv = self.k_cv)
                st.write('LinearSVR score :', svc_scores.mean())

                self.chosen_models_names.append('LinearSVR')
                self.chosen_models.append(svc)

            elif alg == 'RidgeCV':
                from sklearn.linear_model import RidgeCV
                rid = RidgeCV()
                rid_scores = cross_val_score(rid, X_train, Y_train, scoring='r2', cv = self.k_cv)
                st.write('RidgeCV score :', rid_scores.mean())

                self.chosen_models_names.append('RidgeCV')
                self.chosen_models.append(rid)

            elif alg == 'Random Forest Regressor':
                from sklearn.ensemble import RandomForestRegressor
                rfc = RandomForestRegressor()
                rfc_scores = cross_val_score(rfc, X_train, Y_train, scoring='r2', cv = self.k_cv)
                st.write('Random Forest Regressor score :', rfc_scores.mean())

                self.chosen_models_names.append('Random Forest Regressor')
                self.chosen_models.append(rfc)

            elif alg == 'Adaboost':
                from sklearn.ensemble import AdaBoostRegressor
                ada = AdaBoostRegressor()
                ada_scores = cross_val_score(ada, X_train, Y_train, scoring='r2', cv = self.k_cv)
                st.write('Adaboost score :', ada_scores.mean())

                self.chosen_models_names.append('Adaboost')
                self.chosen_models.append(ada)

            elif alg == 'XGBoost':
                import xgboost as xgb
                xgb = xgb.XGBRegressor(n_estimators=300)
                xgb_scores = cross_val_score(xgb, X_train, Y_train, scoring='r2', cv = self.k_cv)
                st.write('xgb score :', xgb_scores.mean())

                self.chosen_models_names.append('XGBoost')
                self.chosen_models.append(xgb)

        if self.meta_model_check:
            if self.meta_model_type == "voting":
                from sklearn.ensemble import VotingRegressor
                stack = VotingRegressor(estimators=list(zip(self.chosen_models_names,self.chosen_models)))
                stack_scores = cross_val_score(stack, X_train, Y_train, scoring='r2', cv = self.k_cv)
                st.write('voting score :', stack_scores.mean())

            else:
                from sklearn.ensemble import StackingRegressor

                if self.meta_model == "GradientBoostingRegressor":
                    from sklearn.ensemble import GradientBoostingRegressor
                    stack = StackingRegressor(estimators=list(zip(self.chosen_models_names,self.chosen_models)), final_estimator=GradientBoostingRegressor())

                elif self.meta_model == "RandomForestRegressor":
                    from sklearn.ensemble import RandomForestRegressor
                    stack = StackingRegressor(estimators=list(zip(self.chosen_models_names,self.chosen_models)), final_estimator=RandomForestRegressor())

                stack_scores = cross_val_score(stack, X_train, Y_train, scoring='r2', cv = self.k_cv)
                st.write(self.meta_model+' stack score using cv :', stack_scores.mean())


    def train(self):
        if not self.cv_check:
            self.train_algs()
        else:
            self.train_algs_cv()

    def routine(self):
        self.get_ml_algs()
        self.get_train_ratio()
        self.get_cv_check()
        self.get_cv_k()
        self.get_meta_model_check()
        self.get_meta_model()
        self.train()
