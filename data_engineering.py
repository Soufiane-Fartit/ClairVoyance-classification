import streamlit as st
from session_state import SessionState

import numpy as np
import pandas as pd

import os


class data_engineering_page():

    """

    THIS CLASS PROVIDES THE NECESSARY TOOLS TO CLEAN THE DATA: IMPUTE, ENCODE, SMOTE, SCALE, DISCRETIZE

    """


    def __init__(self,session_state):

        self.session_state = session_state
        self.raw_data = session_state.raw_data
        self.out_col = session_state.out_col
        self.to_drop = session_state.to_drop
        self.num_impute_strategy = session_state.num_impute_strategy
        self.cat_impute_strategy = session_state.cat_impute_strategy
        self.encode_strategy = session_state.encode_strategy
        self.discretize_bins_number = 7
        self.scale_strategy = session_state.scale_strategy
        self.balance_strategy = session_state.balance_strategy
        self.problem_type = session_state.problem_type


    def update_session(self, session_state):

        """

        UP-DATE THE SESSION_STATE WlTH THE SELF ATTRIBUTES

        """

        session_state.raw_data = self.raw_data
        session_state.to_drop = self.to_drop
        session_state.num_impute_strategy = self.num_impute_strategy
        session_state.cat_impute_strategy = self.cat_impute_strategy
        session_state.encode_strategy = self.encode_strategy
        session_state.scale_strategy = self.scale_strategy
        session_state.balance_strategy = self.balance_strategy


    def get_col_to_drop(self):

        """

        ASKS THE USER OF THEY WANT TO DROP ANY COLUMN OF THE DATASET

        """

        st.title('Dropping columns')
        self.to_drop = st.multiselect('Columns to drop?', self.session_state.raw_data.drop(self.out_col,axis=1).columns)

    def get_impute_strategy(self):

        """

        ASKS THE USER ABOUT HOW THEY WANT TO FILL MISSING VALUES
        FOR BOTH NUMERICAL AND CATEGORICAL COLUMNS

        """

        st.title("Impute Strategy")
        self.num_impute_strategy = st.selectbox("numerical data imputing strategy",
                                            ["None",
                                            "Simple Imputer (mean)",
                                            "Simple Imputer (median)",
                                            "Simple Imputer (most frequent)",
                                            "Simple Imputer (constant)",
                                            "Iterative Imputer (mean)",
                                            "Iterative Imputer (median)",
                                            "Iterative Imputer (most frequent)",
                                            "Iterative Imputer (constant)",
                                            "KNN Imputer"])
        self.cat_impute_strategy = st.selectbox("categorical data imputing strategy",
                                            ["None",
                                            "Simple Imputer (most frequent)",
                                            "Simple Imputer (constant)"])


    def get_encode_strategy(self):

        """

        ASKS THE USER ABOUT HOW THEY WANT TO ENCODE CATEGORICAL COLUMNS

        """

        st.title("Categorical Data Encoding Strategy")
        self.encode_strategy = st.selectbox("encode strategy",
                                        ["None",
                                        "OneHot Encoding",
                                        "Label Encoding"])

    def get_discretize_strategy(self):

        """

        ASKS THE USER IF THEY WANT TO DISCRETIZE ANY COLUMN, AND THE NUMBER OF BINS TO USE

        """

        st.title("Discretizing Strategy")
        st.warning("The discretization happens after the scaling,\n the output will be ordinal high values: [1,2,3,...] ")
        self.discretize_bins_number = st.slider("number of bins", 2,15,7,1)
        self.discretize_cols = st.multiselect("discretize the following columns", self.raw_data.columns)

    def get_scale_strategy(self):

        """

        ASKS THE USER ABOUT HOW THEY WANT TO SCALE THE DATA

        """

        st.title("Scaling Strategy")
        self.scale_strategy = st.selectbox("scaling strategy",
                                        ["None",
                                        "MinMaxScaler",
                                        "StandardScaler",
                                        "MaxAbsScaler",
                                        "RobustScaler"])


    def get_balance_strategy(self):

        """

        ASKS THE USER IF THEY WANT TO FIX THE DATA IMBALANCE
        ONLY IN CASE OF A CLASSIFICATION PROBLEM

        """

        st.title("Balance Strategy")
        self.balance_strategy = st.selectbox("upsample data using SMOTE ?", ["No", "Yes"])

    def merge_data(self,categorical_encoded, numerical_filled):

        """

        CONCATENATE 2 DATAFRAMES ON THE COLUMNS AXIS

        """

        return pd.concat([categorical_encoded, numerical_filled], axis = 1)

    def show_data_balance(self):

        """

        PRINTS THE NUMBER OF INSTANCES FOR EACH CLASS
        ONLY IN CASE OF CLASSIFICATION

        """

        st.subheader('Data balance')
        if self.out_col != None:
            st.table(self.raw_data[self.out_col].value_counts())
        else:
            st.warning("please select the output column (classification problems)")

    def summary(self):

        """

        PRINT A SUMMARY OF THE USERS CHOICES
        THIS FUNCTION IS ONLY USED FOR DEBUGGING

        """

        st.title("Summary")
        st.write(self.to_drop)
        st.write(self.num_impute_strategy)
        st.write(self.cat_impute_strategy)
        st.write(self.encode_strategy)
        st.write(self.scale_strategy)
        st.write(self.balance_strategy)


    def drop_cols(self):

        """

        DROPS THE COLUMNS THAT THE USER ASKED FOR TO BE DROPED

        """

        self.raw_data.drop(self.to_drop, axis=1, inplace=True)
        self.update_session(self.session_state)


    def numerical_impute(self):

        """

        FILL MISSING VALUES OF NUMERICAL COLUMNS

        """

        numerical = self.raw_data.select_dtypes(exclude='object')

        if self.num_impute_strategy == "None":
            pass

        elif "Simple" in self.num_impute_strategy:
            strat = ""
            if "mean" in self.num_impute_strategy:
                strat = "mean"
            elif "median" in self.num_impute_strategy:
                strat = "mean"
            elif "most frequent" in self.num_impute_strategy:
                strat = "most_frequent"
            elif "constant" in self.num_impute_strategy:
                strat = "constant"

            from sklearn.impute import SimpleImputer
            numerical_imputer = SimpleImputer(strategy=strat)
            numerical_filled = numerical_imputer.fit_transform(numerical)
            numerical_filled = pd.DataFrame(numerical_filled, columns = numerical.columns)
            self.raw_data = self.merge_data(numerical_filled, self.raw_data.select_dtypes(include='object'))
            self.update_session(self.session_state)


        elif "Iterative" in self.num_impute_strategy:
            strat = ""
            if "mean" in self.num_impute_strategy:
                strat = "mean"
            elif "median" in self.num_impute_strategy:
                strat = "mean"
            elif "most frequent" in self.num_impute_strategy:
                strat = "most_frequent"
            elif "constant" in self.num_impute_strategy:
                strat = "constant"

            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            numerical_imputer = IterativeImputer(initial_strategy=strat)
            numerical_filled = numerical_imputer.fit_transform(numerical)
            numerical_filled = pd.DataFrame(numerical_filled, columns = numerical.columns)
            self.raw_data = self.merge_data(numerical_filled, self.raw_data.select_dtypes(include='object'))
            self.update_session(self.session_state)


        elif self.num_impute_strategy == "KNN Imputer":

            from sklearn.impute import KNNImputer
            numerical_imputer = KNNImputer()
            numerical_filled = numerical_imputer.fit_transform(numerical)
            numerical_filled = pd.DataFrame(numerical_filled, columns = numerical.columns)
            self.raw_data = self.merge_data(numerical_filled, self.raw_data.select_dtypes(include='object'))
            self.update_session(self.session_state)


    def categorical_impute(self):

        """

        FILL MISSING VALUES OF CATEGORICAL COLUMNS

        """

        categorical = self.raw_data.select_dtypes(include='object')

        if self.cat_impute_strategy == "None":
            pass

        elif "Simple" in self.cat_impute_strategy:
            strat = ""
            if "most frequent" in self.cat_impute_strategy:
                strat = "most_frequent"
            elif "constant" in self.cat_impute_strategy:
                strat = "constant"

            from sklearn.impute import SimpleImputer
            categorical_imputer = SimpleImputer(strategy=strat)
            categorical_filled = categorical_imputer.fit_transform(categorical)
            categorical_filled = pd.DataFrame(categorical_filled, columns = categorical.columns)
            self.raw_data = self.merge_data(categorical_filled, self.raw_data.select_dtypes(exclude='object'))
            self.update_session(self.session_state)

        elif "Iterative" in self.cat_impute_strategy:
            if "most frequent" in self.cat_impute_strategy:
                strat = "most_frequent"
            elif "constant" in self.cat_impute_strategy:
                strat = "constant"

            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            categorical_imputer = IterativeImputer(initial_strategy=strat)
            categorical_filled = categorical_imputer.fit_transform(categorical)
            categorical_filled = pd.DataFrame(categorical_filled, columns = categorical.columns)
            self.raw_data = self.merge_data(categorical_filled, self.raw_data.select_dtypes(exclude='object'))
            self.update_session(self.session_state)

        elif self.cat_impute_strategy == "KNN Imputer":
            from sklearn.impute import KNNImputer
            categorical_imputer = KNNImputer()
            categorical_filled = categorical_imputer.fit_transform(categorical)
            categorical_filled = pd.DataFrame(categorical_filled, columns = categorical.columns)
            self.raw_data = self.merge_data(categorical_filled, self.raw_data.select_dtypes(exclude='object'))
            self.update_session(self.session_state)

    def categorical_encoding(self):
        categorical = self.raw_data.drop(self.out_col,axis=1).select_dtypes(include='object')

        if self.encode_strategy == "None":
            pass

        elif self.encode_strategy == "OneHot Encoding":
            from sklearn.preprocessing import OneHotEncoder
            enc = OneHotEncoder(handle_unknown='ignore')
            categorical_enc = enc.fit_transform(categorical)
            categorical_enc = pd.DataFrame(categorical_enc.toarray(), columns = enc.get_feature_names(categorical.columns))
            self.raw_data = self.merge_data(categorical_enc, self.raw_data.select_dtypes(exclude='object'))
            self.update_session(self.session_state)

        elif self.encode_strategy == "Label Encoding":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            self.raw_data = self.merge_data(self.raw_data.drop(self.out_col,axis=1).apply(le.fit_transform), self.raw_data[self.out_col])
            """
            categorical_enc = le.fit_transform(categorical)
            categorical_enc = pd.DataFrame(categorical_enc.toarray())
            self.raw_data = self.merge_data(categorical_enc, self.raw_data.select_dtypes(exclude='object'))
            """
            self.update_session(self.session_state)


    def scale(self):

        """

        SCALE THE DATASET USING THE SCALER CHOSEN BY THE USER
        NOTE: DATA SHOULD BE ENCODED BEFORE SCALING

        """

        if self.scale_strategy == "None":
            pass

        elif self.scale_strategy == "MinMaxScaler":

            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            y = self.raw_data.loc[:,self.out_col]
            X = self.raw_data.drop(self.out_col, axis = 1)
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            self.raw_data = self.merge_data(X_scaled, y)
            self.update_session(self.session_state)

        elif self.scale_strategy == "StandardScaler":

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            y = self.raw_data.loc[:,self.out_col]
            X = self.raw_data.drop(self.out_col, axis = 1)
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            self.raw_data = self.merge_data(X_scaled, y)
            self.update_session(self.session_state)

        elif self.scale_strategy == "MaxAbsScaler":

            from sklearn.preprocessing import MaxAbsScaler
            scaler = MaxAbsScaler()
            y = self.raw_data.loc[:,self.out_col]
            X = self.raw_data.drop(self.out_col, axis = 1)
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            self.raw_data = self.merge_data(X_scaled, y)
            self.update_session(self.session_state)

        elif self.scale_strategy == "RobustScaler":

            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            y = self.raw_data.loc[:,self.out_col]
            X = self.raw_data.drop(self.out_col, axis = 1)
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            self.raw_data = self.merge_data(X_scaled, y)
            self.update_session(self.session_state)


    def balance(self):

        """

        USES SMOTE TO GENERATE SYNTHETIC DATA FOR LESS PRESENT CLASSES

        """

        if self.balance_strategy == "No":
            pass
        else:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)

            y = self.raw_data.loc[:,self.out_col]
            X = self.raw_data.drop(self.out_col, axis = 1)

            X_res, y_res = sm.fit_resample(X, y)
            X_res = pd.DataFrame(X_res, columns=X.columns)
            y_res = pd.DataFrame(y_res, columns=[self.out_col])

            self.raw_data = self.merge_data(X_res, y_res)

            self.update_session(self.session_state)
            self.show_data_balance()

    def discretize(self):

        """

        DISCRETIZE A CONTINUOUS VALUES COLUMN INTO N VALUES
        THE NUMBER N IS LEFT TO THE USER TO CHOOSE

        """

        if len(self.discretize_cols) != 0:
            from sklearn.preprocessing import KBinsDiscretizer
            st.write(self.discretize_cols)
            est = KBinsDiscretizer(n_bins=self.discretize_bins_number, encode='ordinal', strategy='kmeans')
            self.raw_data[self.discretize_cols] = est.fit_transform(self.raw_data[self.discretize_cols])
        else:
            pass


    def run_data_engineering(self, prob_type):

        """

        RUNS THE EXECUTION PART OF THE ROUTINE FUNCTION

        """

        if st.button("GO"):
            self.drop_cols()
            self.numerical_impute()
            self.categorical_impute()
            self.categorical_encoding()
            self.scale()
            if prob_type == "classification":
                self.balance()
                self.discretize()
            st.subheader("Transformed Data")
            st.write(self.raw_data.head())


    def routine(self, prob_type):

        """

        THE LOOP THAT STREAMLIT WILL BE EXECUTING

        """

        self.get_col_to_drop()
        self.get_impute_strategy()
        self.get_encode_strategy()
        self.get_scale_strategy()
        if prob_type == "classification":
            self.get_balance_strategy()
            self.get_discretize_strategy()
        self.run_data_engineering(prob_type)


if __name__ == "__main__":
    session_state = SessionState.get(raw_data=pd.read_csv("Churn_Modelling.csv", nrows=10000),
                                    out_col="Exited", #change this bro
                                    to_drop=[],
                                    num_impute_strategy="",
                                    cat_impute_strategy="",
                                    encode_strategy="",
                                    scale_strategy="",
                                    balance_strategy="")

    de = data_engineering_page(session_state)
    de.routine()
