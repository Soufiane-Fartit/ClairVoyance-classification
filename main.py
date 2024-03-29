import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from session_state import SessionState
from load_page import loading_page
from data_engineering import data_engineering_page
from exploration import exploration_page
from machine_learning import ml_page
from machine_learning_regression import ml_reg_page

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

import os
import base64


class clairevoyance():

    """

    The core of the application
    This class orchestrate all the main components (loading page, data engineering page, EDA page, ML page)
    It contains all of the sidebar elements and initiate the other classes with the right "session_state"

    """

    def __init__(self, session_state):
        self.tab = "Data Loading"
        self.loading_session_state = SessionState.get(file = session_state.file,
                                                file_button = session_state.file_button,
                                                link = session_state.link,
                                                path = session_state.path,
                                                path_button = session_state.path_button,
                                                link_button = session_state.link_button,
                                                raw_data = session_state.raw_data,
                                                data_check = session_state.data_check,
                                                data_load_state = session_state.data_load_state,
                                                out_col = session_state.out_col,
                                                unique_values = session_state.unique_values,
                                                selected_filename = session_state.selected_filename,
                                                out_col_check = session_state.out_col_check,
                                                problem_type = session_state.problem_type)

        self.de_session_state = SessionState.get(raw_data=session_state.raw_data,
                                                out_col=session_state.out_col,
                                                to_drop=session_state.to_drop,
                                                num_impute_strategy=session_state.num_impute_strategy,
                                                cat_impute_strategy=session_state.cat_impute_strategy,
                                                encode_strategy=session_state.encode_strategy,
                                                scale_strategy=session_state.scale_strategy,
                                                balance_strategy=session_state.balance_strategy,
                                                problem_type = session_state.problem_type)

        self.viz_session_state = SessionState.get(raw_data = session_state.raw_data,
                                                out_col=session_state.out_col,
                                                problem_type = session_state.problem_type)

        self.ml_session_state = SessionState.get(raw_data = session_state.raw_data,
                                                out_col=session_state.out_col)

        self.load_page = loading_page(self.loading_session_state)
        self.de_page = data_engineering_page(self.de_session_state)
        self.viz_page = exploration_page(self.viz_session_state)
        self.ml_page = ml_page(self.ml_session_state)
        self.ml_reg_page = ml_reg_page(self.ml_session_state)
        self.problem_type = session_state.problem_type

    def switcher(self):
        st.sidebar.title("ClairVoyance - classification/regression")
        self.problem_type = st.sidebar.selectbox("type of problem", ["classification", "regression"])
        self.tab = st.sidebar.radio('Pick an option', ['Data Loading', 'Data Engineering', 'Data Exploration', 'Machine Learning'])
        st.sidebar.header("Download current data")
        if st.sidebar.button("download"):
            csv = session_state.raw_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.sidebar.markdown(href, unsafe_allow_html=True)
        st.sidebar.header("About")
        st.sidebar.info('This app is made and maintained \n\nby [Soufiane Fartit]('
                    'https://www.linkedin.com/in/soufiane-fartit/).')


    def go_to_load(self):
        if self.tab == 'Data Loading':
            self.load_page.routine(self.problem_type)

    def go_to_de(self):
        if self.tab == 'Data Engineering':
            self.de_page.routine(self.problem_type)

    def go_to_viz(self):
        if self.tab == 'Data Exploration':
            self.viz_page.routine(self.problem_type)

    def go_to_ml(self):
        if self.tab == 'Machine Learning':
            if self.problem_type == "classification":
                self.ml_page.routine()
            if self.problem_type == "regression":
                self.ml_reg_page.routine()

    def routine(self):

        """

        THE LOOP THAT STREAMLIT WILL BE EXECUTING

        """

        self.switcher()
        self.go_to_load()
        self.go_to_de()
        self.go_to_viz()
        self.go_to_ml()






if __name__ == "__main__":

    session_state = SessionState.get(file = None,
                                    file_button = None,
                                    link = "",
                                    path = "",
                                    path_button = None,
                                    link_button = None,
                                    raw_data = None,
                                    data_check = False,
                                    data_load_state = 0,
                                    out_col = "",
                                    unique_values = [],
                                    selected_filename = "",
                                    out_col_check = False,
                                    to_drop=[],
                                    num_impute_strategy="",
                                    cat_impute_strategy="",
                                    encode_strategy="",
                                    scale_strategy="",
                                    balance_strategy="",
                                    problem_type = "classification")

    page_zero = clairevoyance(session_state)
    page_zero.routine()
