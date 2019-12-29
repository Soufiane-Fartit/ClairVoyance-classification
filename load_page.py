import streamlit as st
from session_state import SessionState

import numpy as np
import pandas as pd

import os

class loading_page():

    """

    THIS CLASS CONTROL THE LOADING PAGE WHERE THE USER CHOSES A DATASET
    AND INSPECT IT. IT GIVES A FEW INFORMATIONS ABOUT THE COLUMNS AND THE
    MISSING VALUES AND IN CASE OF A CLASSIFICATION PROBLEM, THE DATA BALANCE

    """

    def __init__(self,session_state):

        self.session_state = session_state
        self.file = session_state.file
        self.file_button = session_state.file_button
        self.link = session_state.link
        self.path = session_state.path
        self.path_button = session_state.path_button
        self.link_button = session_state.link_button
        self.raw_data = session_state.raw_data
        self.data_check = session_state.data_check
        self.data_load_state = session_state.data_load_state
        self.out_col = session_state.out_col
        self.unique_values = session_state.unique_values
        self.selected_filename = session_state.selected_filename
        self.out_col_check = session_state.out_col_check
        self.problem_type = session_state.problem_type

    def update_session(self, session_state):

        """

        SAVE THE STATE OF THE CLASS IN THE SESSION_STATE

        """

        session_state.file = self.file
        session_state.file_button = self.file_button
        session_state.link = self.link
        session_state.path = self.path
        session_state.path_button = self.path_button
        session_state.link_button = self.link_button
        session_state.raw_data = self.raw_data
        session_state.data_check = self.data_check
        session_state.data_load_state = self.data_load_state
        session_state.out_col = self.out_col
        session_state.unique_values = self.unique_values
        session_state.selected_filename = self.selected_filename
        session_state.out_col_check = self.out_col_check



    def file_selector(self, folder_path='.'):

        """

        THIS IS A TEMPORARY WORKAOURND OF A FILE_SELECtOR PROVIDED BY
        ONE OF THE CREATORS OF STREAMLIT ONLINE

        """

        filenames = os.listdir(folder_path)
        self.selected_filename = st.selectbox('or select a demo file', list(filter(lambda k: 'csv' in k, filenames)))
        return os.path.join(folder_path, self.selected_filename)


    def load_data_from_path_or_link(self, path):

        """

        TAKES A PATH (SERVER OR USER) OR A LINE TO A DATASET
        AND LOAD THE DATA

        """

        self.data_load_state = st.text('Loading dataset...')
        self.raw_data = pd.read_csv(path, nrows=10000, index_col=False)
        #self.raw_data = pd.read_csv(path, nrows=10000, sep= ';|,', index_col=False)
        self.data_check = True
        self.data_load_state.success('Loading dataset...done!')


    def load_data(self):

        """

        CHECKS WICH BUTTON HAVE BEEN PRESSED AND LOAD THE DATA ACCORDINLY

        """

        if self.file_button:
            self.load_data_from_path_or_link(self.file)
        if self.link_button:
            self.load_data_from_path_or_link(self.link)
        if self.path_button:
            self.load_data_from_path_or_link(self.path)
        self.update_session(self.session_state)


    def get_path_or_link(self):

        """

        asks the user to chose how to upload data (from the users computer, or from an online dataset, or a demo dataset that we have provided)

        """

        self.file = st.file_uploader("Upload a file from your computer", type=["csv"])
        self.file_button = st.button('Load local data')
        self.link = st.text_input("or provide a link to dataframe", "")
        self.link_button = st.button('Load data')
        self.path = self.file_selector()
        self.path_button = st.button('Load demo data')
        self.update_session(self.session_state)


    def show_raw_data(self):

        """

        PRINT THE 5 FIRST ROWS OF THE DATASET FOR INSPECTION PURPOSES

        """

        st.subheader('Raw data')
        st.write(self.data_check)
        if self.data_check:
            st.write(self.raw_data.head())
        else:
            st.warning("please load the data first")


    def show_data_balance(self):

        """

        PRINTS INFORMATION ABOUT THE NUMBER OF INSTANCES OF EACH CLASS

        """

        st.subheader('Data balance')
        if self.out_col_check:
            st.table(self.raw_data[self.out_col].value_counts())
        else:
            st.warning("please select the output column (classification problems)")


    def show_infos_data(self):

        """

        PRINTS THE INFORMATIONS OF THE DATA GIVEN BY THE PANDAS DESCRIBE FUNCTION
        (MEAN, STD, MIN, MAX ...)

        """

        st.subheader('Infos about data')
        if self.data_check:
            st.dataframe(self.raw_data.describe())
        else:
            st.warning("please load the data first")


    def show_infos_nan(self):

        """

        PRINTS THE NUMBER OF MISSING VALUES OF EACH COLUMN

        """

        st.subheader('Infos about missing values')
        if self.data_check:
            st.table(self.raw_data.isna().sum())
        else:
            st.warning("please load the data first")


    def get_pred_column(self):

        #ASK THE USER FOR THE COLUMN CONTAINING THE PREDICTION VALUES
        #THE LAST COLUMN IS SELECTED BY DEFAULT, AND ONLY CHANGED IF
        #THE USER SPECIFIES OTHERWISE

        if self.data_check:
            self.out_col = st.selectbox('Select the column containing the predictions please',
                                        self.raw_data.columns,
                                        len(self.raw_data.columns)-1,
                                        key = "out_col_selectbox")
            self.out_col_check = st.button('select')
            self.update_session(self.session_state)


    def get_unique_values(self):

        """

        CALCULATE THE UNIQUE VALUES OF THE PREDICTIONS COLUMN

        """

        if self.data_check:
            self.unique_values = list(map(str,self.raw_data[self.out_col].unique().tolist()))
            self.update_session(self.session_state)


    def routine(self, prob_type):

        """

        THE LOOP THAT STREAMLIT WILL BE EXECUTING

        """

        self.get_path_or_link()
        self.load_data()
        self.show_raw_data()
        self.show_infos_data()
        self.get_pred_column()
        self.get_unique_values()
        if prob_type == "classification":
            self.show_data_balance()
        self.show_infos_nan()

if __name__ == "__main__":
    session_state = SessionState.get(link = "",
                                    path = "",
                                    path_button = None,
                                    link_button = None,
                                    raw_data = None,
                                    data_check = False,
                                    data_load_state = 0,
                                    data_separator_check = False,
                                    data_separator= ',',
                                    out_col = "",
                                    unique_values = [],
                                    selected_filename = "",
                                    out_col_check = False)

    l = loading_page(session_state)
    l.routine()
