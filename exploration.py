import streamlit as st
from session_state import SessionState

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

import os
import functools


class exploration_page():


    def __init__(self,session_state):
        self.session_state = session_state
        self.raw_data = session_state.raw_data
        self.out_col = session_state.out_col
        self.unique_out = None
        self.x_axis = None
        self.y_axis = None
        self.scatter_rows = []
        self.corr_method = 'pearson'
        self.problem_type = session_state.problem_type


    def get_2_axis_viz(self):
        st.subheader("2 axis visualization")
        self.x_axis = st.selectbox("Choose a variable for the x-axis", self.raw_data.columns, index=3)
        self.y_axis = st.selectbox("Choose a variable for the y-axis", self.raw_data.columns, index=4)


    def get_scatter_matrix_rows(self):
        st.subheader("scatter matrix")
        self.scatter_rows = st.multiselect('rows for the scatter matrix',
                                        self.raw_data.columns)

    def get_hist_col(self):
        st.subheader("histogram")
        self.hist_col = st.selectbox('Which feature?',
                                    self.raw_data\
                                    .select_dtypes(exclude='object')\
                                    .columns)
    def get_corr(self):
        st.subheader("Correlation Matrix")
        self.corr_method = st.selectbox('correlation method :', ['pearson', 'kendall', 'spearman'])


    def visualize_2_axis(self):
        graph = alt.Chart(self.raw_data).mark_circle().encode(x=self.x_axis, y=self.y_axis)
        st.write(graph)


    def scatter_matrix(self):
        if len(self.scatter_rows) != 0:
            g = sns.pairplot(self.raw_data[self.scatter_rows+[self.out_col]],
                            vars = self.scatter_rows,
                            hue = self.out_col,
                            diag_kind = 'hist')
            st.pyplot(g)
        else:
            st.warning("select rows first")

    def plot_hist(self):
        new_df_0 = self.raw_data.loc[self.raw_data[self.out_col]==0 ][self.hist_col]
        new_df_1 = self.raw_data.loc[self.raw_data[self.out_col]==1 ][self.hist_col]
        hist0, _ = np.histogram(new_df_0)
        hist1, _ = np.histogram(new_df_1)
        plt.figure()
        self.unique_out = list(map(str,self.raw_data[self.out_col].unique().tolist()))
        fig2 = sns.distplot(new_df_0, color="blue", label=self.unique_out[0])
        fig2 = sns.distplot(new_df_1, color="red", label=self.unique_out[1])
        st.pyplot()

    def plot_hist_m(self, unique_values):
        new_df = [self.raw_data.loc[self.raw_data[self.out_col]==x ][self.hist_col] for x in unique_values]
        H = []
        for x in new_df:
            hist, _ = np.histogram(x)
            H.append(hist)

        plt.figure()
        self.unique_out = list(map(str,self.raw_data[self.out_col].unique().tolist()))
        for i,h in enumerate(H):
            fig2 = sns.distplot(h, color="blue", label=self.unique_out[i])

        st.pyplot()

    def plot_corr_matrix(self):
        plt.figure()
        sns.heatmap(self.raw_data\
                    .select_dtypes(exclude='object')\
                    .corr(self.corr_method),
                    cmap="YlGnBu")
        st.pyplot()

    def plot_pca(self):
        st.subheader("PCA")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_res = self.raw_data.drop(self.out_col, axis=1)
        y_res = self.raw_data[self.out_col]
        principalComponents = pca.fit_transform(X_res)

        principalDf = pd.DataFrame(data = principalComponents,
                               columns = ['principal component 1',
                                          'principal component 2'])
        finalDf = pd.concat([principalDf,
                            y_res],
                            axis = 1)
        plt.figure()
        graph = alt.Chart(finalDf)\
                    .mark_circle()\
                    .encode(x='principal component 1',
                            y='principal component 2',
                            color=self.out_col+':N')
        st.write(graph)


    def routine(self, prob_type, unique_values):
        self.get_2_axis_viz()
        self.visualize_2_axis()
        self.get_scatter_matrix_rows()
        self.scatter_matrix()
        if prob_type == "classification":
            self.get_hist_col()
            try:
                self.plot_hist_m()
            except:
                pass
        self.get_corr()
        self.plot_corr_matrix()
        self.plot_pca()


if __name__ == "__main__":
    session_state = SessionState.get(raw_data=pd.read_csv("Churn_Modelling.csv", nrows=10000),
                                    out_col="Exited")#change this bro)
    ep = exploration_page(session_state)
    ep.routine()
