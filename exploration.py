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
        self.box_col = None
        self.scatter_rows = []
        self.corr_method = 'pearson'
        self.problem_type = session_state.problem_type


    def get_2_axis_viz(self):
        st.subheader("Joint plot")
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

    def joint_plot(self):
        try:
            sns.jointplot(x=self.raw_data[self.x_axis], y=self.raw_data[self.y_axis], kind="kde", palette="Blues")
        except:
            try:
                sns.catplot(x=self.x_axis, y=self.y_axis, kind="swarm", data=self.raw_data, palette="Blues")
            except:
                st.error("something is wrong, please chose another column")
        st.pyplot()


    def scatter_matrix(self, prob_type):
        if len(self.scatter_rows) != 0:
            if prob_type == "classification":
                g = sns.pairplot(self.raw_data[self.scatter_rows+[self.out_col]],
                                vars = self.scatter_rows,
                                hue = self.out_col,
                                diag_kind = 'hist', palette="Blues")
                st.pyplot(g)
            else:
                g = sns.pairplot(self.raw_data[self.scatter_rows+[self.out_col]],
                                vars = self.scatter_rows,
                                diag_kind = 'hist', palette="Blues")
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

    def plot_hist_m(self):
        self.unique_out = list(map(str,self.raw_data[self.out_col].unique().tolist()))

        plt.figure()
        for i,x in enumerate(self.unique_out):
            new_df = self.raw_data.loc[self.raw_data[self.out_col]==int(float(x))][self.hist_col]
            sns.distplot(new_df, label=self.unique_out[i])
        st.pyplot()

    def get_box_col(self):
        st.subheader("Box plot")
        self.box_col = st.selectbox('Which feature?',
                                    self.raw_data.drop(self.out_col, axis=1)\
                                    .columns)

    def plot_box_altair(self):
        graph = alt.Chart(self.raw_data).mark_boxplot().encode(x = self.box_col, y = self.out_col).properties(width=500,height=500)
        st.write(graph)

    def plot_box_seaborn(self):
        plt.figure()
        if self.raw_data[self.box_col].nunique()<7:
            sns.boxplot(x = self.box_col, y = self.out_col, data=self.raw_data)
            st.pyplot()
        else:
            st.warning("feature contain too much unique values \n please choose another column")

    def plot_corr_matrix(self):
        plt.subplots(figsize=(20,15))
        sns.heatmap(self.raw_data\
                    .select_dtypes(exclude='object')\
                    .corr(self.corr_method),
                    cmap="Blues")
        st.pyplot()


    def plot_pca(self):
        st.subheader("PCA")
        st.warning("This will ignore non numerical columns")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_res = self.raw_data.select_dtypes(exclude='object').drop(self.out_col, axis=1)
        y_res = self.raw_data[self.out_col]
        principalComponents = pca.fit_transform(X_res)

        principalDf = pd.DataFrame(data = principalComponents,
                               columns = ['principal component 1',
                                          'principal component 2'])
        finalDf = pd.concat([principalDf,
                            y_res],
                            axis = 1)
        """
        plt.figure()
        graph = alt.Chart(finalDf)\
                    .mark_circle()\
                    .encode(x='principal component 1',
                            y='principal component 2',
                            color=self.out_col+':N')\
                    .properties(width=500,
                                height=500)
        st.write(graph)
        """
        plt.figure()
        sns.scatterplot(x="principal component 1", y="principal component 2", hue=self.out_col, palette="Blues",data=finalDf)
        st.pyplot()


    def routine(self, prob_type):
        self.get_2_axis_viz()
        self.joint_plot()
        self.get_scatter_matrix_rows()
        try:
            self.scatter_matrix(prob_type)
        except:
            self.scatter_matrix(prob_type)
            #st.warning("something is wrong, can't plot scatter matrix")

        if prob_type == "classification":
            self.get_hist_col()
            self.plot_hist_m()
        else:
            self.get_box_col()
            self.plot_box_seaborn()
        self.get_corr()
        self.plot_corr_matrix()
        self.plot_pca()


if __name__ == "__main__":
    session_state = SessionState.get(raw_data=pd.read_csv("Churn_Modelling.csv", nrows=10000),
                                    out_col="Exited")#change this bro)
    ep = exploration_page(session_state)
    ep.routine()
