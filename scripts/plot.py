import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def box_plot (df):
    df.boxplot()
    plt.title('Box plot to detect outliers')
    plt.show()

def scatter_plot(df, column1,column2):
    sns.scatterplot(x=column1, y=column2, data=df)
    plt.title('Correlation Scatter Plot between ER and CTR')
    plt.show()

def density_plot(df, column1, column2):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=df[column1], y=df[column2], cmap="Reds", shade=True, thresh=0.05)
    plt.title(f'Joint Density Plot between {column1} and {column2}')
    plt.xlabel(f'{column1}')
    plt.ylabel(f'{column2}')
    plt.show()

