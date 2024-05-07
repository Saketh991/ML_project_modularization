import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap.umap_ as umap
from mpl_toolkits.mplot3d import Axes3D
import os

def load_data(filepath):
   
    return pd.read_csv(filepath)

def save_plot(figure, filename, directory='plots'):
    
    path = os.path.join(directory, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path)
    plt.close(figure)  # Close the plot to free up memory

def univariate_analysis(data):
    """ Perform univariate analysis and save all histograms in one PNG file. """
    num_columns = len(data.columns)
    fig, axes = plt.subplots(nrows=num_columns, figsize=(10, 5 * num_columns))
    for i, column in enumerate(data.columns):
        sns.histplot(data[column], kde=True, ax=axes[i])
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
    save_plot(fig, 'histograms.png')

def bivariate_analysis(data, target):
    """ Perform bivariate analysis  """
    num_features = len(data.columns) - 1  # Exclude the target from the features count
    fig_scatter, axes_scatter = plt.subplots(nrows=num_features, figsize=(10, 5 * num_features))
    fig_box, axes_box = plt.subplots(nrows=num_features, figsize=(10, 5 * num_features))

    for i, column in enumerate(data.columns):
        if column != target:
            # Create scatter plot for each feature against the target
            sns.scatterplot(x=data[column], y=data[target], ax=axes_scatter[i])
            axes_scatter[i].set_title(f'Scatter Plot of {column} vs {target}')
            axes_scatter[i].set_xlabel(column)
            axes_scatter[i].set_ylabel(target)

            # Create box plot for each feature against the target
            sns.boxplot(x=data[target], y=data[column], ax=axes_box[i])
            axes_box[i].set_title(f'Box Plot of {column} vs {target}')
            axes_box[i].set_xlabel(target)
            axes_box[i].set_ylabel(column)

    # Adjust layout, save the plots, and close the figures to free up memory
    fig_scatter.tight_layout()
    fig_scatter.savefig('scatter_plots.png')
    plt.close(fig_scatter)

    fig_box.tight_layout()
    fig_box.savefig('box_plots.png')
    plt.close(fig_box)
    

def visualize_tsne(data):
    """ Visualize data using t-SNE in 2D and 3D and save plots. """
    tsne_2d = TSNE(n_components=2, perplexity=30, learning_rate=200)
    tsne_3d = TSNE(n_components=3, perplexity=30, learning_rate=200)
    
    result_2d = tsne_2d.fit_transform(data)
    result_3d = tsne_3d.fit_transform(data)

    # Plotting and saving 2D t-SNE
    fig_2d, ax_2d = plt.subplots(figsize=(8, 6))
    ax_2d.scatter(result_2d[:, 0], result_2d[:, 1], alpha=0.5)
    ax_2d.set_title('2D t-SNE')
    ax_2d.set_xlabel('Component 1')
    ax_2d.set_ylabel('Component 2')
    save_plot(fig_2d, 'tsne_2d.png')

    # Plotting and saving 3D t-SNE
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.scatter(result_3d[:, 0], result_3d[:, 1], result_3d[:, 2], alpha=0.5)
    ax_3d.set_title('3D t-SNE')
    save_plot(fig_3d, 'tsne_3d.png')

def visualize_umap(data):
    """ Visualize data using UMAP in 2D and 3D and save plots. """
    reducer_2d = umap.UMAP(n_neighbors=15, n_components=2)
    reducer_3d = umap.UMAP(n_neighbors=30, n_components=3)
    
    embedding_2d = reducer_2d.fit_transform(data)
    embedding_3d = reducer_3d.fit_transform(data)

    # Plotting and saving 2D UMAP
    fig_2d, ax_2d = plt.subplots(figsize=(8, 6))
    ax_2d.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.5)
    ax_2d.set_title('2D UMAP')
    ax_2d.set_xlabel('Component 1')
    ax_2d.set_ylabel('Component 2')
    save_plot(fig_2d, 'umap_2d.png')

    # Plotting and saving 3D UMAP
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], alpha=0.5)
    ax_3d.set_title('3D UMAP')
    save_plot(fig_3d, 'umap_3d.png')

if __name__ == "__main__":
    df = load_data('data/cleaned_data.csv')
    print("Loaded the data into visualization.py")
    univariate_analysis(df.drop('class', axis=1))
    bivariate_analysis(df, 'class')
    visualize_tsne(df.drop('class', axis=1))
    visualize_umap(df.drop('class', axis=1))
