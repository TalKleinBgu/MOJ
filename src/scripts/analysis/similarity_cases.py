import pandas as pd
import matplotlib.pyplot as plt


def plot_comparison_column_distributions(df1_path, df2_path, label_filter=None, save_path='distribution_comparison.png'):
    """
    Plot the distribution of numeric columns for two dataframes and save the plot.

    Parameters:
    df1_path (str): Path to the first CSV file.
    df2_path (str): Path to the second CSV file.
    label_filter (float): If provided, filter rows where 'label' is less than this value.
    save_path (str): Path to save the resulting plot.

    """
    # Load the CSV files
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)

    # Dropping non-numeric columns for distribution plotting
    df1_numeric = df1.drop(columns=['Unnamed: 0', 'source', 'target'])
    df2_numeric = df2.drop(columns=['Unnamed: 0', 'source', 'target'])

    # Filter by label if label_filter is provided
    if label_filter is not None:
        df1_numeric = df1_numeric[df1_numeric['label'] < label_filter]
        df2_numeric = df2_numeric[df2_numeric['label'] < label_filter]

    # Renaming the columns in the new dataframe to match the original dataframe for comparison
    column_mapping = {
        'OBTAIN_WAY': 'OBTAIN_WAY_WEP',
        'HELD_WAY': 'HELD_WAY_WEP',
        'סוג עבירה': 'OFFENCE_TYPE'  # Assuming this corresponds to 'OFFENCE_TYPE' based on context
    }
    df2_numeric = df2_numeric.rename(columns=column_mapping)

    # Ensure we only plot columns that exist in both dataframes
    common_columns = df1_numeric.columns.intersection(df2_numeric.columns)

    # Plotting the distribution of each numeric column for both dataframes
    fig, axes = plt.subplots(len(common_columns), 1, figsize=(15, 30))

    for i, column in enumerate(common_columns):
        axes[i].hist(df1_numeric[column], bins=30, alpha=0.5, label='Model FE (label < 3)', color='blue')
        axes[i].hist(df2_numeric[column], bins=30, alpha=0.5, label='Tagged FE (label < 3)', color='red')
        axes[i].set_title(f'Distribution of {column}')
        axes[i].legend()

    plt.suptitle('Distribution Comparison of Each Numeric Column')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    plt.savefig(save_path)
    plt.show()

# Example usage
# plot_distributions('/mnt/data/case_pairs_model_fe.csv', '/mnt/data/case_pairs_tagged_fe.csv', label_filter=3, save_path='/mnt/data/distribution_comparison.png')
