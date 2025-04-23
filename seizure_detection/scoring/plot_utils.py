import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns

def compute_metrics(df, label_column='true_label', output_column="predicted_output",group_column='group',
                    roc=True,
                    pr=True):
    # Store results
    roc_data, pr_data = {}, {}

    for group, group_df in df.groupby(group_column):
        y_true = group_df[label_column].values
        y_scores = group_df[output_column].values

        if roc:

            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            roc_data[group] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        if pr:
            # Compute PR curve and Average Precision
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            avg_precision = average_precision_score(y_true, y_scores)
            pr_data[group] = {'precision': precision, 'recall': recall, 'avg_precision': avg_precision}

    return roc_data, pr_data


def interpolate_and_average(metric_data, x_key, y_key, num_points):
    """
    Interpolates metrics to a common scale and averages over groups.

    :param metric_data: Dictionary of group-specific metrics.
    :param x_key: Key for the x-axis metric (e.g., 'fpr' or 'recall').
    :param y_key: Key for the y-axis metric (e.g., 'tpr' or 'precision').
    :param num_points: Number of points for interpolation.
    :return: Average x and y values over groups.
    """
    x_common = np.linspace(0, 1, num_points)  # Common grid for interpolation
    interpolated_y = []  # Store interpolated y-values

    for group_data in metric_data.values():
        x_vals = group_data[x_key]
        y_vals = group_data[y_key]
        y_interp = np.interp(x_common, x_vals, y_vals)
        y_interp[0] = 0.0
        interpolated_y.append(y_interp)

    avg_y = np.mean(interpolated_y, axis=0)
    avg_y[-1] = 1.0
    std_y = np.std(interpolated_y, axis=0)
    return x_common, avg_y, std_y

def plot_n_average_curves(df_list, label_column='true_label', output_column="predicted_output", group_column='group',
                          model_names=None, num_points=100, bounds=False, fig_size=(6,6), save_file=None, LOSI=None,
                          lw=3):
    """
    Plot average ROC curves for multiple models over groups.
    Args:
        df_list:
        label_column:
        output_column:
        group_column:
        model_names:
        num_points:

    Returns:

    """
    # sns.set_palette("colorblind")
    cmap = sns.color_palette("colorblind")
    sns.set_palette(cmap)
    sns.set_style("whitegrid")
    plt.figure(figsize=fig_size)
    # generate a color map
    # colors = plt.cm.get_cmap('colorblind', len(df_list))
    colors = cmap
    if LOSI is None:
        LOSI = [False] * len(df_list)
    for i, df in enumerate(df_list):
        if LOSI[i]: # compute average over the estimators
            roc_data = {}
            for patient in df['patient'].unique():
                patient_df = df[df['patient'] == patient]
                roc_data_pat, _ = compute_metrics(patient_df,
                                              label_column=label_column,
                                              group_column='estimator',
                                              output_column=output_column,
                                              roc=True,
                                              pr=False
                                              )
                avg_fpr, avg_tpr, std_tpr = interpolate_and_average(roc_data_pat, 'fpr', 'tpr', num_points)
                roc_data[patient] = {'fpr': avg_fpr, 'tpr': avg_tpr, 'auc': np.mean([v['auc'] for v in roc_data_pat.values()])}
        else:
            roc_data, _ = compute_metrics(df,
                                          label_column=label_column,
                                          group_column=group_column,
                                          output_column=output_column,
                                          roc=True,
                                          pr=False
                                          )
        avg_fpr, avg_tpr, std_tpr = interpolate_and_average(roc_data, 'fpr', 'tpr', num_points)

        plt.plot(avg_fpr, avg_tpr, label=f'{model_names[i]} (AUC: '
                                         f'{np.mean([v["auc"] for v in roc_data.values()]):.2f})', color=colors[i],
                 lw=lw)

        if bounds:
            plt.fill_between(avg_fpr, avg_tpr - std_tpr, avg_tpr + std_tpr, alpha=0.3, color=colors[i])

    # plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.8)

    plt.xlabel('False Positive Rate', labelpad=15)
    plt.ylabel('True Positive Rate', labelpad=15)
    # plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
    return


def plot_average_curves(df1, df2, label_column='true_label', group_column='group', model1_name='Model 1',
                        model2_name='Model 2', num_points=100):
    """
    Plot average ROC and PR curves for two models over groups.

    :param df1: DataFrame for model 1 with 'predicted_output', 'group', and 'actual_output' columns.
    :param df2: DataFrame for model 2 with 'predicted_output', 'group', and 'actual_output' columns.
    :param label_column: Name of the column containing actual labels (default: 'actual_output').
    """

    # Compute metrics for each model
    roc_data_1, pr_data_1 = compute_metrics(df1, label_column, group_column)
    roc_data_2, pr_data_2 = compute_metrics(df2, label_column, group_column)

    # Interpolate and average metrics
    avg_fpr_1, avg_tpr_1, _ = interpolate_and_average(roc_data_1, 'fpr', 'tpr', num_points)
    avg_fpr_2, avg_tpr_2, _ = interpolate_and_average(roc_data_2, 'fpr', 'tpr', num_points)
    avg_tpr_1[0] = 0.0
    avg_tpr_2[0] = 0.0
    avg_tpr_2[-1] = 1.0
    avg_tpr_1[-1] = 1.0

    # avg_recall_1, avg_precision_1 = interpolate_and_average(pr_data_1, 'recall', 'precision', num_points)
    # avg_recall_2, avg_precision_2 = interpolate_and_average(pr_data_2, 'recall', 'precision', num_points)
    # avg_precision_1[0] = 1.0
    # avg_precision_2[0] = 1.0
    # avg_precision_1[-1] = 0.0
    # avg_precision_2[-1] = 0.0   # Set the first and last points to (0, 1) for PR curves

    # Plot ROC Curves
    plt.figure(figsize=(14, 14))
    plt.plot(avg_fpr_1, avg_tpr_1, label=f'{model1_name} (AUC: {np.mean([v["auc"] for v in roc_data_1.values()]):.2f})')
    plt.plot(avg_fpr_2, avg_tpr_2, label=f'{model2_name} (AUC: {np.mean([v["auc"] for v in roc_data_2.values()]):.2f})')
    # plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    plt.legend(loc='lower right')
    #
    # # Plot PR Curves
    # plt.subplot(1, 2, 2)
    # plt.plot(avg_recall_1, avg_precision_1,
    #          label=f'{model1_name} (Avg Precision: {np.mean([v["avg_precision"] for v in pr_data_1.values()]):.2f})')
    # plt.plot(avg_recall_2, avg_precision_2,
    #          label=f'{model2_name} (Avg Precision: {np.mean([v["avg_precision"] for v in pr_data_2.values()]):.2f})')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_curves_across_groups(df1, df2, label_column='actual_output', model1_name='Model 1', model2_name='Model 2'):
    """
    Plot ROC and PR curves for two models by computing metrics across all groups (globally).

    :param df1: DataFrame for model 1 with 'predicted_output', 'group', and 'actual_output' columns.
    :param df2: DataFrame for model 2 with 'predicted_output', 'group', and 'actual_output' columns.
    :param label_column: Name of the column containing actual labels (default: 'actual_output').
    :param model1_name: Name of the first model (default: 'Model 1').
    :param model2_name: Name of the second model (default: 'Model 2').
    """

    def compute_global_metrics(df):
        """
        Compute global ROC and PR metrics for the entire dataset.

        :param df: Input DataFrame with 'predicted_output' and actual labels.
        :return: fpr, tpr, auc, recall, precision, avg_precision
        """
        y_true = df[label_column]
        y_scores = df['predicted_output']

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Compute PR curve and Average Precision
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)

        return fpr, tpr, roc_auc, recall, precision, avg_precision

    # Compute global metrics for each model
    fpr1, tpr1, auc1, recall1, precision1, avg_prec1 = compute_global_metrics(df1)
    fpr2, tpr2, auc2, recall2, precision2, avg_prec2 = compute_global_metrics(df2)

    # Plot ROC Curves
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr1, tpr1, label=f'{model1_name} (AUC: {auc1:.2f})')
    plt.plot(fpr2, tpr2, label=f'{model2_name} (AUC: {auc2:.2f})')
    # plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    plt.legend(loc='lower right')

    # Plot PR Curves
    plt.subplot(1, 2, 2)
    plt.plot(recall1, precision1, label=f'{model1_name} (Avg Precision: {avg_prec1:.2f})')
    plt.plot(recall2, precision2, label=f'{model2_name} (Avg Precision: {avg_prec2:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()