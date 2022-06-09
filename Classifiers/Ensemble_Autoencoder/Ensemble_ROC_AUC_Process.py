from sklearn.metrics import roc_auc_score
import pandas as pd

from Classifiers.Ensemble_Autoencoder.Ensemple_Thresholds_Tuning_Data_Process import get_dataframes, get_tuned_parameters, \
    evaluate_by_thresholds_and_active_autoencoders, evaluate_by_threshold, aggregate_data_for_encoders


def aggregate_data(df):
    lst_years =  list(set(df['year']))
    lst_runs = list(set(df['run']))
    lst_runs.sort()
    lst_years.sort()
    df_geometric_means = pd.DataFrame()
    for year in lst_years:
        lst_sector_year_gm = []
        for run in lst_runs:
            # grouped_data = (grouped_by_sector_year.get_group((year, run)))
            grouped_data = df.loc[(df['year'] == year) & (df['run'] == run)]
            #gm = (roc_auc_score(grouped_data.true_class, grouped_data.reconstruction_error_01))
            gm = (roc_auc_score(grouped_data.true_class, grouped_data.predicted_class))
            # print(gm)
            lst_sector_year_gm.append(gm)
        df_rows = pd.DataFrame();
        df_rows["ROC_AUC"] = lst_sector_year_gm
        #df_rows["Sector"] = self.experiment_sector
        df_rows["Year"] = (str(year))
        df_geometric_means = pd.concat([df_geometric_means, df_rows])
    return df_geometric_means


def evaluate_roc_auc_for_autoencoders_and_thresholds(sector,experiment_name, autoencoders):
    df_y = get_dataframes(sector, experiment_name, autoencoders)
    tuned_params = get_tuned_parameters(experiment_name)
    eval_data = evaluate_by_thresholds_and_active_autoencoders(tuned_params, df_y)
    aggregated_data = aggregate_data(eval_data)
    statistics = aggregated_data.groupby(['Year'])['ROC_AUC'].describe()
    statistics.to_csv(
        "Results/statistics/{}_ROC_AUC_FOR_{}_ENCODERS_AND_{}_THRESHOLDS.csv".format(experiment_name, autoencoders,
                                                                                       autoencoders))
    print(statistics)

def evaluate_roc_auc_for_autoencoders_and_single_threshold(sector,experiment_name, autoencoders, threshold):
    df_y = get_dataframes(sector, experiment_name, autoencoders)
    tuned_params = get_tuned_parameters(experiment_name)
    eval_data = evaluate_by_thresholds_and_active_autoencoders(tuned_params, df_y)
    aggregated_data = aggregate_data(eval_data)
    statistics = aggregated_data.groupby(['Year'])['ROC_AUC'].describe()
    statistics.to_csv(
        "Results/statistics/{}_ROC_AUC_FOR_{}_ENCODERS_AND_{}_THRESHOLDS.csv".format(experiment_name, autoencoders,
                                                                                       autoencoders))
    print(statistics)

def evaluate_roc_auc_for_autoencoders_and_single_threshold(sector, experiment_name, autoencoders, threshold):
    df_aggregate_data = aggregate_data_for_encoders(sector, experiment_name, autoencoders)
    eval_data = evaluate_by_threshold(df_aggregate_data, threshold)
    aggregated_data = aggregate_data(eval_data)
    statistics = aggregated_data.groupby(['Year'])['ROC_AUC'].describe()
    statistics.to_csv(
        "Results/statistics/{}_ROC_AUC_FOR_{}_AGGREGATED_ENCODERS_AND_ONE_THRESHOLDS.csv".format(experiment_name,autoencoders))

    return

SECTOR = "AGRICULTURE"

AUTOENCODERS = 100
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_Y-2_AND_Y-1_ENCODER".format(SECTOR,AUTOENCODERS)
EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#evaluate_roc_auc_for_autoencoders_and_thresholds(SECTOR, EXPERIMENT_NAME, AUTOENCODERS)

SECTOR = "CONSTRUCTION"
AUTOENCODERS = 100
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_Y-2_AND_Y-1_ENCODER".format(SECTOR,AUTOENCODERS)
EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#evaluate_roc_auc_for_autoencoders_and_thresholds(SECTOR, EXPERIMENT_NAME, AUTOENCODERS)

SECTOR = "MANUFACTURE"
AUTOENCODERS = 66
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_Y-2_AND_Y-1_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#evaluate_roc_auc_for_autoencoders_and_thresholds(SECTOR, EXPERIMENT_NAME, AUTOENCODERS)

SECTOR = "RETAIL"
AUTOENCODERS = 66
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_Y-2_AND_Y-1_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#evaluate_roc_auc_for_autoencoders_and_thresholds(SECTOR, EXPERIMENT_NAME, AUTOENCODERS)







SECTOR = "AGRICULTURE"
AUTOENCODERS = 33
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_Y-1_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#evaluate_roc_auc_for_autoencoders_and_single_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.0913899999999966)

SECTOR = "CONSTRUCTION"
AUTOENCODERS = 33
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_Y-1_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#evaluate_roc_auc_for_autoencoders_and_single_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.0542349999999982)

SECTOR = "MANUFACTURE"
AUTOENCODERS = 33
#_
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_Y-1_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#evaluate_roc_auc_for_autoencoders_and_single_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.057009999999998)

SECTOR = "RETAIL"
AUTOENCODERS = 33
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_Y-1_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#evaluate_roc_auc_for_autoencoders_and_single_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.0457299999999985)



SECTOR = "AGRICULTURE"
AUTOENCODERS = 1
EXPERIMENT_NAME = "{}_16_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_60_RUNS_20_STANDARD_1_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#evaluate_roc_auc_for_autoencoders_and_single_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.0913899999999966)
#evaluate_roc_auc_for_autoencoders_and_single_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.121809999999995)

SECTOR = "CONSTRUCTION"
AUTOENCODERS = 1
EXPERIMENT_NAME = "{}_16_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_60_RUNS_20_STANDARD_1_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#evaluate_roc_auc_for_autoencoders_and_single_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.101244999999996)

SECTOR = "MANUFACTURE"
AUTOENCODERS = 1
#_
EXPERIMENT_NAME = "{}_16_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_60_RUNS_20_STANDARD_1_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#evaluate_roc_auc_for_autoencoders_and_single_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.0567399999999981)

SECTOR = "RETAIL"
AUTOENCODERS = 1
EXPERIMENT_NAME = "{}_16_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_60_RUNS_20_STANDARD_1_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
#EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER".format(SECTOR,AUTOENCODERS)
evaluate_roc_auc_for_autoencoders_and_single_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.0784149999999972)