import pandas as pd
from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
#from Classifier.Data_Process import RESULTS_DIRECTORY
from Classifiers.Ensemble_Autoencoder.Ensemble_Constants import RESULTS_DIRECTORY


def evaluate_by_thresholds_and_active_autoencoders(X,df_y):
    thresholds = []
    encoder_states = []
    for i in range(0, int(len(X) / 2)):
        thresholds.append(X[2 * i])

    for i in range(0, int(len(X) / 2)):
        encoder_states.append(X[(2 * i) + 1])
    df_res = pd.DataFrame()

    active_encoders = 0
    for i in range(len(thresholds)):
        if (encoder_states[i] == 1):
            active_encoders = active_encoders + 1
            if df_res.empty == True:
                df_res = df_y[i].copy()
            # povodna chyba
            # df_res["predicted_class_" + str(i)] = [1 if e > X[i] else 0 for e in self.df_y[i].reconstruction_error]
            # print(thresholds[i])
            df_res["predicted_class_" + str(i)] = [1 if e > thresholds[i] else 0 for e in
                                                   df_y[i].reconstruction_error]

    class_sum = (df_res.iloc[:, 8:].sum(axis=1))

    df_res["sum"] = class_sum
    df_finale = pd.DataFrame()

    df_finale['sum'] = df_res['sum']

    df_finale['predicted_class'] = 0
    df_finale[df_finale['sum'] >= (active_encoders / 2)] = 1
    df_res['predicted_class'] = df_finale.predicted_class

    #mean = self.get_mean(df_res)

    #print(df_res.columns)
    return df_res

def get_dataframes(sector, experiment_name, autoencoders):
    columns = ['id', 'sector', 'year', 'run', 'true_class', 'reconstruction_error', 'mean_absolute_error',
               'root_mean_squared_error', 'epochs', 'thread_id']
    df_y = []
    for i in range(autoencoders):
        df_y.append(pd.read_csv(
            './results/' + sector + '/' + experiment_name + '_{:0>2}'.format(
                i + 1) + '_result.csv'))
        df_y[i].columns = columns

        df_y[i] = df_y[i].drop(columns=['mean_absolute_error', 'root_mean_squared_error'])
    return df_y

def get_tuned_parameters(experiment_name):
    ch = np.loadtxt('./results/tuning/' + experiment_name + "_tuned.txt")
    for i in range(1, len(ch)):
        if i % 2 == 1:
            ch[i] = int(ch[i])
    return ch

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
            gm = (geometric_mean_score(grouped_data.true_class, grouped_data.predicted_class, average='binary'))
            # print(gm)
            lst_sector_year_gm.append(gm)
        df_rows = pd.DataFrame();
        df_rows["Geom_Mean"] = lst_sector_year_gm
        #df_rows["Sector"] = self.experiment_sector
        df_rows["Year"] = (str(year))
        df_geometric_means = pd.concat([df_geometric_means, df_rows])
    return df_geometric_means

def plot_geom_means_boxplots(sector,experiment_name,df,autoencoders):
    plt.figure()
    ax = sns.boxplot(x="Year", y="Geom_Mean", data=df, palette="Set1", width=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set(xlabel='Year', ylabel='Value', title='Geometric Mean for:\n' + sector, ylim=(-0.1, 1.1))
    plt.savefig("Results/Plots/" + experiment_name+ "_GEOM_MEAN_FOR_{}_ENCODERS_AND_{}_THRESHOLDS.png".format(autoencoders, autoencoders), dpi=199)
    plt.close()
    return


def aggregate_data_for_encoders(sector, experiment, encoders=30):
    df_final_result = pd.DataFrame()
    results_path = RESULTS_DIRECTORY + '/' + sector
    loaded_encoders = 0;
    #print(experiment)
    for index, filename in enumerate(os.listdir(results_path)):

        if loaded_encoders == encoders:
            break
        if not filename.startswith(experiment):
            continue
        if "AGGREGATED" in filename:
            continue

        loaded_encoders = loaded_encoders + 1
        filename_splits = filename.split('_')
        thread_id = (filename_splits[len(filename_splits) - 2])
        file_path = results_path + '/' + filename
        df_result = pd.read_csv(file_path,  index_col=0)
        df_result.columns = ['sector', 'year', 'run', 'true_class', 'reconstruction_error_' + str(thread_id),'mean_absolute_error'+ str(thread_id),'root_mean_squared_error'+ str(thread_id), 'epochs', 'thread_id']

        if df_final_result.empty:
            df_final_result = df_result[['sector', 'year', 'run', 'true_class', 'epochs', 'thread_id', 'reconstruction_error_' + str(thread_id)]]
        else:
            df_final_result['reconstruction_error_' + str(thread_id)] = df_result['reconstruction_error_' + str(thread_id)]

    median = df_final_result.iloc[:, 6:].median(axis=1)
    mean = df_final_result.iloc[:, 6:].mean(axis=1)
    #df_final_result["reconstruction_error_median"] = median
    #print(df_final_result)
    #print(mean)
    df_final_result["reconstruction_error_mean"] = mean
    return df_final_result


def create_boxplots_for_autoencoders_and_thresholds(sector,experiment_name, autoencoders):
    df_y = get_dataframes(sector, experiment_name, autoencoders)
    tuned_params = get_tuned_parameters(experiment_name)
    eval_data = evaluate_by_thresholds_and_active_autoencoders(tuned_params, df_y)
    aggregated_data = aggregate_data(eval_data)
    plot_geom_means_boxplots(sector, experiment_name, aggregated_data,autoencoders)
    statistics = aggregated_data.groupby(['Year'])['Geom_Mean'].describe()
    statistics.to_csv(
        "Results/statistics/{}_GEOM_MEAN_FOR_{}_ENCODERS_AND_{}_THRESHOLDS.csv".format(experiment_name, autoencoders,
                                                                                       autoencoders))

def evaluate_by_threshold(df,threshold):
    df["predicted_class"] = [1 if e > threshold else 0 for e in
                                           df.reconstruction_error_mean]
    return df

def get_best_threshold(sector, experiment_name, autoencoders, thresholds):
    df_aggregate_data = aggregate_data_for_encoders(sector, experiment_name, autoencoders)
    best_threshold = 0
    best_mean = 0
    means = []
    i = 0;
    for threshold in thresholds:
        eval_data = evaluate_by_threshold(df_aggregate_data, threshold)
        aggregated_data = aggregate_data(eval_data)
        statistics = aggregated_data.groupby(['Year'])['Geom_Mean'].describe()
        # print(statistics['mean'])

        mean = np.mean(statistics['mean'])
        means.append(mean)
        if(mean> best_mean):
            best_mean = mean
            best_threshold = threshold
            #print("{}/{} mean: {}, threshold: {}".format(i+1,len(thresholds),best_mean, best_threshold))
        i = i+ 1
    arg = np.argmax(means)

    return best_threshold, best_mean

def create_boxplots_for_aggregated_reconstruction_errors_and_one_threshold(sector, experiment_name, autoencoders, threshold):
    df_aggregate_data = aggregate_data_for_encoders(sector, experiment_name, autoencoders)
    eval_data = evaluate_by_threshold(df_aggregate_data, threshold)
    aggregated_data = aggregate_data(eval_data)
    statistics = aggregated_data.groupby(['Year'])['Geom_Mean'].describe()
    statistics.to_csv(
        "Results/statistics/{}_GEOM_MEAN_FOR_{}_AGGREGATED_ENCODERS_AND_ONE_THRESHOLDS.csv".format(experiment_name,autoencoders))
    plt.figure()
    ax = sns.boxplot(x="Year", y="Geom_Mean", data=aggregated_data, palette="Set1", width=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set(xlabel='Year', ylabel='Value', title='Geometric Mean for:\n' + sector, ylim=(-0.1, 1.1))
    plt.savefig("Results/Plots/" + experiment_name+ "_GEOM_MEAN_FOR_{}_AGGREGATED_ENCODERS_AND_ONE_THRESHOLD.png".format(autoencoders), dpi=199)
    plt.close()
    return
'''
SECTOR = "MANUFACTURE"
AUTOENCODERS = 1
EXPERIMENT_NAME = "{}_16_ENCODERS_01_EPOCHS_100_LR_0.01_FEATURES_60_RUNS_20_STANDARD_1_ENCODER".format(SECTOR,AUTOENCODERS)
#create_boxplots_for_autoencoders_and_thresholds(SECTOR, EXPERIMENT_NAME, AUTOENCODERS)
create_boxplots_for_aggregated_reconstruction_errors_and_one_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.0567399999999981)

SECTOR = "AGRICULTURE"
EXPERIMENT_NAME = "{}_16_ENCODERS_01_EPOCHS_100_LR_0.01_FEATURES_60_RUNS_20_STANDARD_1_ENCODER".format(SECTOR,AUTOENCODERS)
#create_boxplots_for_autoencoders_and_thresholds(SECTOR, EXPERIMENT_NAME, AUTOENCODERS)
create_boxplots_for_aggregated_reconstruction_errors_and_one_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.12180999999999544)

SECTOR = "CONSTRUCTION"
EXPERIMENT_NAME = "{}_16_ENCODERS_01_EPOCHS_100_LR_0.01_FEATURES_60_RUNS_20_STANDARD_1_ENCODER".format(SECTOR,AUTOENCODERS)
#create_boxplots_for_autoencoders_and_thresholds(SECTOR, EXPERIMENT_NAME, AUTOENCODERS)
create_boxplots_for_aggregated_reconstruction_errors_and_one_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.10124499999999628)

SECTOR = "RETAIL"
EXPERIMENT_NAME = "{}_16_ENCODERS_01_EPOCHS_100_LR_0.01_FEATURES_60_RUNS_20_STANDARD_1_ENCODER".format(SECTOR,AUTOENCODERS)
#create_boxplots_for_autoencoders_and_thresholds(SECTOR, EXPERIMENT_NAME, AUTOENCODERS)
create_boxplots_for_aggregated_reconstruction_errors_and_one_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.07841499999999721)
'''


SECTOR = "AGRICULTURE"
AUTOENCODERS = 33
EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_Y-1_ENCODER".format(SECTOR,AUTOENCODERS)
#create_boxplots_for_autoencoders_and_thresholds(SECTOR, EXPERIMENT_NAME, AUTOENCODERS)
#create_boxplots_for_aggregated_reconstruction_errors_and_one_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.141459999999994)

SECTOR = "CONSTRUCTION"
AUTOENCODERS = 33
EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_Y-1_ENCODER".format(SECTOR,AUTOENCODERS)
#create_boxplots_for_autoencoders_and_thresholds(SECTOR, EXPERIMENT_NAME, AUTOENCODERS)
#create_boxplots_for_aggregated_reconstruction_errors_and_one_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.0877599999999968)

SECTOR = "MANUFACTURE"
AUTOENCODERS = 33
EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_Y-1_ENCODER".format(SECTOR,AUTOENCODERS)
#create_boxplots_for_autoencoders_and_thresholds(SECTOR, EXPERIMENT_NAME, AUTOENCODERS)
#create_boxplots_for_aggregated_reconstruction_errors_and_one_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.0806199999999971)

SECTOR = "RETAIL"
AUTOENCODERS = 33
EXPERIMENT_NAME = "{}_01_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_Y-1_ENCODER".format(SECTOR,AUTOENCODERS)
#create_boxplots_for_autoencoders_and_thresholds(SECTOR, EXPERIMENT_NAME, AUTOENCODERS)
#create_boxplots_for_aggregated_reconstruction_errors_and_one_threshold(SECTOR, EXPERIMENT_NAME, AUTOENCODERS,0.143559999999994)
'''
'''
'''
SECTOR = 'RETAIL'
AUTOENCODERS = 1
EXPERIMENT_NAME = "{}_16_ENCODERS_{:0>2}_EPOCHS_100_LR_0.01_FEATURES_40_RUNS_20_STANDARD_Y-2_AND_Y-1_ENCODER".format(SECTOR,AUTOENCODERS)
lower = 0.01
upper = 0.16
n = 10000
length = upper-lower
delta = length / n
#print(delta)
thresholds = []
for threshold in np.arange(lower, upper, delta):
    thresholds.append(threshold)

print(get_best_threshold(SECTOR,EXPERIMENT_NAME,AUTOENCODERS,thresholds))
'''
#means = create_aggregated_boxplots(SECTOR, EXPERIMENT_NAME, AUTOENCODERS, thresholds)
#arg = np.argmax(means)
#maxx = np.max(means)
#print(arg)
#print(thresholds[arg])
#print(maxx)





