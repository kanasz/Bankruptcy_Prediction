import pandas as pd
from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def create_box_plots(s, filename, thresholds, experiment_name,column):
    print(filename)
    df_results = pd.read_csv(filename, sep=",")
    print(df_results.columns)
    #df_results.columns = ['id', 'sector', 'year', 'run', 'true_class', column, 'epochs', 'thread_id']
    df_geometric_means = pd.DataFrame()

    lst_years = list(set(df_results['year']))
    lst_sectors = list(set(df_results['sector']))
    lst_runs = list(set(df_results['run']))
    lst_sectors.sort()
    lst_runs.sort()
    lst_years.sort()
    thresholds.sort()

    print(lst_sectors)

    grouped_by_sector_year = df_results.groupby(
        [df_results.sector, df_results.year, df_results.run])

    for sector in lst_sectors:
        for year in lst_years:

            for threshold in thresholds:
                lst_sector_year_gm = []

                for run in lst_runs:
                    grouped_data = (grouped_by_sector_year.get_group((sector, year, run)))
                    predicted_class = [1 if e > threshold else 0 for e in
                                       grouped_data[column]]

                    gm = (geometric_mean_score(grouped_data.true_class, predicted_class, average='binary'))
                    lst_sector_year_gm.append(gm)

                df_rows = pd.DataFrame();

                df_rows["Geom_Mean"] = lst_sector_year_gm
                df_rows["Sector"] = sector
                df_rows["Year"] = (str(year))
                df_rows["Threshold"] = threshold
                df_rows["Sector_Year"] = sector + "_" + str(year)

                df_geometric_means = pd.concat([df_geometric_means, df_rows])

    plt.figure()
    ax = sns.boxplot(x="Year", y="Geom_Mean", hue="Threshold", data=df_geometric_means, palette="Set1", width=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set(xlabel='Year', ylabel='Value', title='Geometric Mean for:\n' + s, ylim=(-0.1, 1.1))
    plt.savefig("Results/Plots/" + s.upper() + "_Geom_Mean_" + experiment_name + "_" + column + '.png', dpi=199)
    plt.close()
    data = df_geometric_means.groupby(['Year', 'Threshold'])['Geom_Mean'].describe()
    data.to_csv("Results/Descriptions/" + s.upper() + "_" + experiment_name + "_" + column + "_description.csv")

    df_final_statistics = pd.DataFrame()

    for sector in lst_sectors:
        for year in lst_years:
            for threshold in thresholds:
                grouped_data = df_geometric_means[(df_geometric_means['Sector'] == sector)
                                                  & (df_geometric_means['Year'] == str(year))
                                                  & (df_geometric_means['Threshold'] == threshold)
                                                  ]

                df_rows = pd.DataFrame()
                df_rows['sector'] = sector
                df_rows['year'] = year
                df_rows['threshold'] = threshold
                df_rows['mean_geom_mean'] = np.mean(grouped_data.Geom_Mean)
                df_rows['std_geom_mean'] = np.std(grouped_data.Geom_Mean)

                df_final_statistics = df_final_statistics.append({
                    'sector': sector,
                    'year': year,
                    'threshold': threshold,
                    'mean': np.mean(grouped_data.Geom_Mean),
                    'std': np.std(grouped_data.Geom_Mean)

                }, ignore_index=True)

    # df_final_statistics = df_final_statistics[["year", "threshold", "mean", "std"]]
    # df_final_statistics.to_csv('final_statistics.csv')
    dict_result = {
        "geom_means": df_geometric_means
    }
    return dict_result


def plot_confusion_matrixces(sector,experiment_name,thresholds,encoders):
    for i in range(encoders):
        filename =  './results/'+sector + '/' +experiment_name + "_" + str(i) + "_result.csv"
        df_results = pd.read_csv(filename, sep=",")

        df_results.columns = ['id', 'sector', 'year', 'run', 'true_class', 'reconstruction_error', 'epochs', 'thread_id']
        df_geometric_means = pd.DataFrame()

        lst_years = list(set(df_results['year']))
        lst_sectors = list(set(df_results['sector']))
        lst_runs = list(set(df_results['run']))
        lst_sectors.sort()
        lst_runs.sort()
        lst_years.sort()
        thresholds.sort()

        grouped_by_sector_year = df_results.groupby(
            [df_results.sector, df_results.year, df_results.run])

        for sector in lst_sectors:
            for year in lst_years:
                for run in lst_runs:

                    for threshold in thresholds:
                        grouped_data = (grouped_by_sector_year.get_group((sector, year, run)))
                        predicted_class = [1 if e > threshold else 0 for e in
                                           grouped_data.reconstruction_error]
                        cf = confusion_matrix(grouped_data.true_class, predicted_class)
                        print(cf)
        #print(df_results.head())

    return

def plot_gemetric_means(sector,experiment_name,thresholds):
    #thresholds = [0.0008, 0.0009, 0.001, 0.0015, 0.002, 0.003]
    #experiment_name = 'Deep_Dropout_Autoencoder_1_B_MIN_MAX_Scaler_1_300_Epochs'
    #sector, filename, thresholds, experiment_name
    create_box_plots(sector, './results/'+sector + '/' + experiment_name + '_result.csv',
                             thresholds,
                             experiment_name,'')
    return


def plot_geometric_means_for_aggregated_data(experiment_name,sector,thresholds,column):
    # experiment_name = 'Deep_Dropout_Autoencoder_1_B_MIN_MAX_Scaler_1_300_Epochs'
    # sector, filename, thresholds, experiment_name
    create_box_plots(sector, './results/' + sector + '/' + experiment_name + '_aggregated_results.csv',
                     thresholds,
                     experiment_name, column)
    return

def plot_geometric_means_by_years(experiment_name, sector, thresholds,save_plot = False,autoencoders = 3,positive_class_sum = 3):
    df = pd.DataFrame()
    columns = ['id', 'sector', 'year', 'run', 'true_class', 'reconstruction_error', 'epochs', 'thread_id']
    print(autoencoders)
    for i in range(autoencoders):
        df_y = pd.read_csv('./results/' + sector + '/' + experiment_name + '_'+str(i)+'_result.csv')
        df_y.columns = columns
        df_y["predicted_class_" + str(i)] = [1 if e > thresholds[i] else 0 for e in df_y.reconstruction_error]
        #df.append(df_y)\
        if df.empty==True:
            df = df_y
        else:
            df["predicted_class_" + str(i)] = df_y["predicted_class_" + str(i)]
    print(df.iloc[:,0:8])
    df["sum"]=df.iloc[:,8:].sum(axis = 1)
    #df.to_csv('test_1.csv')

    #df.to_csv('./results/DF.csv')
    #print(df['sum'])

    df_finale = pd.DataFrame()
    df_finale['sum'] = df['sum']
    df_finale['predicted_class']=0

    df_finale[df_finale['sum']>=positive_class_sum]=1
    df['predicted_class'] = df_finale.predicted_class
    df_geometric_means = pd.DataFrame()

    lst_years = list(set(df['year']))
    lst_sectors = list(set(df['sector']))
    lst_runs = list(set(df['run']))
    lst_sectors.sort()
    lst_runs.sort()
    lst_years.sort()
    thresholds.sort()

    grouped_by_sector_year = df.groupby(
        [df.sector, df.year, df.run])

    for sector in lst_sectors:
        for year in lst_years:
                lst_sector_year_gm = []
                for run in lst_runs:
                    grouped_data = (grouped_by_sector_year.get_group((sector, year, run)))
                    gm = (geometric_mean_score(grouped_data.true_class, grouped_data.predicted_class, average='binary'))
                    lst_sector_year_gm.append(gm)

                df_rows = pd.DataFrame();

                df_rows["Geom_Mean"] = lst_sector_year_gm
                df_rows["Sector"] = sector
                df_rows["Year"] = (str(year))
                df_rows["Sector_Year"] = sector + "_" + str(year)

                df_geometric_means = pd.concat([df_geometric_means, df_rows])
    if save_plot==True:
        plt.figure()
        ax = sns.boxplot(x="Year", y="Geom_Mean", data=df_geometric_means, palette="Set1", width=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set(xlabel='Year', ylabel='Value', title='Geometric Mean for:\n' + sector, ylim=(-0.1, 1.1))
        plt.savefig("Results/Plots/" + sector.upper() + "_Geom_Mean_By_Year_" + experiment_name + '.png', dpi=199)
        plt.close()
    data = df_geometric_means.groupby(['Year'])['Geom_Mean'].describe()
    print(data)
    print(np.mean(data['mean']))
    data.to_csv("Results/Descriptions/" + sector.upper() + "_" + experiment_name + "_description_by_year.csv")
    #print(data)
    df_final_statistics = pd.DataFrame()
    #print()
    return np.median(data['mean']), np.mean(data['mean'])


#plot_geometric_means_by_years('Ensemble_Autoencoder_11_MIN_MAX_2_10_ENCODERS_100_EPOCHS_20_FEATURES','Agriculture',[0.56,0.02,0.2], save_plot=True)
#plot_geometric_means_by_years('Ensemble_Autoencoder_11_STANDARD_3_ENCODERS_BY_YEARS_100_EPOCHS_SORTED','Agriculture',[0.15,0.15,0.15])
#plot_geometric_means_by_years('Ensemble_Autoencoder_11_STANDARD_3_ENCODERS_BY_YEARS_100_EPOCHS_SORTED','Agriculture',[0.1,0.1,0.1])

#possible solution
#plot_geometric_means_by_years('Ensemble_Autoencoder_12_STANDARD_35_ENCODERS_100_EPOCHS_SORTED','Agriculture',[0.15,0.15,0.15,0.16,0.17]*7,True,35,30)

#possible solution
#plot_geometric_means_by_years('Ensemble_Autoencoder_12_STANDARD_35_ENCODERS_100_EPOCHS_SORTED','Agriculture',[0.20,0.20,0.20,0.20,0.20]*7,True,35,30)


#plot_geometric_means_by_years('Ensemble_Autoencoder_12_STANDARD_35_ENCODERS_100_EPOCHS_SORTED','Agriculture',[0.20,0.20,0.25,0.25,0.30]*7,True,35,30)

#plot_geometric_means_for_aggregated_data('Ensemble_Autoencoder_12_STANDARD_35_ENCODERS_100_EPOCHS_SORTED',
#                                         'Agriculture',
#                                         [0.15,0.16,0.17,0.18,0.19,0.2])

