import multiprocessing
from multiprocessing.pool import Pool
from random import seed

from imblearn.metrics import geometric_mean_score
#from geneticalgorithm import geneticalgorithm as ga
from Classifiers.Ensemble_Autoencoder.GA import geneticalgorithm as ga
from Classifiers.Ensemble_Autoencoder.Ensemble_Plot_Functions import plot_geometric_means_by_years
import  numpy as np
import pandas as pd

seed(2)
np.random.seed(2)

class ThresholdTuner:
    def __init__(self, experiment_name, experiment_sector, autoencoders, max_num_iterations, population_size):
        self.experiment_name = experiment_name
        self.max_num_iterations = max_num_iterations
        self.population_size = population_size
        self.experiment_sector = experiment_sector
        self.autoencoders = autoencoders
        self.df_y = []

    def get_mean(self, df):
        if (len(df.columns) < 3):
            return 0
        df_geometric_means = pd.DataFrame()
        lst_years = [13,14,15,16]#list(set(df['year']))
        lst_runs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] #list(set(df['run']))
        # gm = (geometric_mean_score(df.true_class, df.predicted_class, average='binary'))
        # return gm
        #grouped_by_sector_year = df.groupby([df.year, df.run])
        for year in lst_years:
            lst_sector_year_gm = []
            for run in lst_runs:
                #grouped_data = (grouped_by_sector_year.get_group((year, run)))
                grouped_data = df.loc[(df['year']==year) & (df['run']==run)]
                gm = (geometric_mean_score(grouped_data.true_class, grouped_data.predicted_class, average='binary'))
                #print(gm)
                lst_sector_year_gm.append(gm)
            df_rows = pd.DataFrame();
            df_rows["Geom_Mean"] = lst_sector_year_gm
            df_rows["Sector"] = self.experiment_sector
            df_rows["Year"] = (str(year))
            df_geometric_means = pd.concat([df_geometric_means, df_rows])
            #print(df_rows)

        #print(df_geometric_means)
        data = df_geometric_means.groupby(['Year'])['Geom_Mean'].describe()
        mean = np.mean(data['mean'])
        return mean

    def evaluate_by_sum_of_classes(self,X):
        df = pd.DataFrame()
        positive_class_sum = X[len(X) - 1]
        for i in range(self.autoencoders):
            if df.empty == True:
                df = self.df_y[i].copy()
            df["predicted_class_" + str(i)] = [1 if e > X[i] else 0 for e in self.df_y[i].reconstruction_error]

        df["sum"] = df.iloc[:, 8:].sum(axis=1)
        df_finale = pd.DataFrame()
        df_finale['sum'] = df['sum']
        df_finale['predicted_class'] = 0
        df_finale[df_finale['sum'] >= positive_class_sum] = 1
        df['predicted_class'] = df_finale.predicted_class

        mean = self.get_mean(df)
        return -1 * mean

    def evaluate_by_year(self,X):
        df = pd.DataFrame()
        predicted_count_threshold = X[len(X) - 1]
        for i in range(self.autoencoders):
            if df.empty == True:
                df = self.df_y[i].copy()
            df["predicted_class_" + str(i)] = [1 if e > X[i] else 0 for e in self.df_y[i].reconstruction_error]
            # df["reconstruction_error_" + str(i)] = df_y[i].reconstruction_error

        # print(df["reconstruction_error"])
        # df["reconstruction_error"] =df["reconstruction_error_0"] + df["reconstruction_error_1"] + df["reconstruction_error_2"]

        # df["predicted_class"] = [1 if e > X[i] else 0 for e in df_y[i].reconstruction_error]

        df["sum"] = df.iloc[:, 8:].sum(axis=1)
        df.to_csv('test_2.csv')
        df_finale = pd.DataFrame()
        df_finale['sum'] = df['sum']
        df_finale['predicted_class'] = 0
        df_finale[df_finale['sum'] >= predicted_count_threshold] = 1
        df['predicted_class'] = df_finale.predicted_class
        df_geometric_means = pd.DataFrame()

        mean = self.get_mean(df)
        # print(mean)
        if (mean == 0):
            print("ZERO {} {}".format(mean, X))
            # print(X)
        #global maxMeann
        if mean > self.maxMeann:
            self.maxMeann = mean
            print(self.maxMeann)
            # print(X)
        return -1 * mean

    def evaluate_by_thresholds_and_active_autoencoders(self, X):
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
                    df_res = self.df_y[i].copy()
                #povodna chyba
                #df_res["predicted_class_" + str(i)] = [1 if e > X[i] else 0 for e in self.df_y[i].reconstruction_error]
                #print(thresholds[i])
                df_res["predicted_class_" + str(i)] = [1 if e > thresholds[i] else 0 for e in self.df_y[i].reconstruction_error]

        class_sum = (df_res.iloc[:, 8:].sum(axis=1))
        df_res["sum"] = class_sum
        df_finale = pd.DataFrame()
        df_finale['sum'] = df_res['sum']
        df_finale['predicted_class'] = 0
        df_finale[df_finale['sum'] >= (active_encoders / 2) ] = 1
        df_res['predicted_class'] = df_finale.predicted_class

        mean = self.get_mean(df_res)
        return -1 * mean

    def get_dataframes(self):
        columns = ['id', 'sector', 'year', 'run', 'true_class', 'reconstruction_error', 'mean_absolute_error',
                   'root_mean_squared_error', 'epochs', 'thread_id']
        for i in range(self.autoencoders):
            self.df_y.append(pd.read_csv(
                './results/' + self.experiment_sector + '/' + self.experiment_name + '_{:0>2}'.format(i + 1) + '_result.csv'))
            self.df_y[i].columns = columns

            self.df_y[i] = self.df_y[i].drop(columns=['mean_absolute_error', 'root_mean_squared_error'])
        return

    def evaluate_by_single_threshold_and_active_autoencoders(self,X):
        threshold = X[0]
        active_ancoders_count = sum(X[1:])
        # print(X)
        # print(active_ancoders_count)
        df_res = pd.DataFrame()
        for i in range(1, len(X) - 1):
            if X[i] == 1:
                if df_res.empty == True:
                    df_res = self.df_y[i].copy()
                # print(i)
                # print(len(X))
                # print(df_y[i])
                df_res["reconstruction_error_" + str(i)] = df_res.reconstruction_error

        avg_sum = df_res.iloc[:, 8:].sum(axis=1) / active_ancoders_count
        df_res['avg_sum'] = avg_sum
        df_res["predicted_class"] = [1 if e > threshold else 0 for e in df_res['avg_sum']]
        # print(df_res.columns)
        mean = self.get_mean(df_res)
        global maxMeann
        if mean > maxMeann:
            maxMeann = mean
            print(maxMeann)
        # print(mean)
        return -1 * mean

    def evaluate_by_multiple_thresholds_and_all_active_autoencoders(self, X):
        thresholds = X
        df_res = pd.DataFrame()
        active_encoders = len(X)
        for i in range(len(thresholds)):
            if df_res.empty == True:
                df_res = self.df_y[i].copy()
            df_res["predicted_class_" + str(i)] = [1 if e > thresholds[i] else 0 for e in
                                                       self.df_y[i].reconstruction_error]
        class_sum = (df_res.iloc[:, 8:].sum(axis=1))
        df_res["sum"] = class_sum
        df_finale = pd.DataFrame()
        df_finale['sum'] = df_res['sum']
        df_finale['predicted_class'] = 0
        df_finale[df_finale['sum'] >= (active_encoders / 2)] = 1
        df_res['predicted_class'] = df_finale.predicted_class

        mean = self.get_mean(df_res)
        return -1 * mean

    def tuneModel(self, predefinedChromosome = None):
        self.get_dataframes()
        #original with error
        #evaluate_by_thresholds_and_active_autoencoders
        varbound = np.array([[0.01, 1.5], [0, 1]] * self.autoencoders)
        vartype = np.array([['real'],['int']] * self.autoencoders)
        #evaluate_by_multiple_thresholds_and_all_active_autoencoders
        #varbound = np.array([[0, 1.5]] * self.autoencoders)
        #vartype = np.array([['real']] * self.autoencoders)
        algorithm_param = {'max_num_iteration': self.max_num_iterations, \
                           'population_size': self.population_size,\
                           'mutation_probability': 0.1, \
                           'elit_ratio': 0.1, \
                           'crossover_probability': 0.3, \
                           'parents_portion': 0.5, \
                           'crossover_type': 'uniform', \
                           'max_iteration_without_improv': None, \
                           'multiprocessing_engine': None, \
                           'multiprocessing_ncpus': 9,
                           'experiment_name':self.experiment_name}

        model = ga(function=self.evaluate_by_thresholds_and_active_autoencoders, \
                   dimension=len(varbound), \
                   variable_boundaries=varbound, \
                   variable_type_mixed=vartype, \
                   algorithm_parameters=algorithm_param)
        model.run(True,predefinedChromosome)

#experiment_name = "AGRICULTURE_16_ENCODERS_120_EPOCHS_300_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER"
#data = open(experiment_name + "_tuned_2.txt", "r").read()
#print((list(map(float,data))))
#print(data)
#chromosome = []
#for d in data.split("\n"):
#    if(d!=""):
#        chromosome.append(float(d))

#print(len(chromosome[1::2]))
#tuner = ThresholdTuner(experiment_name,"agriculture", 120, 600, 100)
#tuner.tuneModel()
#tuner.get_dataframes()
#fitness = tuner.evaluate_by_thresholds_and_active_autoencoders(chromosome)
#print(fitness)


#f = open("AGRICULTURE_16_ENCODERS_120_EPOCHS_300_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER_raw.txt", "r")

#data = (f.read())
#data = data.replace("[","").replace("]","")
#lines = data.split("\n")
#arr = []
#for l in lines:
#    for val in l.split(" "):
#        if val!="":
#            arr.append(float(val))
#print(arr)
#np.savetxt("AGRICULTURE_16_ENCODERS_120_EPOCHS_300_LR_0.01_FEATURES_08_RUNS_20_STANDARD_ENCODER_tuned.txt", arr)




