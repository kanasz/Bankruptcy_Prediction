from Classifiers.Ensemble_Autoencoder.Ensemble_Models import create_ensemble_autoencoder_model_1_dynamic, \
    create_ensemble_autoencoder_model_16_dynamic
from Classifiers.Ensemble_Autoencoder.Ensemble_Plot_Functions import create_box_plots, plot_geometric_means_for_aggregated_data, \
    plot_geometric_means_by_years, plot_gemetric_means, plot_confusion_matrixces

SECTOR = 'All'

#Autoencoder Standard by year 01 THRESHOLDS = [0.00251706, 0.00194314, 0.12493834] positive_class_sum = 3
#THRESHOLDS = [0.00251706, 0.00194314, 0.12493834]
#plot_geometric_means_by_years('Ensemble_Autoencoder_01_MINMAX_1_3_ENCODERS_BY_YEARS_100_EPOCHS_LR_0.01','Agriculture',THRESHOLDS, save_plot=True,positive_class_sum=3)
#Autoencoder Standard by year 02 THRESHOLDS = [0.00132984, 0.0025657,  0.09559555] positive_class_sum = 3
#THRESHOLDS = [0.00132984, 0.0025657,  0.09559555]
#plot_geometric_means_by_years('Ensemble_Autoencoder_02_MINMAX_1_3_ENCODERS_BY_YEARS_100_EPOCHS_LR_0.01','Agriculture',THRESHOLDS, save_plot=True,positive_class_sum=3)
#Autoencoder Standard by year 03 THRESHOLDS = [0.00506822, 0.00458535, 0.14679794] positive_class_sum = 3
#THRESHOLDS = [0.00506822, 0.00458535, 0.14679794]
#plot_geometric_means_by_years('Ensemble_Autoencoder_03_MINMAX_1_3_ENCODERS_BY_YEARS_100_EPOCHS_LR_0.01','Agriculture',THRESHOLDS, save_plot=True,positive_class_sum=3)
#Autoencoder Standard by year 04 THRESHOLDS = [0.00616162, 0.00480936, 0.15013722] positive_class_sum = 3
#THRESHOLDS = [0.00616162, 0.00480936, 0.15013722]
#plot_geometric_means_by_years('Ensemble_Autoencoder_04_MINMAX_1_3_ENCODERS_BY_YEARS_100_EPOCHS_LR_0.01','Agriculture',THRESHOLDS, save_plot=True,positive_class_sum=3)

#Autoencoder MinMax 1 by year 01 THRESHOLDS = [0.28036128, 0.27677144, 0.15500977] positive_class_sum = 1
#Autoencoder MinMax 1 by year 02 THRESHOLDS = [0.22487098, 0.24185781, 0.13072868] positive_class_sum = 1
#Autoencoder MinMax 1 by year 03 THRESHOLDS = [0.29697025, 0.29922197, 0.23337252] positive_class_sum = 1
#Autoencoder MinMax 1 by year 04 THRESHOLDS = [0.2987139, 0.29895944, 0.29433519] positive_class_sum = 1
#THRESHOLDS = [0.008,0.009,0.01,0.025,0.03]
#THRESHOLDS = [0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
#EXPERIMENT_NAME='Ensemble_All_17_DYNAMIC_STANDARD_1_ENCODERS_100_EPOCHS_LR_0.01_60_FEATURES_FIRST_3_YEARS_GOOD_FIT_IMPUTER_UNSORTED_FEATURES_30_RUNS'
#plot_geometric_means_for_aggregated_data(EXPERIMENT_NAME, SECTOR, THRESHOLDS)
#THRESHOLDS = [0.2]
#plot_confusion_matrixces(sector="all",experiment_name=EXPERIMENT_NAME,thresholds=THRESHOLDS,encoders=1)

#THRESHOLDS = [0.0008,0.09,0.1,0.2,0.3]
THRESHOLDS = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17]
#plot_geometric_means_for_aggregated_data(EXPERIMENT_NAME, 'Agriculture', THRESHOLDS)
SECTOR = 'Agriculture'
'''
for FROM_COLUMNS_COUNT in [60]:
    for NUMBER_OF_FEATURES in [8,10,12,14,16, 20,30,40]:
        for ENCODERS in [8, 10, 15, 20, 25, 30]:
            EXPERIMENT_NAME = 'ALL_17_DYNAMIC_STANDARD_' + str(
                ENCODERS) + '_ENCODERS_50_EPOCHS_LR_0.01_' + str(NUMBER_OF_FEATURES) + '_FEATURES_FIRST_' + str(
                int(FROM_COLUMNS_COUNT / 20)) + '_YEARS_IMPUTER_UNSORTED_FEATURES_20_RUNS_UNSCALPED2'

            plot_geometric_means_for_aggregated_data(EXPERIMENT_NAME, SECTOR, THRESHOLDS)

'''

#plot_gemetric_means('Agriculture','AGRICULTURE_01_ENCODERS_01_EPOCHS_100_LR_0.01_FEATURES_60_RUNS_20_STANDARD_1_ENCODER_01',THRESHOLDS)

#Ensemble_Agriculture_13_DYNAMIC_STANDARD_10_ENCODERS_300_EPOCHS_LR_0.01_15_FEATURES_FIRST_3_YEARS_GOOD_FIT_IMPUTER_UNSORTED_FEATURES_0_result
SECTOR = 'RETAIL'
ENCODERS = 1
EPOCHS = 100
NUMBER_OF_FEATURES = 60
RUN_COUNT = 20
EXPERIMENT_NAME = '{}_01_ENCODERS_{:0>2}_EPOCHS_{}_LR_0.01_FEATURES_{:0>2}_RUNS_{:0>2}_STANDARD_1'.format(SECTOR,ENCODERS,EPOCHS, NUMBER_OF_FEATURES, RUN_COUNT)
#plot_geometric_means_for_aggregated_data(EXPERIMENT_NAME + "_ENCODERS_{:0>2}".format(ENCODERS), SECTOR, THRESHOLDS, "reconstruction_error_mean")


from keras.utils.vis_utils import plot_model
model  = create_ensemble_autoencoder_model_16_dynamic()
plot_model(model, to_file='model_16_plot.png', show_shapes=True, show_layer_names=True)