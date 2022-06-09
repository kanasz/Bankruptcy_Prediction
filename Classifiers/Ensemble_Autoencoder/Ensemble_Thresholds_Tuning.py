import multiprocessing

from Classifiers.Ensemble_Autoencoder.Ensemple_Thresholds_Tuning_Functions import ThresholdTuner
import numpy as np

sector = "AGRICULTURE"

if __name__ == '__main__':
    EXPERIMENT_NAME = "{}_01_ENCODERS_33_EPOCHS_100_LR_0.01_FEATURES_08_RUNS_20_STANDARD_Y-1_ENCODER".format(sector)
    tuner = ThresholdTuner(EXPERIMENT_NAME, sector, 33, 400, population_size= 600)
    ch = np.loadtxt('./results/tuning/' + EXPERIMENT_NAME + "_tuned.txt")
    #for some encoders
    for i in range(1,len(ch)):
        if i % 2 == 1:
            ch[i] = int(ch[i])
    #ch = None

    #print(ch)
    tuner.tuneModel(ch)



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
