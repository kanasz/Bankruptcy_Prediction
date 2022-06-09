import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
import numpy as np

import Classifiers.Ensemble_Autoencoder.Ensemble_Constants as ec

def run_experiment(args):
    #c_range, weight_range, years, sectors, run_count

    c = args[2]
    weight = args[3]
    year = args[0]
    sector = args[1]
    runs = args[4]
    features_from = args[5]
    features_to = args[6]
    gms = []
    roc_auc_scores = []
    for run in range(runs):

        data_filename = "{}_{}_{}.csv".format(sector, year, 1)
        df = pd.read_csv("../"+ec.PREPARED_DATA_DIRECTORY + "/" + data_filename)
        df.loc[df['IS_BANKRUPT'].isnull()] = 0
        imputer = SimpleImputer()
        scaler = StandardScaler()
        labels = df['IS_BANKRUPT'].copy()
        columns_to_drop = ['Seq_Number', 'IS_BANKRUPT', 'TYPE', 'RUN']

        data = df.drop(columns_to_drop, axis=1)
        data = data.iloc[:,features_from:features_to]
        x_train, x_test, y_train, y_test = train_test_split(data,
                                                            labels,
                                                            test_size=0.20, stratify=labels,shuffle=True)

        x_train = imputer.fit_transform(x_train)
        x_test = imputer.transform(x_test)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        clf = SVC(kernel='linear', class_weight={0: weight}, gamma='auto', C=c)
        clf.fit(x_train, y_train)
        test_predictions = clf.predict(x_test)

        gm = (geometric_mean_score(y_test, test_predictions, average='binary'))
        roc_auc = roc_auc_score(y_test, test_predictions)


        df_result = pd.DataFrame({
            'sector': sector,
            'year': year,
            'run': run,
            'c': c,
            'weight': weight,
            'features_from': features_from,
            'features_to': features_to,
            'geom_mean': gm,
            'roc_auc': roc_auc

        }, index=[0])
        gms=np.append(gms,gm)
        roc_auc_scores = np.append(roc_auc_scores, roc_auc)
        result_file_name = "{}/{}_RESULT.csv".format(ec.RESULTS_DIRECTORY, sector)
        df_result.to_csv(result_file_name, mode='a', header=False,index=None)
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(year,c,weight, np.mean(gms), np.std(gms), np.mean(roc_auc_scores), np.std(roc_auc_scores)))
    return