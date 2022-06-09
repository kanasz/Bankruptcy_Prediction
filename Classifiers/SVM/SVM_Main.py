import os
import multiprocessing

from Classifiers.SVM.SVM_Functions import run_experiment


def create_args_list(c_range, weight_range,sectors,years, run_count, features_from, features_to):
    args = []
    for c in c_range:
        for weight in weight_range:
            for year in years:
                for sector in sectors:
                    args.append((sector,year, c, weight, run_count , features_from, features_to))
    return args

c_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
#c_range = [0.0001]
weight_range = [50, 100, 200, 300, 400, 500, 600]
#weight_range=[300]
years = [13,14,15,16]
#print(years)
sectors = ['retail']
run_count = 20
features_from = 0
features_to = 60

data_pairs = create_args_list(c_range, weight_range,years,sectors,run_count,features_from, features_to)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=12)
    pool.map(run_experiment, data_pairs)