import pickle
import fancyimpute as fi
import knnimpute
import numpy as np

def run():
    f = open('/Users/colleensmith/Documents/BluWave_Challenge/cleaned_data.p', 'r')
    cleaned_data = pickle.load(f)
    f.close()

    pw_cols = ['PW1', 'PW2', 'PW3', 'PW4']

    #X_filled_knn = fi.IterativeImputer().fit_transform(cleaned_data.loc[:, ['PW1', 'PW2', 'PW3', 'PW4']])
    X_filled_knn = knnimpute.knn_impute_optimistic(np.array(cleaned_data.loc[:, pw_cols]), 
                                    np.array(cleaned_data.loc[:, pw_cols].isnull()), 3)

    f = open('/Users/colleensmith/Documents/BluWave_Challenge/imputed_data_knn.p', 'w')
    #f = open('/Users/colleensmith/Documents/BluWave_Challenge/imputed_data_mice.p', 'w')
    pickle.dump(X_filled_knn, f)
    f.close()

if __name__ == '__main__':
    run()