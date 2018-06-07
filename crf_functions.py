# Machine Learning Librariesc
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split # Parameter selection
import sklearn_crfsuite
from sklearn_crfsuite import scorers, metrics
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV


import time # Execution time of some blocks

def trainCRFAndEvaluate(X_train, y_train, X_test, y_test, labels, c1 = 0.1, c2 = 0.1, hyperparam_optim = False, n_cv = 3, n_iter = 10):

    for i in range(len(y_train)):
            if y_train[i][0] is None:
                y_train[i][0] = 'none'

    if hyperparam_optim == False:

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=c1,
            c2=c2,
            max_iterations=100,
            all_possible_transitions=True
        )
        
        # Train the model
        crf.fit(X_train, y_train)
        
        # Create the predictions
        y_pred = crf.predict(X_test)

        for i in range(len(y_pred)):
            if y_pred[i][0] is None:
                y_pred[i][0] = 'none'
        
        print(metrics.flat_classification_report(
        y_test, y_pred, labels = labels, digits=3
        ))
        
        return(crf)

    elif hyperparam_optim == True:

        # Define fixed parameters and parameters to search
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
        )

        ## Parameter search
        # Use the same metric for evaluation
        f1_scorer = make_scorer(
            metrics.flat_f1_score,
            labels = labels,
            average='weighted')

        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        rs = RandomizedSearchCV(crf, params_space,
                                cv=n_cv,
                                verbose=1,
                                n_jobs=-1,
                                n_iter=n_iter,
                                scoring=f1_scorer)

        start_time = time.time()
        rs.fit(X_train, y_train)
        print("Hyperparameter optimization took %s seconds to complete" % round((time.time() - start_time), 2))

        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)
        print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

        crf = rs.best_estimator_

        crf.fit(X_train, y_train)

        # Create the predictions
        y_pred = crf.predict(X_test)

        for i in range(len(y_pred)):
            if y_pred[i][0] is None:
                y_pred[i][0] = 'none'
        
        print(metrics.flat_classification_report(
        y_test, y_pred, labels = labels, digits=3
        ))
        
        return(crf)



def joinResultsFirstSecondModel(y_test_type, y_pred_type, y_pred_int):

    y_final_pred = []
    y_pred_type_aux = list(y_pred_type)

    for int_pred in y_pred_int:

        if int_pred[0] == 'false':
            y_final_pred.append(['none'])

        elif int_pred[0] == 'true':
            y_final_pred.append(y_pred_type_aux[0])
            del y_pred_type_aux[0]

        else:
            raise ValueError('Unknown result value in the list of predictions "y_test_int": ', int_pred[0])

    if len(y_pred_type_aux) != 0:
        raise ValueError('Oops! Something went wrong')

    print(metrics.flat_classification_report(
        y_test_type, y_final_pred, labels = ['mechanism', 'advise', 'effect', 'int'], digits=3))
        




