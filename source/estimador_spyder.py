"""
Script com ColumnTransformer
"""

"""
ToDo:
    - Tratar colunas NOM de df_test tbm.
      - Atribuir a uma mesma nova classe todos os valores em teste não vistos em treino.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
# from FNC_tratandoFeatures import FNC_tratandoFeatures
import time
import matplotlib.pyplot as plt
import numpy as np



start = time.time()


##--- Constantes
PATH_FILE_TRAIN = 'data/train.csv'
PATH_FILE_TEST  = 'data/test.csv'
USE_COL_ID     = ['id']
# USE_COLS_CAT_BIN   = ['bin_0','bin_1','bin_2','bin_3','bin_4']
# USE_COLS_CAT_BIN   = ['bin_0','bin_1','bin_2','bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4']
USE_COLS_CAT_BIN   = ['bin_0','bin_1','bin_2','bin_3','bin_4']
# USE_COLS_CAT_NOM   = ['nom_0','nom_1','nom_2','nom_3','nom_4']
USE_COLS_CAT_NOM   = ['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9']
# USE_COLS_CAT_NOM   = ['nom_5','nom_6','nom_7']
USE_COLS_CAT_ORD   = []
USE_COL_TARGET = ['target']
TEST_SIZE = 0.2
RANDOM_SEED = 0
CAT_STRATEGY = 'most_frequent'
N_SPLITS = 2
SCORING = 'accuracy'


#--- Flags
flag_nom    = 1
flag_ML     = 1
flag_kaggle = 0


##--- Leitura
df_train = pd.read_csv( PATH_FILE_TRAIN, index_col = USE_COL_ID, usecols = USE_COL_ID + USE_COLS_CAT_BIN + USE_COLS_CAT_NOM + USE_COLS_CAT_ORD + USE_COL_TARGET)
df_test  = pd.read_csv( PATH_FILE_TEST,  index_col = USE_COL_ID, usecols = USE_COL_ID + USE_COLS_CAT_BIN + USE_COLS_CAT_NOM + USE_COLS_CAT_ORD )

df_train_2 = df_train.copy()
df_test_2  = df_test.copy()


if flag_nom:
    list_feat = USE_COLS_CAT_NOM
    for feat in list_feat:
        # feat = "nom_5"
        v_unique = df_train_2[feat].unique()
        v_bar = []
        v_bar_round = []
        for value in v_unique:
            df1 = df_train_2[feat]
            len_value = len( df_train_2[ df_train_2[feat] == value ] )
            sum_value = sum( df_train_2.target[ df_train_2[feat] == value ] )
            score_value = sum_value / len_value
            score_value_round = np.round( score_value * 20 ) / 20
            v_bar.append( score_value )
            v_bar_round.append( score_value_round )
            df_train_2[feat] [ df_train_2[feat]==value ] = score_value_round
            
        
        plt.figure()
        plt.subplot(211)
        plt.bar( list(range(0,len(v_bar))), v_bar )
        plt.subplot(212)
        plt.bar( list(range(0,len(v_bar_round))), v_bar_round )


if flag_ML:
    ##--- Split features e target
    if flag_kaggle:
        X_train = df_train_2.drop(columns=USE_COL_TARGET)
        y_train = df_train_2.loc[:, USE_COL_TARGET]
    
        X_test = df_test_2
    
    else:
        X = df_train_2.drop(columns=USE_COL_TARGET)
        y = df_train_2.loc[:, USE_COL_TARGET]
    
        ##--- Split train e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    
    ##--- Instanciando
    # ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
    cat_imputer = SimpleImputer(strategy=CAT_STRATEGY)
    clf1 = LogisticRegression(random_state=RANDOM_SEED)
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    
    ##--- Pipe categoric
    cat_feat = USE_COLS_CAT_BIN + USE_COLS_CAT_NOM + USE_COLS_CAT_ORD
    cat_transf = Pipeline([('Cat_Imputer', cat_imputer), ('OneHot', ohe)])
    
    
    ##--- Preprocessador
    preprocessor = ColumnTransformer(transformers=[    
        ('Categoric', cat_transf, cat_feat)
    ])
    
    
    name = 'LogReg'
    pipe = Pipeline([('Preprocessor', preprocessor), (name, clf1)])
    scores = cross_val_score(pipe, X_train, y_train, scoring=SCORING, cv=kfold)
    print('------------------------------------------------------------------')
    print(f'Cross-val {name} com {N_SPLITS} folds')
    print(f'Mean: {scores.mean()*100:.2f}%')
    print(f'Std : {scores.std()*100:.2f}%')  
    pipe.fit(X_train, y_train)
    print("TERMINOU FIT!")
    y_pred = pipe.predict(X_test)
    print("TERMINOU PREDICT!")
    if not(flag_kaggle):
        print(f'Acc teste: {accuracy_score(y_test, y_pred)*100:.2f}%')
    print('------------------------------------------------------------------')
    
    
    ##--- Criando arquivo para submeter ao Kaggle
    import csv
    with open('data/submission.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow( USE_COL_ID + USE_COL_TARGET )
        for x,y in zip( df_test_2.index, y_pred ):
            spamwriter.writerow([x,y])
    
    print(f"Soma de y_pred: {sum(y_pred)}")


end = time.time()
print(f"Tempo em segundos: {(end - start):.2f}")