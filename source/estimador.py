"""
Script com ColumnTransformer
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
from FNC_tratandoFeatures import FNC_tratandoFeatures

##--- Constantes
PATH_FILE_TRAIN = 'data/train.csv'
PATH_FILE_TEST  = 'data/test.csv'
USE_COL_ID     = ['id']
USE_COLS_CAT   = ['bin_0','bin_1','bin_2','bin_3','bin_4']
USE_COL_TARGET = ['target']
TEST_SIZE = 0.2
RANDOM_SEED = 0
CAT_STRATEGY = 'most_frequent'
N_SPLITS = 2
SCORING = 'accuracy'


#--- Flags
flag_kaggle = 1


#--- Aux's de features
aux_name = 2 # 0: Não realizar nenhum tratamento nesta feature.
             # 1: Apenas separa quem tem título e quem não.
             # 2: Separa também pelo tipo de título.

aux_cabin = 2 # 0: Não realizar nenhum tratamento nesta feature.
              # 1: Substituir string apenas por 1º elemento, e NaN ser mantido como NaN.
              # 2: Substitui todas samples por 0.

aux_ticket = 2 # 0: Não realizar nenhum tratamento nesta feature.
               # 1: Substituir string apenas por 1º elemento.
               # 2: Substitui todas samples por 0.


##--- Leitura
df_train = pd.read_csv( PATH_FILE_TRAIN, index_col = USE_COL_ID, usecols = USE_COL_ID + USE_COLS_CAT + USE_COL_TARGET)
df_test  = pd.read_csv( PATH_FILE_TEST,  index_col = USE_COL_ID, usecols = USE_COL_ID + USE_COLS_CAT )

df_train_2 = df_train.copy()
df_test_2  = df_test.copy()


##--- Tratando feature 'Name'
# df_train_2 = FNC_tratandoFeatures( df_train_2, aux_name, aux_cabin, aux_ticket )

# df_test_2  = FNC_tratandoFeatures( df_test_2,  aux_name, aux_cabin, aux_ticket )


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
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
cat_imputer = SimpleImputer(strategy=CAT_STRATEGY)
scaler = StandardScaler()
clf1 = LogisticRegression(random_state=RANDOM_SEED)
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)


##--- Pipe categoric
cat_feat = USE_COLS_CAT
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
y_pred = pipe.predict(X_test)
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
        