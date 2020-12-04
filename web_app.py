import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import svm


@st.cache
def read_data():
    players_data = pd.read_csv('https://raw.githubusercontent.com/Kun97/FIFA20_DataAnalysis/main/data/players_overview.csv')
    recommender_data = pd.read_csv('https://raw.githubusercontent.com/Kun97/FIFA20_DataAnalysis/main/data/recommender_data.csv')
    scaler_data = pd.read_csv('https://raw.githubusercontent.com/Kun97/FIFA20_DataAnalysis/main/data/scaler_data.csv')
    regression_scaler_data = pd.read_csv('https://raw.githubusercontent.com/Kun97/FIFA20_DataAnalysis/main/data/regression_scaler_data.csv')
    regression_data = pd.read_csv('https://raw.githubusercontent.com/Kun97/FIFA20_DataAnalysis/main/data/regression_data.csv')
    classfication_data = pd.read_csv('https://raw.githubusercontent.com/Kun97/FIFA20_DataAnalysis/main/data/classification_data.csv')

    num_col_all = regression_data.describe().columns.tolist()
    y = regression_scaler_data['value_eur']
    X = regression_scaler_data[num_col_all].drop('value_eur',axis=1)

    y1 = regression_data['value_eur']/10000
    X1 = regression_data[num_col_all].drop('value_eur',axis=1)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1,test_size=0.3, random_state=1)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)

    name = players_data['Name']

    Y_class = classfication_data['Positon_General']
    X_class = classfication_data.drop('Positon_General', axis=1)
    X_train_class, X_test_class, Y_train_class, Y_test_class = train_test_split(X_class, Y_class, test_size = .3, random_state = 1)

    return players_data, recommender_data, scaler_data, X_train, y_train, X_test, y_test, X_train1, y_train1, X_test1, y_test1, name, X_train_class, X_test_class, Y_train_class, Y_test_class

# data reading
players_data, recommender_data, scaler_data, X_train, y_train, X_test, y_test, X_train1, y_train1, X_test1, y_test1, name, X_train_class, X_test_class, Y_train_class, Y_test_class = read_data()



## recommender system
@st.cache()
def recommender_system(name, num_players):

  try:
    cha = list([name])
    num = int(num_players)
  except:
    return 'Check your inputs!'

  length_check = (len(cha) == 1)
  exist_check = (recommender_data['Name'] == name).any()

  if length_check & exist_check:
      PC1 = recommender_data[recommender_data['Name'] == name]['PC1'].values[0]
      PC2 = recommender_data[recommender_data['Name'] == name]['PC2'].values[0]

      result = recommender_data.apply(lambda row: np.sqrt((row.PC1 - PC1)**2 + (row.PC2 - PC2)**2), axis=1).sort_values().iloc[:num+1]
    
    
  else:
    return "Player Doesn't exists!"
  
  return players_data.iloc[result.index[1:]].reset_index(drop=True)

@st.cache()
def kmeans(num_k):
    km = KMeans(n_clusters=num_k)
    pred = km.fit_predict(scaler_data)

    return pred
    

@st.cache()
def lasso_regression(a):
    lasso=Lasso(alpha=a)
    lasso.fit(X_train, y_train)
    ls_pred = lasso.predict(X_test)
    R2 = metrics.r2_score(y_test, ls_pred)
    #st.write('Coefficient of determination(R2) of lasso regression = {}'.format(R2))
    lasso_coeffcients = pd.DataFrame([X_train.columns,lasso.coef_]).T
    lasso_coeffcients = lasso_coeffcients.rename(columns={0: 'Attribute', 1: 'Coefficients'})
    lasso_coeffcients = lasso_coeffcients[lasso_coeffcients['Coefficients'] != 0].sort_values(by='Coefficients', ascending=False)
    #st.text('Non-zero Attributes Coefficients')
    #st.table(lasso_coeffcients.reset_index(drop=True))

    return R2, lasso_coeffcients, ls_pred

@st.cache()
def random_forest(n_estimators,min_samples_split,max_depth,bootstrap):
    rf = RandomForestRegressor(n_estimators=n_estimators
        , min_samples_split=min_samples_split
        , max_depth=max_depth
        , bootstrap=bootstrap
        , n_jobs=-1)

    rf.fit(X_train1, y_train1)
    rf_pred = rf.predict(X_test1)
    R2 = metrics.r2_score(y_test1, rf_pred)

    importances = rf.feature_importances_
    fi_df = pd.DataFrame(importances, columns=['importances'])
    fi_df['feature_name'] =  X_train1.columns
    fi_df = fi_df[fi_df['importances']>0.00115]
    fi_df.sort_values(by=['importances'], ascending=False, inplace=True)

    return R2, fi_df, rf_pred

@st.cache()
def DT_classifier(criterion, max_depth, max_leaf_nodes, min_samples_split):
    #build model with hyperparameter initialized
    dt_clf = DecisionTreeClassifier(criterion = criterion, max_depth = max_depth, max_leaf_nodes = max_leaf_nodes, min_samples_split =min_samples_split)
    dt_clf.fit(X_train_class, Y_train_class)
    dt_clf_pred = dt_clf.predict(X_test_class)
    #estimate the model
    acc = round(metrics.accuracy_score(Y_test_class, dt_clf_pred),2)
    f1 = round(metrics.f1_score(Y_test_class, dt_clf_pred, average = 'macro'),2)
    recall = round(metrics.recall_score(Y_test_class, dt_clf_pred, average = 'micro'),2)
    precision = round(metrics.precision_score(Y_test_class, dt_clf_pred, average = 'micro'),2)
    estimator_df = pd.DataFrame({'metrics':['accuracy','f1 score','recall','precision'],
                            'value':[acc,f1,recall,precision]})
    return estimator_df

@st.cache()
def SVM_classifier(max_iteration,cost):
    #build model with hyperparameter initialized
    sc = StandardScaler()
    svc = svm.LinearSVC(max_iter=max_iteration, C = cost, random_state=1,
                       multi_class='crammer_singer')
    svc_pipe = make_pipeline(sc, svc)
    svc_pipe.fit(X_train_class,Y_train_class)
    svc_pred = svc_pipe.predict(X_test_class)
    #estimate the model
    acc = round(metrics.accuracy_score(Y_test_class, svc_pred),2)
    f1 = round(metrics.f1_score(Y_test_class, svc_pred, average = 'macro'),2)
    recall = round(metrics.recall_score(Y_test_class, svc_pred, average = 'micro'),2)
    precision = round(metrics.precision_score(Y_test_class, svc_pred, average = 'micro'),2)
    estimator_df = pd.DataFrame({'metrics':['accuracy','f1 score','recall','precision'],
                            'value':[acc,f1,recall,precision]})
    #model inference
    columns = list(X_train_class.columns)
    coef_df = pd.DataFrame(svc_pipe.steps[-1][1].coef_, columns=columns, index = ['GK','Defender','Midfileder','Attacker'])
    return estimator_df, coef_df

@st.cache()
def Random_Forest_classifier(criterion ,max_depth,n_estimators):
    #build model with hyperparameter initialized
    RF_clf = RandomForestClassifier(criterion = criterion, max_depth = max_depth,
                                               n_estimators=n_estimators, random_state=16)
    RF_clf.fit(X_train_class, Y_train_class)
    RF_clf_pred = RF_clf.predict(X_test_class)

    #estimate the model
    acc = round(metrics.accuracy_score(Y_test_class, RF_clf_pred),2)
    f1 = round(metrics.f1_score(Y_test_class, RF_clf_pred, average = 'macro'),2)
    recall = round(metrics.recall_score(Y_test_class, RF_clf_pred, average = 'micro'),2)
    precision = round(metrics.precision_score(Y_test_class, RF_clf_pred, average = 'micro'),2)
    estimator_df = pd.DataFrame({'metrics':['accuracy','f1 score','recall','precision'],
                            'value':[acc,f1,recall,precision]})
    #model inference
    columns = list(X_train_class.columns)
    coef_df = pd.DataFrame({'Feature':columns,'Importance': RF_clf.feature_importances_}).sort_values('Importance', ascending= False)
    return estimator_df, coef_df
       
st.sidebar.title("About this project:")
st.sidebar.subheader('Data:')
st.sidebar.write('The dataset contains 18,278 players with 104 different attributes like age, overall score, value and skill abilities.')
st.sidebar.subheader('What you can do here:')
st.sidebar.write('1. Using recommender system to find alternative players')
st.sidebar.write('2. Check the result for different parameters settings of different ML algorithms')
st.sidebar.subheader('Author:')
st.sidebar.write('Kun Yang')
st.sidebar.write('Email: kyang23@syr.edu')
st.sidebar.markdown('Dataset Link: https://www.kaggle.com/stefanoleone992/fifa-20-complete-player-dataset?select=players_20.csv' )

st.title('FIFA20 Player Dataset Analysis')

# recommender system
st.header('Recommender System')
select_name = st.selectbox('Please choose the player', name)
num = st.slider('Choose the number of alternative players:', 1, 100, 5, 1 )
st.write(recommender_system(select_name,num))


# cluster
st.header('Clustering Analysis')
clusert_num = st.slider('Choose the number of clusters', 2, 10, 2, 1)
pred = kmeans(clusert_num)
fig, ax = plt.subplots()
ax.scatter(recommender_data.PC1, recommender_data.PC2, c=pred)
fig.suptitle("PC2 Scores vs. PC1 Scores", y=.92)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")  
st.pyplot(fig)


# regression
st.header('Regression Analysis')
st.subheader('Linear Regression with L1 Regularization')
lasso_num = st.slider('Choose the penalty:', 0.00, 1.00, 0.02, 0.01 )
R2, lasso_coeffcients, ls_pred = lasso_regression(lasso_num)

st.write('Coefficient of determination(R2) of lasso regression = **_{}_**'.format(R2))

fig, ax = plt.subplots()
ax.scatter(y_test, ls_pred)
ax.plot(y_test, y_test, ':r')
fig.suptitle("Lasso Result", y=.92)
ax.set_xlabel("Real Value (After Standardization)") 
ax.set_ylabel("Predict Value (After Standardization)")  
st.pyplot(fig)

st.text('Non-zero Attributes Coefficients')
st.table(lasso_coeffcients.reset_index(drop=True))


st.subheader('Random Forest Regression')
trees = st.slider('Choose the number of trees in the forest:(n_estimators)', 0, 500, 150, 10)
split = st.slider('The minimum number of samples required to split an internal node:(min_samples_split)', 0, 10, 2, 1)
depth = st.slider('The maximum depth of the tree:(max_depth)', 0, 50, 30, 1)
bootstrap = st.selectbox('Whether bootstrap samples are used when building trees:',['True', 'False'])

rf_R2, fi_df, rf_pred = random_forest(trees, split, depth, bootstrap)
st.write('Coefficient of determination(R2) of Random Forest = **_{}_**'.format(rf_R2))

rf_fig, rf_ax = plt.subplots()
rf_ax.scatter(y_test1*10000, rf_pred*10000)
rf_ax.plot(y_test1*10000, y_test1*10000, ':r')
rf_fig.suptitle("Random Forest Result", y=.92)
rf_ax.set_xlabel("Real Value")
rf_ax.set_ylabel("Predict Value")  
st.pyplot(rf_fig)

st.text('Feature Importances')
st.table(fi_df.reset_index(drop=True))

# classfication
st.header('Classification Analysis')
# decision tree
st.subheader('Decision Tree Classifier')

class_criterion = st.selectbox('Choose the function to measure the quality of a split:(criterion)', ['gini', 'entropy'])

class_split = st.slider('The minimum number of samples required to split an internal node:(min_samples_split)', 2, 10, 2, 1,key=2)

class_nodes = st.slider('The maximum leaf nodes in the decision tree:(max_leaf_nodes)', 0, 50, 10, 1)

class_depth = st.slider('The maximum depth of the decision tree:(max_depth)', 0, 50, 10, 1)

dtc = DT_classifier(class_criterion, class_depth, class_nodes, class_split)
st.table(dtc)

# svm
st.subheader('Support Vector Machine')
svm_iter = st.number_input('The maximum iteration:',min_value=1, step=100, value=100)
svm_c = st.slider('Choose regularization rate:',.0, max_value=10.,step=0.1, value=0.5)
svm_a, svm_b = SVM_classifier(svm_iter, svm_c)
st.text('Result of SVM')
st.table(svm_a)
st.write(svm_b)

# RandomForestClassifier
st.subheader('Random Forest Classifier')
rfc_criterion = st.selectbox('Choose the function to measure the quality of a split:(criterion)', ['gini', 'entropy'], key=2)
rfc_trees = st.slider('The number of trees in random forest:', 0, 1000, 100, 100, key=2)
rfc_depth = st.slider('The maximum depth of the decision tree:(max_depth)', 0, 50, 10, 1, key=2)

rfc_a, rfc_b = Random_Forest_classifier(rfc_criterion, rfc_depth, rfc_trees)
st.text('Result of Random Forest Classifier')
st.table(rfc_a)
st.write(rfc_b)