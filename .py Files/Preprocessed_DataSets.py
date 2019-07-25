
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.manifold import TSNE
%pylab inline
from IPython.display import Markdown, display
#For displaying titles
def printmrdwn(string):
    display(Markdown(string))
#To disable scroll bar    
# %%javascript
# IPython.OutputArea.prototype._should_scroll = function(lines) {
#     return false;
# }
# %load_ext disable_autoscroll


def plot_data(Dataset_name,X_data,y_target):
    model =TSNE(n_components=2,random_state=0,perplexity=30,n_iter=1000)
    tsne_data=model.fit_transform(X_data)
    tsne_datastack=np.vstack((tsne_data.T,y_target)).T
    tsne_df=pd.DataFrame(data=tsne_datastack,columns=("Dim_1","Dim_2","class"))
    sn=sns.FacetGrid(tsne_df,hue="class",size=4)
    sn.map(plt.scatter,'Dim_1','Dim_2',alpha=1, 
          edgecolor='white', linewidth=0.25, s=100).add_legend(title="class")
    plt.axis('off')
    plt.title(Dataset_name)
    plt.show()

#Titanic Dataset
#https://www.kaggle.com/c/titanic/data
def get_titanic_data():
    df_titanic=pd.read_csv('titanic.csv',)
    #Remove unnecessary columns (body= Body Indentification number, Name= Name)
    df_titanic.drop(['name','body'],1,inplace=True)
    #Fill all the na  
    df_titanic.cabin.fillna('unknown',inplace=True)
    df_titanic.age.fillna(df_titanic['age'].mean(),inplace=True)
    df_titanic.fillna(0,inplace=True)
    #Covert nonnumeric value into numeric
    df_titanic['sex'] = LabelEncoder().fit_transform(df_titanic['sex'])
    df_titanic['cabin'] = LabelEncoder().fit_transform(df_titanic['cabin'].astype(str))
    df_titanic['embarked'] = LabelEncoder().fit_transform(df_titanic['embarked'].astype(str))
    df_titanic['home.dest'] = LabelEncoder().fit_transform(df_titanic['home.dest'].astype(str))
    df_titanic['ticket'] = LabelEncoder().fit_transform(df_titanic['ticket'])
    df_titanic['boat'] = LabelEncoder().fit_transform(df_titanic['boat'].astype(str))
    # df_titanic.head()
    # df_titanic.dtypes
    # print(df_titanic.isnull().sum())
    X = df_titanic.drop("pclass", axis = 1)
    y = df_titanic['pclass']
    print("\n** Titanic Dataset **\n\n",X.head())
    print("\n** Class Column **\n",y.head())    
    return X,y

X_titanic,y_titanic=get_titanic_data()
plot_data("Titanic Data",X_titanic,y_titanic)

#Breast Cancer Dataset
#http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
def get_breastCancer_dataset():

    df_cancer=pd.read_csv('breast-cancer-wisconsin.csv')
    printmrdwn("Data Description")
    print(df_cancer.describe())
    df_cancer['Bare_Nuclei'] = LabelEncoder().fit_transform(df_cancer['Bare_Nuclei'])
    df_cancer= df_cancer.drop("ID_Number", axis = 1)
    printmrdwn("Data Distribution")
    df_cancer['Class'].value_counts().sort_index().plot.bar()
    plt.show()
    printmrdwn("Data Types")
    print(df_cancer.dtypes)
    y = df_cancer['Class']
    X = df_cancer.drop("Class", axis = 1)
    printmrdwn("Breast Cancer Dataset")
    print(X.head())
    printmrdwn("Class Column")
    print(y.head()) 
    return X, y

X_breastCancer,y_breastCancer=get_breastCancer_dataset()
plot_data("Breast Cancer",X_breastCancer,y_breastCancer)

def get_glass():
    #Reading dataset from the csv file
    df_glass=pd.read_csv('glass.csv')
    printmrdwn("Data Description")
    print(df_glass.describe())
    #Randomozing the dataset rows
    np.random.seed(1)
    shuffled_index=np.random.permutation(df_glass.index)
    df_glass=df_glass.loc[shuffled_index]
    #NonUniform distribution of data
    printmrdwn("Data Distribution")
    df_glass['Type'].value_counts().sort_index().plot.bar()
    plt.show()
    printmrdwn("Data Types")
    print(df_glass.dtypes)
    #Convert dataset into taarget and attribute set
    # axis 1 refers to the columns
    X= df_glass.drop('Type', axis = 1)
    y=df_glass['Type']
    printmrdwn("Glass Dataset",)
    print(X.head())
    printmrdwn("Class Column")
    print(y.head()) 
    #feature_list=dataset_data.columns
    return X, y

X_glass,y_glass=get_glass()
plot_data("Glass Dataset",X_glass,y_glass)

#https://archive.ics.uci.edu/ml/machine-learning-databases/00475/
#Audit Dataset
def get_auditData():
    #Reading dataset from the csv file
    df_auditrisk=pd.read_csv('audit_risk.csv')
    printmrdwn("Data Description")
    print(df_auditrisk.describe())
    #Randomozing the dataset rows
    np.random.seed(1)
    shuffled_index=np.random.permutation(df_auditrisk.index)
    df_auditrisk=df_auditrisk.loc[shuffled_index]
    #NonUniform distribution of data
    printmrdwn("Data Distribution")
    df_auditrisk['Risk'].value_counts().sort_index().plot.bar()
    # print(df_auditrisk.isna().sum())
    df_auditrisk.Money_Value.fillna(0,inplace=True)
    plt.show()    
    df_auditrisk['LOCATION_ID'] = LabelEncoder().fit_transform(df_auditrisk['LOCATION_ID'])
    printmrdwn("Data Types")
    print(df_auditrisk.dtypes)
    #Convert dataset into taarget and attribute set
    # axis 1 refers to the columns
    X= df_auditrisk.drop('Risk', axis = 1)
    y=df_auditrisk['Risk']
    printmrdwn("Audit Dataset")
    print(X.head())
    printmrdwn("Risk Column")
    print(y.head()) 
    return(X,y)

X_audit,y_audit=get_auditData()
plot_data("Audit Dataset",X_audit,y_audit)

#Car Dataset
#https://archive.ics.uci.edu/ml/datasets/car+evaluation
def get_car():
    #Reading dataset from the csv file
    df_car=pd.read_csv('car.csv')
    printmrdwn("Data Description")
    print(df_car.describe())
    #Randomozing the dataset rows
    np.random.seed(1)
    shuffled_index=np.random.permutation(df_car.index)
    df_car=df_car.loc[shuffled_index]
    #NonUniform distribution of data
    printmrdwn("Data Distribution")
    df_car['Class'].value_counts().sort_index().plot.bar()
    plt.show() 
    printmrdwn("Nan Values if any")
    print(df_car.isna().sum())
    printmrdwn("Data Types")
    print(df_car.dtypes)
    printmrdwn("Data Set")
    print(df_car.head(5))
    #Encoding for ordinal value
    printmrdwn("After Ordinal Encoding")
    df_car['Buying'] = ce.OrdinalEncoder().fit_transform(df_car['Buying'])
    df_car['Buying'] = ce.OrdinalEncoder().fit_transform(df_car['Buying'])
    df_car['Maintance'] = ce.OrdinalEncoder().fit_transform(df_car['Maintance'])
    df_car['Doors'] = ce.OrdinalEncoder().fit_transform(df_car['Doors'])
    df_car['Persons'] = ce.OrdinalEncoder().fit_transform(df_car['Persons'])
    df_car['Lug_boot'] = ce.OrdinalEncoder().fit_transform(df_car['Lug_boot'])
    df_car['Sfety'] = ce.OrdinalEncoder().fit_transform(df_car['Sfety'])
    printmrdwn("Modified Data Types")
    print(df_car.dtypes)
    printmrdwn("Modified Data")
    print(df_car.head(5))
    #Convert dataset into taarget and attribute set
    # axis 1 refers to the columns
    X= df_car.drop('Class', axis = 1)
    y=df_car['Class']
    printmrdwn("Car Dataset")
    print(X.head(5))
    printmrdwn("Class Column")
    print(y.head(5))    
    return X, y

X_car,y_car=get_car()
plot_data("Car Dataset",X_car,y_car)

#Vote Dataset
#https://archive.ics.uci.edu/ml/datasets/congressional+voting+records
def get_voteData():
    #Reading dataset from the csv file
    df_vote=pd.read_csv('house_votes.csv')
    printmrdwn("Data Description")
    print(df_vote.describe())
    #Randomozing the dataset rows
    np.random.seed(1)
    shuffled_index=np.random.permutation(df_vote.index)
    df_vote=df_vote.loc[shuffled_index]
    #NonUniform distribution of data
    printmrdwn("Data Distribution")
    df_vote['Class'].value_counts().sort_index().plot.bar()
    plt.show() 
    printmrdwn("Nan Values if any")
    print(df_vote.isna().sum())
    printmrdwn("Data Types")
    print(df_vote.dtypes)
    printmrdwn("Data Set")
    print(df_vote.head(5))
    #Here in this dataaset we have question marks. I am not handling question mark. Considering the question mark as another categorical varable
    #Encoding for ordinal value
    printmrdwn("After Ordinal Encoding")
    df_vote_en= ce.OrdinalEncoder().fit_transform(df_vote)
    printmrdwn("Modified Data Types")
    print(df_vote_en.dtypes)
    printmrdwn("Modified Data")
    print(df_vote_en.head(5))
    #Convert dataset into taarget and attribute set
    # axis 1 refers to the columns
    X= df_vote_en.drop('Class', axis = 1)
    y=df_vote['Class']
    printmrdwn("Vote Dataset")
    print(X.head(5))
    printmrdwn("Class Column")
    print(y.head(5))    
    return X, y

X_vote,y_vote=get_voteData()
plot_data("Vote Dataset",X_vote,y_vote)

#Sonar Dataset
#http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)
def get_sonarData():
    #Reading dataset from the csv file
    df_sonar=pd.read_csv('sonar.csv')
    printmrdwn("Data Description")
    print(df_sonar.describe())
    #Randomozing the dataset rows
    np.random.seed(1)
    shuffled_index=np.random.permutation(df_sonar.index)
    df_sonar=df_sonar.loc[shuffled_index]
    #NonUniform distribution of data
    printmrdwn("Data Distribution")
    df_sonar['Class'].value_counts().sort_index().plot.bar()
    plt.show() 
    printmrdwn("Nan Values if any")
    print(df_sonar.isna().sum())
    printmrdwn("Data Types")
    print(df_sonar.dtypes)
    printmrdwn("Data Set")
    print(df_sonar.head(5))
    #Convert dataset into taarget and attribute set
    # axis 1 refers to the columns
    X= df_sonar.drop('Class', axis = 1)
    y=df_sonar['Class']
    printmrdwn("Sonar Dataset")
    print(X.head(5))
    printmrdwn("Class Column")
    print(y.head(5))
    return X, y

X_sonar,y_sonar=get_sonarData()
plot_data("Sonar Dataset",X_sonar,y_sonar)

#Digit-Recognizer Dataset- From kaggle
#https://www.kaggle.com/c/digit-recognizer/data
def get_digitData():
    #Reading dataset from the csv file
    df_digitRecognizer=pd.read_csv('DigitRecognizer.csv')
    printmrdwn("Data Description")
    print(df_digitRecognizer.describe())
    #Randomozing the dataset rows
    np.random.seed(1)
    shuffled_index=np.random.permutation(df_digitRecognizer.index)
    df_digitRecognizer=df_digitRecognizer.loc[shuffled_index]
    # #NonUniform distribution of data
    printmrdwn("Data Distribution")
    df_digitRecognizer['label'].value_counts().sort_index().plot.bar()
    plt.show() 
    printmrdwn("Nan Values if any")
    print(df_digitRecognizer.isna().sum())
    printmrdwn("Data Types")
    print(df_digitRecognizer.dtypes)
    printmrdwn("Data Set")
    print(df_digitRecognizer.head(5))
    #To display digits in the dataset
    printmrdwn("Data Image")
    indx=205
    img=np.array(df_digitRecognizer.loc[indx,'pixel0':'pixel783']).reshape(28,28)
    plt.imshow(img,cmap='gray')
    plt.show()
    #Convert dataset into taarget and attribute set
    # axis 1 refers to the columns
    X= df_digitRecognizer.drop('label', axis = 1)
    X=X.loc[:,'pixel0':'pixel783']/255
    y=df_digitRecognizer['label']
    printmrdwn("Digit Recognizer Dataset")
    print(X.head(5))
    printmrdwn("Label Column")
    print(y.head(5))
    return X, y



X_digit,y_digit=get_digitData()
plot_data("Digit Dataset",X_digit,y_digit)

#Iris Dataset from UCI
#https://archive.ics.uci.edu/ml/datasets/iris
def get_IrisData():
    #Reading dataset from the csv file
    df_iris=pd.read_csv('iris.csv')
    printmrdwn("Data Description")
    print(df_iris.describe())
    #Randomozing the dataset rows
    np.random.seed(1)
    shuffled_index=np.random.permutation(df_iris.index)
    df_iris=df_iris.loc[shuffled_index]
    #NonUniform distribution of data
    printmrdwn("Data Distribution")
    df_iris['species'].value_counts().sort_index().plot.bar()
    plt.show() 
    printmrdwn("Nan Values if any")
    print(df_iris.isna().sum())
    printmrdwn("Data Types")
    print(df_iris.dtypes)
    printmrdwn("Data Set")
    print(df_iris.head(5))
    #Convert dataset into taarget and attribute set
    # axis 1 refers to the columns
    X= df_iris.drop('species', axis = 1)
    y=df_iris['species']
    printmrdwn("Iris Dataset")
    print(X.head(5))
    printmrdwn("Class Column")
    print(y.head(5))
    return X, y



X_iris,y_iris=get_IrisData()
plot_data("Iris Data",X_iris,y_iris)

#Mushroom Dataset
#https://www.kaggle.com/uciml/mushroom-classification/downloads/mushrooms.csv/1
def get_mushroomData():
    #Reading dataset from the csv file
    df_mushrooms=pd.read_csv('mushrooms.csv')
    printmrdwn("Data Description")
    print(df_mushrooms.describe())
    #Randomozing the dataset rows
    np.random.seed(1)
    shuffled_index=np.random.permutation(df_mushrooms.index)
    df_mushrooms=df_mushrooms.loc[shuffled_index]
    #NonUniform distribution of data
    printmrdwn("Data Distribution")
    df_mushrooms['class'].value_counts().sort_index().plot.bar()
    plt.show() 
    printmrdwn("Nan Values if any")
    print(df_mushrooms.isna().sum())
    printmrdwn("Data Types Before Encoding")
    print(df_mushrooms.dtypes)
    printmrdwn("Data Set")
    print(df_mushrooms.head(5))
    #Convert dataset into taarget and attribute set
    # axis 1 refers to the columns
    X= df_mushrooms.drop('class', axis = 1)
    y=df_mushrooms['class']
    #Encoded data
    X=X.apply(LabelEncoder().fit_transform)
    printmrdwn("Data Set After Encoding")
    print(X.head(5),y.head(5))
    printmrdwn("Data Types After Encoding")
    print(X.dtypes)
    printmrdwn("Mushroom Dataset")
    print(X.head(5))
    printmrdwn("Class Column")
    print(y.head(5))
    return X, y



X_mushroom,y_mushroom=get_mushroomData()
plot_data("Mushroom Data",X_mushroom,y_mushroom)


