#!/usr/bin/env python
# coding: utf-8

# ###### Import Statement

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydot
import random
from random import randrange
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,KFold
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
from scipy import misc
from collections import OrderedDict
from operator import itemgetter
get_ipython().run_line_magic('matplotlib', 'inline')


# ###### Preprocessing

# In[36]:


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
    y = df_titanic['pclass']
    X = df_titanic.drop("pclass", axis = 1)
    return X,y


# ###### Train_Test_Spliting

# In[37]:


def data_split(X_Data,y_Target):    
    #To split the dataset into 3 parts   
    X_train, X_test, y_train, y_test= train_test_split(X_Data, y_Target, test_size=0.4)
    X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.5)
    
    return X_train,X_val,X_test,y_train,y_val,y_test


# In[38]:


X,y=get_titanic_data()
X_train,X_val,X_test,y_train,y_val,y_test=data_split(X,y)


# In[39]:


# print(X_train.shape)
# print(X_val.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_val.shape)
# print(y_test.shape)


# ###### Methods and submethods declaration for genetic algorithm optimization

# In[40]:


#Method used to split the initial random forest into random forest vector population
def partition_gene_to_chromosome(listin, n):
    random.shuffle(listin)
    return [listin[i::n] for i in range(n)]


# In[63]:


#Function to determin the best fit parents
def call_population_fitness(randomForest_population):
    print(len(randomForest_population))
    fitnesslist=[]
    for rf in randomForest_population:
        print(rf)
        fitness=rf.score(X_val,y_val)
        fitnesslist.append(fitness)

    fitness_matrix = { i : fitnesslist[i] for i in range(0, len(fitnesslist) ) }
    return fitness_matrix


# In[42]:


#Fuction to select the parent population
def call_population_selection(fitness_matrix,randomForest_population,parents_number):
    fitness_ordered=sorted(fitness_matrix.items(), key=itemgetter(1),reverse=True)
    selected_chromRF_Index=list(list(zip(*fitness_ordered))[0])[:parents_number]
    print(selected_chromRF_Index)
    selected_rf_population=[randomForest_population[i] for i in selected_chromRF_Index]
    
    return selected_rf_population


# In[43]:


#Sub Function for cross over
def get_offcross_spring(rf_chromosome1,rf_chromosome2,cross_over_point):
    rf_chromosome1.estimators_=rf_chromosome1.estimators_[0:cross_over_point]
    rf_chromosome2.estimators_=rf_chromosome2.estimators_[cross_over_point:]
    modified_estimators=rf_chromosome1.estimators_+rf_chromosome2.estimators_
    rf_chromosome1.estimators_=modified_estimators
    return rf_chromosome1


# In[44]:


# Main Crossover Function
def call_crossover_generation(selected_rf_population,crossover_per_gen):
    offsprings_gen = []
    # The crossoverpoint
    cross_over_point = np.uint8(len(selected_rf_population[1])/2)   
    
    for k in range(int(crossover_per_gen)):
        # Crossoverindex of the first parent
        parent1_index = k % len(selected_rf_population)
        # Crossoverindex of the seconf parent
        parent2_index = (k+1) % len(selected_rf_population)
        offspring=get_offcross_spring(selected_rf_population[parent1_index],selected_rf_population[parent1_index],cross_over_point)
        offsprings_gen.append(offspring)
    return offsprings_gen


# In[45]:


# Method to get the mutated chromosomes
def get_mutation_gen(randomForest,selected_rf_population,mutation_per_gen):
    offsprings_gen = []
#     print(offsprings_gen)
    rf_for_mutation=[]
    for k in range(int(mutation_per_gen)):  
        # Randomly choose a chromosome for mutation.
        parent1_index = k % len(selected_rf_population)
        rf_for_mutation=random.choice(selected_rf_population)
#         print(k,"k")
#         print(rf_for_mutation)
#         print(len(rf_for_mutation))
        #Randomly choose one of the gene for mutation
        gene_for_mutation = random.choice(rf_for_mutation.estimators_)
#         print(gene_for_mutation) 
        rf_for_mutation.estimators_.remove(gene_for_mutation)
#         print(len(rf_for_mutation.estimators_))
        slctimpurity_df=random.choice(randomForest)
#         print(slctimpurity_df)
        rf_for_mutation.estimators_.append(slctimpurity_df)
        offsprings_gen.append(rf_for_mutation)
    
    return offsprings_gen


# In[46]:


M,N=X.shape
F=int(N**(1/2))

#Total number of decision trees all random forest
initial_population=1000
initial_population_list=list(range(0,initial_population))
#Total number of chromosome(No of sub random forest for genetic algorithm)
n=100
cross_over_rate=0.9
mutation_rate=0.1
number_of_gen=100
crossover_per_gen=(cross_over_rate*n)
mutation_per_gen=(mutation_rate*n)
print(crossover_per_gen)
#Number of parents for next generation
parents_number=50
# Random forest model 
# Generating first population of raandom forest
randomForest = RandomForestClassifier(n_estimators=initial_population,max_features=F)
randomForest.fit(X_train,y_train)


# In[47]:


sub_forest_chromosome=partition_gene_to_chromosome(initial_population_list,n)
# print(sub_forest_chromosome)


# In[48]:


#List of initial population (chromosomes)

randomForest_population=[]
for chrom in range(n):
    
    rf_classifier=RandomForestClassifier(n_estimators=len(sub_forest_chromosome[chrom]))
    rf_classifier.estimators_=[]
    for gene in sub_forest_chromosome[chrom]:
        rf_classifier.estimators_.append(randomForest.estimators_[(gene-1)])    
    
    rf_classifier.classes_=randomForest.classes_
    rf_classifier.n_classes_=randomForest.n_classes_
    rf_classifier.n_outputs_=randomForest.n_outputs_ 
    randomForest_population.append(rf_classifier)
print(len(randomForest_population))
print(randomForest_population)


# In[49]:


for gen in range(number_of_gen):
    print("Generation Number: ", gen)
    # To measure the fitness of each chromosome
    fitness_matrix=call_population_fitness(randomForest_population)
    print(fitness_matrix)
    # To select the best parents for next generation
    next_gen_parents = call_population_selection(fitness_matrix,randomForest_population,parents_number)
    
    # Crossover for generating offspring
    crossover_offsprings = call_crossover_generation(next_gen_parents,crossover_per_gen)
    print(next_gen_parents)
    # Mutation for the generated offspring
    mutated_offspring = get_mutation_gen(randomForest,next_gen_parents,mutation_per_gen)

    # New population for next generation
    next_population=[]
    next_population.append(crossover_offsprings)
    next_population.append(mutated_offspring)
    len(next_population)
    randomForest_population=next_population


# In[66]:


print("Generation Number: ", gen)
# To measure the fitness of each chromosome
# print(randomForest_population)
fitness_matrix=call_population_fitness(randomForest_population)
# print(fitness_matrix)
# To select the best parents for next generation
# next_gen_parents = call_population_selection(fitness_matrix,randomForest_population,parents_number)

# Crossover for generating offspring
# crossover_offsprings = call_crossover_generation(next_gen_parents,crossover_per_gen)
# print(next_gen_parents)
# Mutation for the generated offspring
# mutated_offspring = get_mutation_gen(randomForest,next_gen_parents,mutation_per_gen)

# New population for next generation
# next_population=[]
# next_population.append(crossover_offsprings)
# next_population.append(mutated_offspring)
# len(next_population)
# randomForest_population=next_population


# In[ ]:





# In[ ]:




