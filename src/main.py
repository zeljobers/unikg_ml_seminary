import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split

class Klasifikacije:

	def __init__(self, data):
		self.data = data
		
		self.prikaz_klasifikatornih_vrednosti()
		# uklonim duplikate		

		self.data.drop_duplicates()
		self.uradi_enkodiranje_prvog_atributa()
		self.uradi_enkodiranje_klasnog_atributa()
	
	# koristim klase zato što je jednostavnije za rad nad dataset promenljivom
	def uradi_enkodiranje_prvog_atributa(self):
		temp = dict()
		for i, value in enumerate(self.data[self.data.columns[0]]):
		 	temp[value] = i
		self.data[self.data.columns[0]] = self.data[self.data.columns[0]].map(temp)
	def uradi_enkodiranje_klasnog_atributa(self):
		temp = dict()
		for i, value in enumerate(self.data[self.data.columns[-1]].unique()):
		 	temp[value] = i
		self.data[self.data.columns[-1]] = self.data[self.data.columns[-1]].map(temp)
	def prikaz_klasifikatornih_vrednosti(self):
		print('Vrednosti klasifikatornog atributa')
		print(list(self.data[self.data.columns[-1]].unique()))
	

input_data = pd.read_csv('yeast/yeast.csv')
klasifikacije = Klasifikacije(input_data)

import seaborn
import matplotlib.pyplot as plt
import numpy as np
n_unique_instances_of_class = len(klasifikacije.data[klasifikacije.data.columns[-1]].unique())
def form_matrix(y_test, y_pred):
	matrix = np.zeros ((n_unique_instances_of_class, n_unique_instances_of_class))
	for i in range(len(y_test)):
		 matrix[y_pred[i]][y_test[i]] += 1
	return matrix.astype('uint64')
		 

def display_matrix(matrix):
 	plt.figure()
 	seaborn.heatmap(matrix, annot=True, cbar=False, cmap='Blues', fmt='d')

 	plt.xlabel('true label')
 	plt.ylabel('predicted label')
 	plt.title('Confusion matrix')
 	plt.show()
 	
def evaluate_metrics(matrix):
 	temp = np.array(matrix)
 	for i in range(n_unique_instances_of_class):
		 TP = temp[i][i]
		 FP = np.sum(temp[i]) - TP
		 FN = np.sum(temp, axis=0)[i] - TP
		 TN = np.sum(np.sum(temp)) - TP - FP - FN		
		 print(f'Evaluacija po Klasi {i} ima metrike:')
		 print(f'TP {TP}, TN {TN}, FN {FN}, FP {FP}')
		
		 TP = TP.item()
		 FN = FN.item()
		 FP = FP.item()
		 TN = TN.item()
		
		 if TP + FP != 0 and TP + FN != 0 and TN + FP != 0 and TN + FN != 0 and (TP != 0 or TN != 0):
 			precision = TP / float(TP + FP)
 			print('Precision: ' + str(round(precision, 2)))
 			NPV = TN / float((TN + FN))
 			print('Negative predictive value: ' + str(round(NPV, 2)))
 			recall = TP / float((TP + FN))
 			print('Recall :' + str(round(recall,2)))
 			specificity = TN / float((TN + FP))
 			print('Specificity :' + str(round(specificity,2)))
 			if  0 != float((precision + recall)):
				 f1 = (2*precision * recall) / float((precision + recall))
 			print('f1 score : ' + str(round(f1, 2)))
 			print()
		 else:
 			print('metrics were unable to be calculated')

def evaluate(prediction, test):
 	counter=0
 	for i in range(len(prediction)):
		 if prediction[i]==test[i]:
 			counter+=1
 	accuracy = counter / len(test)
 	return accuracy

from sklearn.metrics import classification_report

def evaluate_naive_bayes_custom_35_percent(data):
	x_train, x_test, y_train, y_test = train_test_split(
			data[data.columns[:-1]],
			data[data.columns[-1]],
			test_size=0.35
	 	)
	
	GNB=GaussianNB()
	GNB.fit(x_train,y_train)
	prediction=GNB.predict(x_test)
	accuracy=evaluate(prediction, list(y_test))
	print(f'Gaussian Naive Bayes prediction accuracy : {accuracy}')
	
	matrix = form_matrix(list(y_test), prediction)
	
	display_matrix(matrix)
	
	evaluate_metrics(matrix)
	
	
	print()
	report = classification_report(list(y_test), prediction)
	print(report)
	
	MNB=MultinomialNB()
	MNB.fit(x_train,y_train)
	prediction=MNB.predict(x_test)
	accuracy=evaluate(prediction, list(y_test))
	print(f'Multinomial Naive Bayes prediction accuracy : {accuracy}')
	
	BNB=BernoulliNB()
	BNB.fit(x_train,y_train)
	prediction=BNB.predict(x_test)
	accuracy=evaluate(prediction, list(y_test))
	print(f'Bernoulli Naive Bayes prediction accuracy : {accuracy}')
	
	CNB=CategoricalNB()
	CNB.fit(x_train,y_train)
	prediction=CNB.predict(x_test)
	accuracy=evaluate(prediction, list(y_test))
	print(f'Categorical Naive Bayes prediction accuracy : {accuracy}')

x_train, x_test, y_train, y_test = train_test_split(
		klasifikacije.data[klasifikacije.data.columns[:-1]],
		klasifikacije.data[klasifikacije.data.columns[-1]],
		test_size=0.2
 	)



GNB=GaussianNB()
GNB.fit(x_train,y_train)
prediction=GNB.predict(x_test)
accuracy=evaluate(prediction, list(y_test))
print(f'Gaussian Naive Bayes prediction accuracy : {accuracy}')

matrix = form_matrix(list(y_test), prediction)

display_matrix(matrix)

evaluate_metrics(matrix)

from sklearn.metrics import classification_report
print()
report = classification_report(list(y_test), prediction)
print(report)

MNB=MultinomialNB()
MNB.fit(x_train,y_train)
prediction=MNB.predict(x_test)
accuracy=evaluate(prediction, list(y_test))
print(f'Multinomial Naive Bayes prediction accuracy : {accuracy}')

BNB=BernoulliNB()
BNB.fit(x_train,y_train)
prediction=BNB.predict(x_test)
accuracy=evaluate(prediction, list(y_test))
print(f'Bernoulli Naive Bayes prediction accuracy : {accuracy}')

CNB=CategoricalNB()
CNB.fit(x_train,y_train)
prediction=CNB.predict(x_test)
accuracy=evaluate(prediction, list(y_test))
print(f'Categorical Naive Bayes prediction accuracy : {accuracy}')

evaluate_naive_bayes_custom_35_percent(klasifikacije.data)

# -------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.tree import plot_tree


def show_text_representation(tree):
    text_representation=export_text(tree)
    print(text_representation)
    
def show_graphical_representation(tree):
    fig = plt.figure(figsize=(25,20))
    _ =plot_tree(tree, feature_names=klasifikacije.data.columns, filled=True)


def basic_decision_tree(data):
	x_train, x_test, y_train, y_test = train_test_split(
			data[data.columns[:-1]],
			data[data.columns[-1]],
			test_size=0.2
	 	)
		
	decision_tree_model = DecisionTreeClassifier()
	decision_tree_model.fit(x_train, y_train)
	predicted = decision_tree_model.predict(x_test)
	
	show_text_representation(decision_tree_model)
	show_graphical_representation(decision_tree_model)
	
	matrix = form_matrix(list(y_test), predicted)
	
	display_matrix(matrix)

	report = classification_report(list(y_test), predicted)
	print(report)

basic_decision_tree(klasifikacije.data)

from sklearn.model_selection import GridSearchCV
max_depth=[4,6,8,10]
min_samples_leaf=[2,5,7,10]
min_samples_split=[2,5,7,10]

num_classes=len(np.unique(klasifikacije.data[klasifikacije.data.columns[-1]]))

no_weights = dict()
for i in range(n_unique_instances_of_class):
 	no_weights[i] = 0.1

weights = dict()
for i in range(n_unique_instances_of_class):
 	weights[i] = 0
 	
counts=[0 for i in range(n_unique_instances_of_class)]
  
values=list(klasifikacije.data[klasifikacije.data.columns[-1]])
for i in values:
 	counts[i]+=1

for i in range(len(weights)):
    weights[i]=counts[i]/len(klasifikacije.data)
decision_tree_model = DecisionTreeClassifier()
class_weights=[no_weights , weights]    

params={'max_depth':max_depth,
        'min_samples_leaf':min_samples_leaf,
        'min_samples_split':min_samples_split,
        'class_weight':class_weights}

model=GridSearchCV(decision_tree_model, params, cv=5)
model.fit(x_train,y_train)
predicted=model.predict(x_test)

params=model.best_params_
print()
print('best results were achieved with the following parameters: ')
print(params)
print()

dcc=model.best_estimator_
show_graphical_representation(dcc)

matrix = form_matrix(list(y_test), predicted)

display_matrix(matrix)

report = classification_report(list(y_test), predicted)
print(report)
# -----------------------------------------------
from sklearn.ensemble import RandomForestClassifier

n_estimators=[64, 128]
max_depth=[5,10,15]
min_samples_leaf=[2,4,6]
min_samples_split=[2,4,6]
class_weight=['balanced', 'balanced_subsample']

params={'n_estimators':n_estimators,
        'max_depth':max_depth,
        'min_samples_leaf':min_samples_leaf,
        'min_samples_split':min_samples_split,
        'class_weight':class_weight}

optimizer=GridSearchCV(RandomForestClassifier(), params, cv=3, verbose=True)
optimizer.fit(x_train,y_train)
predicted=optimizer.predict(x_test)

params=optimizer.best_params_
print('The best results were obtained using the following set of hyperparameters: ')
print(params)
print()

matrix = form_matrix(list(y_test), predicted)

display_matrix(matrix)

report = classification_report(list(y_test), predicted)
print(report)
# -----------------------------------------------
#Import svm model
from sklearn import svm

kernel=['linear', 'sigmoid'] 
decision_function_shape=['ovo']

params = {
 	'decision_function_shape' : decision_function_shape,
 	'kernel' : kernel
 	}

#Create a svm Classifier
clf = GridSearchCV(svm.SVC(), params, cv=3, verbose=True)
clf.fit(x_train, y_train)
predicted = clf.predict(x_test)

params=clf.best_params_
print('The best results were obtained using the following set of hyperparameters: ')
print(params)
print()

matrix = form_matrix(list(y_test), predicted)

display_matrix(matrix)

report = classification_report(list(y_test), predicted)
print(report)

