# Hoque@UNO
# Load the model & Test

import pickle

f=open('iris_saved_knn_model', 'rb')
loaded_model = pickle.load(f)
#loaded_model = pickle.load(open('iris_saved_knn_model', 'rb'))

X_test = [4.8, 2.9, 1.54, 0.15], [5.9, 2.5, 5.5, 1.2], [5.9, 3.0, 4.6, 1.4]

output = loaded_model.predict(X_test)
print('Predicted outputs are: ', output)

# If you know the test answers and want to compute the accuracy then do the following
Y_test=[0,2,1]

accuracy = loaded_model.score(X_test, Y_test)

print('Accuracy =', accuracy)

f.close() # when done, we better close the file.