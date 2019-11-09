"""
Set up
"""
import data

X_train, y_train, X_val, y_val, X_test, y_test = data.get_machine_learning_data()

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)


