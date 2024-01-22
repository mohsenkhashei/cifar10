from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
 
# Load the digits dataset
digits = load_digits()
X, y = digits.images, digits.target
 
# Preprocessing
# Normalizing pixel values
X = X / 16.0
 
# Reshaping the data to fit the model
# CNN in Keras requires an extra dimension at the end for channels,
# and the digits images are grayscale so it's just 1 channel
X = X.reshape(-1, 8, 8, 1)
 
# Convert labels to categorical (one-hot encoding)
y = to_categorical(y)
 
# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Building a simple CNN model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(8, 8, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10 classes for digits 0-9
 
# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
# Training the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
 
# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print( "Test accuracy: ", accuracy)

