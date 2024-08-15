import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import random

#Kahmar Stathum

#mnist = tf.keras.datasets.mnist

#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x_train = tf.keras.utils.normalize(x_train, axis=1)
#x_test = tf.keras.utils.normalize(x_test, axis=1)

#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
#model.add(tf.keras.layers.Dense(128, activation="relu"))
#model.add(tf.keras.layers.Dense(128, activation="relu"))
#model.add(tf.keras.layers.Dense(10, activation="softmax"))

#model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=8)


#model.save('handwritten.model.keras')


model = tf.keras.models.load_model('handwritten.model.keras')


min_number = 1
max_number = 12

while True:
    image_number = random.randint(min_number, max_number)
    
    if os.path.isfile(f"digits/digit{image_number}.png"):
        try:
            # Load the image in grayscale
            img = cv2.imread(f"digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                raise ValueError(f"Image digits/digit{image_number}.png could not be loaded.")
            
           
            img = cv2.resize(img, (28, 28))
            
            
            img = np.invert(np.array([img]))
            img = img.reshape(1, 28, 28)
            
            # Predict the digit
            prediction = model.predict(img)
            print(f"This digit is probably a {np.argmax(prediction)}")
            
            # Display the image
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
            
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
           
            break
    else:
        print(f"No file found for digits/digit{image_number}.png. Trying another number...")
        
# From Kahmar Stathum to train the model

#Comment everything out and uncomment out the upper half, raise the number of epochs,
# And create another file, I.E Handwritten2.model.keras, this will allow it to be more accurate.
