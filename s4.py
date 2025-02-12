from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

keras.utils.set_random_seed(42)

(train_input, train_target),(test_input, test_target)=keras.datasets.fashion_mnist.load_data()

train_scaled= train_input.reshape(-1, 28, 28, 1) / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=3, activation=keras.activations.relu, padding='same',input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, kernel_size=(3,3),activation=keras.activations.relu,padding='same'),
    keras.layers.MaxPooling2D(2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation=keras.activations.relu),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation=keras.activations.softmax)
])

model.summary()

keras.utils.plot_model(model)

keras.utils.plot_model(model, show_shapes=True, to_file='cnn-architecture.png', dpi=300)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

chekcb = keras.callbacks.ModelCheckpoint('cnnmodel.h5',save_best_only=True)

earlycb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled,val_target),callbacks=[chekcb, earlycb])

plt.plot(history.history ['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()

model.evaluate(val_scaled, val_target)

plt.imshow(val_scaled[0].reshape(28,28),cmap='gray_r')
plt.show()

preds = model.predict(val_scaled[0:1])
print(preds)

plt.bar(range(1,11), preds[0])
plt.xlabel('class')
plt.ylabel('prob')
plt.show()

classes=['T','pant','s','d','c','sd','sc','sn','b','ab']

print(classes[np.argmax(preds)])

test_scaled = test_input.reshape(-1, 28,28,1) / 255.0

model.evaluate(test_scaled,test_target)