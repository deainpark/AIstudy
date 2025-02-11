import tensorflow as tf
from sklearn.model_selection import train_test_split

(train_input,train_target),(test_input,test_target) = tf.keras.datasets.imdb.load_data(num_words=500)

train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2,random_state=42)

train_seq = tf.keras.preprocessing.sequence.pad_sequences(train_input, maxlen=100)

val_seq = tf.keras.preprocessing.sequence.pad_sequences(val_input,maxlen=100)


print(train_seq.shape)

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(8, input_shape=(100, 500)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


train_oh = tf.keras.utils.to_categorical(train_seq)

val_oh = tf.keras.utils.to_categorical(val_seq)

rmsprop = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

chkcb = tf.keras.callbacks.ModelCheckpoint('bestsimplernn.model.x', save_best_only = True)
earlycb = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(train_oh,train_target, epochs=100, batch_size=64,validation_data=(val_oh,val_target),callbacks=[chkcb,earlycb])

"""

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(500, 16, input_length=100),
    tf.keras.layers.SimpleRNN(8),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

rmsprop = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

chkcb = tf.keras.callbacks.ModelCheckpoint('bestembedding.model.x', save_best_only = True)

earlycb = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(train_seq,train_target, epochs=100, batch_size=64,validation_data=(val_seq,val_target),callbacks=[chkcb,earlycb])

"""