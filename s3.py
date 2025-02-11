from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) = keras.datasets.imdb.load_data(num_words=500)

train_input, val_input, train_target,val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

seq = keras.preprocessing.sequence

train_seq = seq.pad_sequences(train_input, maxlen=100)

val_seq = seq.pad_sequences(val_input, maxlen=100)
"""
model = keras.Sequential([
    keras.layers.Embedding(500, 16, input_length=100),
    keras.layers.LSTM(8),
    keras.layers.Dense(1, activation='sigmoid')
])

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)

model.compile(optimizer=rmsprop, loss='binary_crossentropy',metrics=['accuracy'])
chekcb = keras.callbacks.ModelCheckpoint('bestlstm.model.x', save_best_only=True)
earlycb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(train_seq, train_target, epochs=100, batch_size=64,validation_data=(val_seq, val_target), callbacks=[chekcb,earlycb])
"""

model2 = keras.Sequential([
    keras.layers.Embedding(500, 16, input_length=100),
    keras.layers.LSTM(8,dropout=0.3, return_sequences=True),
    keras.layers.LSTM(8, dropout=0.3),
    keras.layers.Dense(1, activation='sigmoid')
],)


rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)

model2.compile(optimizer=rmsprop, loss='binary_crossentropy',metrics=['accuracy'])
model2.save("/aistudy/2nn.x")
chekcb = keras.callbacks.ModelCheckpoint('best2rnn.model.h5', save_best_only=True)
earlycb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model2.fit(train_seq, train_target, epochs=10, batch_size=64,validation_data=(val_seq, val_target), callbacks=[chekcb,earlycb])


model4 = keras.Sequential([
    keras.layers.Embedding(500, 16, input_length=100),
    keras.layers.GRU(8),
    keras.layers.Dense(1, activation='sigmoid')
])

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)

model4.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
chekcb = keras.callbacks.ModelCheckpoint('bestgrumodel.model.h5', save_best_only=True)
earlycb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model4.fit(train_seq, train_target, epochs=10, batch_size=64, validation_data=(val_seq, val_target),callbacks=[chekcb, earlycb])

test_seq = seq.pad_sequences(test_input, maxlen=100)
model4 = keras.models.load_model('/aistudy/2nn.x')
model4.evaluate(test_seq, test_target)
