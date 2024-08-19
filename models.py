from tensorflow.keras.layers import Layer

# Arch 1 Common Code

def create_cnn_model(input_shape, vocab_size, embedding_dim, input_length):
    input_seq = Input(shape=(input_shape, 1))
    kmer_input = Input(shape=(input_length,))
    other_features = Input(shape=(6,))

    cnn_out = Conv1D(64, kernel_size=3, activation='relu')(input_seq)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    cnn_out = Flatten()(cnn_out)

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(kmer_input)
    kmer_out = Flatten()(embedding_layer)

    concatenated = concatenate([cnn_out, kmer_out, other_features])

    # Add a dense layer with multiple neurons
    dense_out = Dense(64, activation='relu')(concatenated)

    output = Dense(1, activation='sigmoid')(dense_out)

    model = Model(inputs=[input_seq, kmer_input, other_features], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_model(input_shape, vocab_size, embedding_dim, input_length):
    input_seq = Input(shape=(input_shape, 1))
    kmer_input = Input(shape=(input_length,))
    other_features = Input(shape=(6,))

    lstm_out = LSTM(50)(input_seq)

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(kmer_input)
    kmer_out = Flatten()(embedding_layer)

    concatenated = concatenate([lstm_out, kmer_out, other_features])

  
    dense_out = Dense(64, activation='relu')(concatenated)

    output = Dense(1, activation='sigmoid')(dense_out)

    model = Model(inputs=[input_seq, kmer_input, other_features], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_bilstm_model(input_shape, vocab_size, embedding_dim, input_length):
    input_seq = Input(shape=(input_shape, 1))
    kmer_input = Input(shape=(input_length,))
    other_features = Input(shape=(6,))

    bilstm_out = Bidirectional(LSTM(50))(input_seq)

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(kmer_input)
    kmer_out = Flatten()(embedding_layer)

    concatenated = concatenate([bilstm_out, kmer_out, other_features])

  
    dense_out = Dense(64, activation='relu')(concatenated)

    output = Dense(1, activation='sigmoid')(dense_out)

    model = Model(inputs=[input_seq, kmer_input, other_features], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_gru_model(input_shape, vocab_size, embedding_dim, input_length):
    input_seq = Input(shape=(input_shape, 1))
    kmer_input = Input(shape=(input_length,))
    other_features = Input(shape=(6,))

    gru_out = GRU(50)(input_seq)

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(kmer_input)
    kmer_out = Flatten()(embedding_layer)

    concatenated = concatenate([gru_out, kmer_out, other_features])

  
    dense_out = Dense(64, activation='relu')(concatenated)

    output = Dense(1, activation='sigmoid')(dense_out)

    model = Model(inputs=[input_seq, kmer_input, other_features], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Model Definitions - Hybrid Models
def create_cnn_lstm_model(input_shape, vocab_size, embedding_dim, input_length):
    input_seq = Input(shape=(input_shape, 1))
    kmer_input = Input(shape=(input_length,))
    other_features = Input(shape=(6,))

    cnn_out = Conv1D(64, kernel_size=3, activation='relu')(input_seq)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    lstm_out = LSTM(50)(cnn_out)

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(kmer_input)
    kmer_out = Flatten()(embedding_layer)

    concatenated = concatenate([lstm_out, kmer_out, other_features])

    dense_out = Dense(64, activation='relu')(concatenated)

    output = Dense(1, activation='sigmoid')(dense_out)

    model = Model(inputs=[input_seq, kmer_input, other_features], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_bilstm_model(input_shape, vocab_size, embedding_dim, input_length):
    input_seq = Input(shape=(input_shape, 1))
    kmer_input = Input(shape=(input_length,))
    other_features = Input(shape=(6,))

    cnn_out = Conv1D(64, kernel_size=3, activation='relu')(input_seq)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    bilstm_out = Bidirectional(LSTM(50))(cnn_out)

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(kmer_input)
    kmer_out = Flatten()(embedding_layer)

    concatenated = concatenate([bilstm_out, kmer_out, other_features])

    dense_out = Dense(64, activation='relu')(concatenated)

    output = Dense(1, activation='sigmoid')(dense_out)

    model = Model(inputs=[input_seq, kmer_input, other_features], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_gru_model(input_shape, vocab_size, embedding_dim, input_length):
    input_seq = Input(shape=(input_shape, 1))
    kmer_input = Input(shape=(input_length,))
    other_features = Input(shape=(6,))

    cnn_out = Conv1D(64, kernel_size=3, activation='relu')(input_seq)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    gru_out = GRU(50)(cnn_out)

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(kmer_input)
    kmer_out = Flatten()(embedding_layer)

    concatenated = concatenate([gru_out, kmer_out, other_features])

    dense_out = Dense(64, activation='relu')(concatenated)

    output = Dense(1, activation='sigmoid')(dense_out)

    model = Model(inputs=[input_seq, kmer_input, other_features], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Arch 3 Common Code

class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        e = tf.keras.backend.squeeze(e, axis=-1)
        alpha = tf.keras.backend.softmax(e)
        alpha = tf.keras.backend.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = tf.keras.backend.sum(context, axis=1)
        return context

# Model Definitions - Hybrid Models with Attention
def create_cnn_lstm_attention_model(input_shape, vocab_size, embedding_dim, input_length):
    input_seq = Input(shape=(input_shape, 1))
    kmer_input = Input(shape=(input_length,))
    other_features = Input(shape=(6,))

    cnn_out = Conv1D(64, kernel_size=3, activation='relu')(input_seq)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    cnn_out = AttentionLayer()(cnn_out)  
    cnn_out = tf.keras.layers.Reshape((cnn_out.shape[1], 1))(cnn_out) 
    lstm_out = LSTM(50, return_sequences=True)(cnn_out)
    lstm_out = Flatten()(lstm_out)  

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(kmer_input)
    kmer_out = Flatten()(embedding_layer) 

    concatenated = concatenate([lstm_out, kmer_out, other_features])

    dense_out = Dense(64, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense_out)

    model = Model(inputs=[input_seq, kmer_input, other_features], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_bilstm_attention_model(input_shape, vocab_size, embedding_dim, input_length):
    input_seq = Input(shape=(input_shape, 1))
    kmer_input = Input(shape=(input_length,))
    other_features = Input(shape=(6,))

    cnn_out = Conv1D(64, kernel_size=3, activation='relu')(input_seq)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    cnn_out = AttentionLayer()(cnn_out) 
    cnn_out = tf.keras.layers.Reshape((cnn_out.shape[1], 1))(cnn_out)  
    bilstm_out = Bidirectional(LSTM(50, return_sequences=True))(cnn_out)
    bilstm_out = Flatten()(bilstm_out) 

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(kmer_input)
    kmer_out = Flatten()(embedding_layer) 

    concatenated = concatenate([bilstm_out, kmer_out, other_features])

    dense_out = Dense(64, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense_out)

    model = Model(inputs=[input_seq, kmer_input, other_features], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_gru_attention_model(input_shape, vocab_size, embedding_dim, input_length):
    input_seq = Input(shape=(input_shape, 1))
    kmer_input = Input(shape=(input_length,))
    other_features = Input(shape=(6,))

    cnn_out = Conv1D(64, kernel_size=3, activation='relu')(input_seq)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    cnn_out = AttentionLayer()(cnn_out)  
    cnn_out = tf.keras.layers.Reshape((cnn_out.shape[1], 1))(cnn_out)  
    gru_out = GRU(50, return_sequences=True)(cnn_out)
    gru_out = Flatten()(gru_out)  

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(kmer_input)
    kmer_out = Flatten()(embedding_layer) 

    concatenated = concatenate([gru_out, kmer_out, other_features])

    dense_out = Dense(64, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense_out)

    model = Model(inputs=[input_seq, kmer_input, other_features], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


