import tensorflow as tf


class HighwayDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=tf.nn.relu):
        """Implements Highway Networks (fully-connected networks inspired by gates from LSTMs)

        Args:
            units (int): Number of output units.
            activation (tf.nn.activation): Activation function, by default tf.nn.relu
        """
        super(HighwayDense, self).__init__(name='HighwayDense')
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w_t = self.add_weight(shape=(input_shape[-1], self.units),
                                   initializer='random_normal',
                                   trainable=True,
                                   name='weight_transform')
        self.b_t = self.add_weight(shape=(self.units,),
                                   initializer='random_normal',
                                   trainable=True,
                                   name='bias_transform')
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='weight')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='bias')

    def call(self, inputs):
        x = inputs
        T = tf.sigmoid(tf.matmul(x, self.w_t) + self.b_t)
        H = self.activation(tf.matmul(x, self.w) + self.b)
        C = 1.0 - T
        return tf.multiply(H, T) + tf.multiply(x, C)


class PreNet(tf.keras.Model):
    def __init__(self, units):
        """Does a projection of inputs into another space.

        Args:
            units (list): Number of units per layer.
        """
        super(PreNet, self).__init__(name='PreNet')
        self.units = units
        self.num_layers = len(self.units)

        self.dense_layers = [tf.keras.layers.Dense(units, activation='relu') for units in self.units]
        self.dropout_layers = [tf.keras.layers.Dropout(0.5) for _ in range(self.num_layers)]

    def call(self, inputs, is_training):
        x = inputs

        for i in range(self.num_layers):
            x = self.dense_layers[i](x)
            x = self.dropout_layers[i](x, training=is_training)

        return x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape.append(self.units[-1])
        return tf.TensorShape(shape)


class CBHG(tf.keras.Model):
    def __init__(self, K, conv_bank_units, conv_projection_units, highway_units, gru_units):
        """Subnetwork containing bank of convolutional filters, followed by highway network and bidirectional GRU.

        Args:
            K (int): Number of sets of convolutional filters of size 1...K.
            conv_bank_units (int): Number of units per layer in convolutional bank.
            conv_projection_units (list): Number of units in convolutional projections layers.
            highway_units (list): Number of units in each layer in highway network.
            gru_units (int): Number of units in GRU RNN.
        """
        super(CBHG, self).__init__(name='CBHG')
        self.K = K
        self.out_shape = gru_units
        self.highway_units = highway_units

        self.conv_filter_bank = [tf.keras.layers.Conv1D(conv_bank_units, k, padding='same') for k in range(1, self.K + 1)]
        self.batch_norm_0 = tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPool1D(2, 1, 'same')

        self.num_proj_convs = len(conv_projection_units)
        self.proj_convs = [tf.keras.layers.Conv1D(units, 3, padding='same') for units in conv_projection_units]
        self.proj_bns = [tf.keras.layers.BatchNormalization() for _ in range(self.num_proj_convs)]

        # projects output from projection convs to size of highway net if there is a mismatch (postprocessing cbhg)
        self.projection_highway_connector = tf.keras.layers.Dense(highway_units[0])

        self.num_highway_layers = len(highway_units)
        self.highway_layers = [HighwayDense(units, tf.nn.relu) for units in highway_units]

        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(gru_units, return_state=True, return_sequences=True),
            merge_mode='sum'
        )

    def call(self, inputs, is_training):
        """
        Args:
            inputs (tf.EagerTensor): Outputs from PreNet, shape: [batch, timesteps, prenet_projection_size]
            is_training (bool): Boolean switch for dropout.

        Returns:
            tf.EagerTensor: Processed inputs, shaped [batch, timesteps, projection_size (128)]
        """
        # convolution banks and max pooling
        x = [self.conv_filter_bank[i](inputs) for i in range(self.K)]
        x = tf.concat(x, -1)
        x = self.batch_norm_0(x, training=is_training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)

        # convolutional projection
        for i in range(self.num_proj_convs - 1):
            x = self.proj_convs[i](x)
            x = self.proj_bns[i](x, training=is_training)
            x = tf.nn.relu(x)
        x = self.proj_convs[-1](x)
        x = self.proj_bns[-1](x, training=is_training)

        # residual connection
        x = x + inputs

        # add additional projection layer in postprocessing CBHG to connect outputs from projection convs to inputs of highway network (not stated in paper)
        if x.shape[-1] != self.highway_units:
            x = self.projection_highway_connector(x)

        # highway network
        for i in range(self.num_highway_layers):
            x = self.highway_layers[i](x)

        # bidirectional, residual GRU
        rnn_out, forward_state, backward_state = self.gru(x)
        rnn_state = tf.concat([forward_state, backward_state], -1)
        return rnn_out, rnn_state

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape.append(self.out_shape)
        return tf.TensorShape(shape)


class Encoder(tf.keras.Model):
    def __init__(self,
                 vocabulary_size,
                 embedding_size,
                 prenet_units,
                 k,
                 conv_bank_units,
                 conv_proj_units,
                 highway_units,
                 gru_units):
        """Initializes Encoder Recurrent Network.

        Args:
            vocabulary_size (int): Size of vocabulary.
            embedding_size (int): Size of embeddings.
            prenet_units (list): List of neurons per prenet layer.
            k (int): Number of convolutional filters in bank.
            conv_conv_bank_units (int): Number of neurons in conv bank layers.
            conv_proj_units (list): Number of neurons per conv projection layer.
            highway_units (list): Number of neurons in each highway layer.
            gru_units (int): Number of neurons in single GRU layer.
        """
        super(Encoder, self).__init__(name='Encoder')

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.prenet_units = prenet_units
        self.K = k
        self.conv_bank_units = conv_bank_units
        self.conv_proj_units = conv_proj_units
        self.highway_units = highway_units
        self.gru_units = gru_units

        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.vocabulary_size,
            output_dim=self.embedding_size,
            mask_zero=True
        )
        self.prenet = PreNet(
            units=self.prenet_units
        )
        self.cbhg = CBHG(
            K=self.K,
            conv_bank_units=self.conv_bank_units,
            conv_projection_units=self.conv_proj_units,
            highway_units=self.highway_units,
            gru_units=self.gru_units
        )

    def __call__(self, inputs, is_training):
        """
        Args:
            inputs (tf.EagerTensor): Indices of input symbols (labels), shape: [batch, timesteps]
            is_training (bool): Boolean switch for dropout.

        Returns:
            tf.EagerTensor: Processed inputs, shaped [batch, timesteps, bottleneck_size (128)]
        """
        x = self.embedding_layer(inputs)
        x = self.prenet(x, is_training=is_training)
        x = self.cbhg(x, is_training=is_training)
        return x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape.append(self.gru_units)
        return tf.TensorShape(shape)


class AttentionModule(tf.keras.Model):
    """
    Attention module computes context vectors and attention scores. Based on https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention#translate.
    """

    def __init__(self, units):
        super(AttentionModule, self).__init__(name='AttentionModule')
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        """Computes context vector for attention and attention scores

        Args:
            query: Decoder hidden state.
            values: Encoder outputs.

        Returns:
            context_vector: Softmax-normalized context vector over encoder outputs.
            attention_scores: Score for each input (returned for debugging purposes).
        """
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(
            tf.nn.tanh(
                self.W1(values)
                + self.W2(hidden_with_time_axis)
            )
        )

        # attention_weights shape == (batch_size, max_length, 1)
        attention_scores = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_scores * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # return attention weights for debugging purposes
        return context_vector, attention_scores


class DecoderRNN(tf.keras.Model):
    def __init__(self,
                 prenet_units,
                 attention_units,
                 gru_units,
                 n_mels,
                 r):
        """Initializes Decoder Recurrent Network.

        Args:
            prenet_units (list): List of neurons per prenet layer.
            attention_units (int): List of neurons in attention layer.
            gru_units (list): Number of neurons in each GRU layer.
            n_mels (int): Number of mels, denoting final output size.
            r (int): Number of steps to predict (a.k.a. reduction factor).
        """

        super(DecoderRNN, self).__init__(name='DecoderRNN')
        self.prenet_units = prenet_units
        self.attention_units = attention_units
        self.gru_units = gru_units
        self.n_mels = n_mels
        self.r = r

        self.prenet = PreNet(
            units=self.prenet_units
        )

        self.attention = AttentionModule(self.attention_units)
        self.attention_gru = tf.keras.layers.GRU(
            self.attention_units,
            return_sequences=True,
            return_state=True
        )

        self.num_gru_layers = len(self.gru_units)
        self.gru_layers = [tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True
        ) for units in self.gru_units]

        self.projection_layer = tf.keras.layers.Dense(self.n_mels * self.r)

    def call(self, inputs, prev_hidden_states, enc_outputs, is_training):
        """Computes single decoder output.

        Args:
            inputs: Output of previous step (i.e. mel-spectrograms generated by decoder in previous step)
            prev_hidden_states: Hidden states for attention layer and all GRU layers from previous step (num_gru_layers + 1 tensors).
            enc_outputs: Outputs from encoder (all time steps)
            is_training: bool flag, mainly for prenet.

        Returns:
            outputs: Mel predictions for r steps, of shape (batch size, r, n_mels)
            hidden_states: Current hidden states for attention layer and all GRU layers (num_gru_layers + 1 tensors).
            attention_score: Scores for each encoder output, of shape (batch_size, encoder_outputs).
        """
        prenet_out = self.prenet(inputs, is_training=is_training)

        # attention rnn
        context_vector, attention_score = self.attention(prev_hidden_states[0], enc_outputs)
        context_vector = tf.expand_dims(context_vector, 1)
        attention_rnn_in = tf.concat([context_vector, prenet_out], axis=-1)
        attention_rnn_out, attention_state = self.attention_gru(attention_rnn_in)
        attention_out = tf.concat([context_vector, attention_rnn_out], -1)
        out_hidden_states = [attention_state]

        # multilayer gru
        x = attention_out
        for i in range(self.num_gru_layers):
            x, res_state = self.gru_layers[i](x, initial_state=prev_hidden_states[i + 1])
            out_hidden_states.append(res_state)

        # final mel-projection layer
        outputs = self.projection_layer(x)
        outputs = tf.reshape(outputs, [-1, self.r, self.n_mels])

        return outputs, out_hidden_states, tf.squeeze(attention_score, -1)
