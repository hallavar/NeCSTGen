"""Reuse recurrent state for the existing masked flow LSTM, using its weights."""

import weakref

import tensorflow as tf


_predictors = weakref.WeakKeyDictionary()


def _compile_predictor(model):
    layers = [layer for layer in model.layers if not isinstance(layer, tf.keras.layers.InputLayer)]
    if (len(layers) != 3 or not isinstance(layers[0], tf.keras.layers.Masking)
            or not isinstance(layers[1], tf.keras.layers.LSTM)
            or not isinstance(layers[2], tf.keras.layers.TimeDistributed)
            or not isinstance(layers[2].layer, tf.keras.layers.Dense)):
        raise ValueError("Expected the existing Masking/LSTM/TimeDistributed(Dense) flow model")
    masking, recurrent, projection = layers
    if (recurrent.go_backwards or recurrent.stateful or not recurrent.return_sequences
            or recurrent.dropout or recurrent.recurrent_dropout
            or model.input_shape[-1] != model.output_shape[-1]):
        raise ValueError("Unsupported recurrent flow model configuration")
    dense = projection.layer
    dtype = tf.as_dtype(recurrent.compute_dtype)
    feature_count = int(model.input_shape[-1])
    units = recurrent.units
    preserve_gpu_projection = "GPU:" in recurrent.cell.kernel.device.upper()

    def iterate(count):
        previous = tf.zeros((1, feature_count), dtype=dtype)
        hidden = tf.zeros((1, units), dtype=dtype)
        cell_state = tf.zeros((1, units), dtype=dtype)
        output = tf.TensorArray(dtype, size=count, element_shape=(feature_count,))

        def step(index, previous, hidden, cell_state, output):
            _, states = recurrent.cell(previous, states=[hidden, cell_state], training=False)
            # Match Masking, including the initial all-zero token. A masked
            # token leaves both recurrent states and the previous output alone.
            valid = tf.reduce_any(tf.not_equal(previous, tf.cast(masking.mask_value, dtype)), axis=-1)
            hidden = tf.where(valid[:, None], states[0], hidden)
            cell_state = tf.where(valid[:, None], states[1], cell_state)
            if preserve_gpu_projection:
                # TimeDistributed originally projects every prefix row. GPU
                # TensorFloat-32 kernel selection depends on that row count.
                # Earlier projected rows are unused, so replace them with
                # zeros while preserving the final row's original GPU math.
                projection_input = tf.concat([tf.zeros((index, units), dtype=dtype), hidden], axis=0)
                predicted = tf.ensure_shape(dense(projection_input, training=False)[-1:, :],
                                            (1, feature_count))
            else:
                predicted = dense(hidden, training=False)
            return index + 1, predicted, hidden, cell_state, output.write(index, predicted[0])

        _, _, _, _, output = tf.while_loop(
            lambda index, *_: index < count, step,
            (tf.constant(0), previous, hidden, cell_state, output), parallel_iterations=1)
        return output.stack()

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32)], autograph=False)
    def predict(count):
        # Keep the tiny recurrent kernel beside its weights. In particular,
        # a CPU-loaded LSTM must not copy its states to the GPU at every step.
        with tf.device(recurrent.cell.kernel.device):
            return iterate(count)

    return predict


def autoregressive_flow_features(model, num_packets):
    """Return autoregressive features without recomputing recurrent prefixes.

    States start at zero for every flow. Masking, trained parameters, feature
    projection, and full history are retained; no context window is truncated.
    """
    num_packets = int(num_packets)
    if num_packets < 1:
        raise ValueError("A flow must contain at least one packet")
    predictor = _predictors.get(model)
    if predictor is None:
        predictor = _compile_predictor(model)
        _predictors[model] = predictor
    return predictor(tf.convert_to_tensor(num_packets, dtype=tf.int32)).numpy()
