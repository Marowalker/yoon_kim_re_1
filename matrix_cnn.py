import tensorflow as tf


def CNN_model_fn(features, labels, mode, params):
    with tf.variable_scope("embedding"):
        word_idx = features['word']
        embedding = tf.Variable(params['embedding'], name='static_embedding', trainable=False)

        word_embedding = tf.cast(tf.nn.embedding_lookup(embedding, word_idx), tf.float32)

    # Calculate pair wise euclid dist.
    # r = tf.reduce_sum(word_embedding*word_embedding, 2)
    # r = tf.expand_dims(r, -1)
    # embedding_distance = 2*tf.matmul(word_embedding, word_embedding, transpose_b=True)
    # embedding_distance = tf.subtract(r, embedding_distance)
    # embedding_distance = tf.add(embedding_distance, tf.matrix_transpose(r))
    embedding_distance = tf.matmul(word_embedding, word_embedding, transpose_b=True)

    # Create diag matrix from pos.
    pos_embedding = tf.get_variable('pos_embedding', [57, 1], tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.5))
    pos = tf.nn.embedding_lookup(pos_embedding, features['pos'])
    pos = tf.squeeze(pos, axis=-1)
    pos = tf.matrix_diag(pos)

    # Create diag matrix from rela.
    rela_embedding = tf.get_variable('rela_embedding', [188, 1], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.5))
    rela = tf.nn.embedding_lookup(rela_embedding, features['relation'])
    rela = tf.squeeze(rela, axis=-1)
    rela = tf.matrix_diag(rela)
    # Expand rela 2nd and 3rd dim by 1 to match dist and pos.
    rela = tf.pad(rela, [[0, 0], [0, 1], [0,1]])

    # Add channel dim to pos, rela and dist for CNN.
    embedding_distance = tf.expand_dims(embedding_distance, -1)
    rela = tf.expand_dims(rela, -1)
    pos = tf.expand_dims(pos, -1)

    # Concat into 3-channel tensor for CNN.
    all_embedding = tf.concat([embedding_distance, rela, pos], -1)

    cnn_outputs = []
    with tf.variable_scope("cnn"):
        for kernel_size, filter_count in params['cnn']:
            output = tf.layers.conv2d(inputs=all_embedding, filters=filter_count, kernel_size=kernel_size, use_bias=False, padding="valid",
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

            output = tf.reduce_max(tf.nn.tanh(output), axis = 1)
            output = tf.reduce_max(output, axis = 1)
            cnn_outputs.append(output)

        final_cnn_output = tf.concat(cnn_outputs, -1)
    
    with tf.variable_scope('logit'):
        hiden_1 = tf.layers.dense(
            inputs=final_cnn_output, units=32, name="hiden_1",
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        output = tf.layers.dense(
            inputs=hiden_1, units=2, name="logits",
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        logits = tf.nn.softmax(output)
    
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }

        return tf.estimator.EstimatorSpec(mode, predictions = predictions)

    # batch_weight = tf.add(tf.multiply(labels, params['positive_weight']-1), 1)
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=batch_weight)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)