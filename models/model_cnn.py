import os

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from dataset import pad_sequences
from utils import Timer, Log
from data_utils import countNumRelation, countNumPos

import constants
from sklearn.metrics import f1_score

seed = 13
np.random.seed(seed)


class CnnModel:
    def __init__(self, model_name, embeddings, batch_size):
        self.model_name = model_name
        self.embeddings = embeddings
        self.batch_size = batch_size

        self.max_length = constants.MAX_LENGTH
        self.input_fasttext_dim = constants.INPUT_W2V_DIM
        self.num_of_depend = countNumRelation()
        self.num_of_pos = countNumPos()
        self.num_of_class = len(constants.ALL_LABELS)
        self.all_labels = constants.ALL_LABELS
        self.trained_models = constants.TRAINED_MODELS

    def _add_placeholders(self):
        """
        Adds placeholders to self
        """
        self.labels = tf.placeholder(name="labels", shape=[None], dtype='int32')
        self.word_ids = tf.placeholder(name='word_ids', shape=[None, None], dtype='int32')
        self.pos_ids = tf.placeholder(name='pos_ids', shape=[None, None], dtype='int32')
        self.relations = tf.placeholder(name='relations', shape=[None, None], dtype='int32')
        self.sequence_lens = tf.placeholder(name='sequence_lens', dtype=tf.int32, shape=[None])
        self.sequence_lens_re = tf.placeholder(name='sequence_lens_re', dtype=tf.int32, shape=[None])
        self.dropout_embedding = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_embedding")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.is_training = tf.placeholder(tf.bool, name='phase')

    def _add_word_embeddings_op(self):
        """
        Adds word embeddings to self
        """
        with tf.variable_scope("embedding"):
            embeddings_re = tf.get_variable(name="re_lut", shape=[self.num_of_depend + 1, 300],
                                            initializer=tf.contrib.layers.xavier_initializer(),
                                            dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(1e-4))

            # n_depend = self.num_of_depend + 1
            # embeddings_re = np.zeros((n_depend, 300), dtype=np.float)
            # embeddings_re[np.arange(n_depend), np.arange(n_depend)] = 1.0
            # embeddings_re = tf.Variable(embeddings_re, dtype=tf.float32)

            embedding_wd = tf.Variable(self.embeddings, name="lut", dtype=tf.float32, trainable=False)
            self._word_re_embedding = tf.concat([embedding_wd, embeddings_re], axis=0)
            self.word_re_embeddings = tf.nn.embedding_lookup(self._word_re_embedding, self.word_ids, name="embeddings")
            self.word_re_embeddings = tf.nn.dropout(self.word_re_embeddings, self.dropout_embedding)

            # embeddings_pos4 = tf.get_variable();

            # dynamic_embeddings_wd = tf.Variable(self.embeddings, name="lut_d", dtype=tf.float32, trainable=True)
            # self._dynamic_embeddings = tf.concat([dynamic_embeddings_wd, embeddings_re], axis=0)
            # self.dynamic_embeddings = tf.nn.embedding_lookup(self._dynamic_embeddings, self.word_ids, name="embeddings_d")
            # self.dynamic_embeddings = tf.nn.dropout(self.dynamic_embeddings, self.dropout_embedding)

            embeddings_pos = tf.get_variable(name='pos_lut', shape=[self.num_of_pos + 1, 300],
                                                  initializer=tf.contrib.layers.xavier_initializer(),
                                                  dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            self._pos_re_embedding = tf.concat([embeddings_pos, embeddings_re], axis=0)
            self.pos_re_embeddings = tf.nn.embedding_lookup(self._pos_re_embedding, self.pos_ids, name="embeddings")
            self.pos_re_embeddings = tf.nn.dropout(self.pos_re_embeddings, self.dropout_embedding)

    def _add_logits_op(self):
        """
        Adds logits to self
        """
        with tf.variable_scope("cnn"):
            self.word_re_embeddings = tf.expand_dims(self.word_re_embeddings, -1)
            self.pos_re_embeddings = tf.expand_dims(self.pos_re_embeddings, -1)
            self.all_embeddings = tf.concat([self.word_re_embeddings, self.pos_re_embeddings], axis=-1)
            cnn_outputs = []
            for k in constants.CNN_FILTERS:
                filters = constants.CNN_FILTERS[k]
                cnn_output = tf.layers.conv2d(
                    self.all_embeddings, filters=filters,
                    kernel_size=(k, constants.INPUT_W2V_DIM),
                    strides=(2,2),
                    use_bias=False, padding="valid",
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                )
                cnn_output = tf.reduce_max(tf.nn.tanh(cnn_output), 1)
                cnn_output = tf.reshape(cnn_output, [-1, filters])
                cnn_outputs.append(cnn_output)

            final_cnn_output = tf.concat(cnn_outputs, axis=-1)
            final_cnn_output = tf.nn.dropout(final_cnn_output, self.dropout)

        with tf.variable_scope("logits"):
            hiden_1 = tf.layers.dense(
                inputs=final_cnn_output, units=128, name="hiden_1",
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
            )
            hiden_2 = tf.layers.dense(
                inputs=hiden_1, units=128, name="hiden_2",
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
            )
            self.output = tf.layers.dense(
                inputs=hiden_2, units=self.num_of_class, name="logits",
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
            )
            self.logits = tf.nn.softmax(self.output)

    def _add_loss_op(self):
        """
        Adds loss to self
        """
        with tf.variable_scope('loss_layers'):
            log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.reduce_mean(log_likelihood)
            self.loss += tf.reduce_sum(regularizer)

    def _add_train_op(self):
        """
        Add train_op to self
        """
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope("train_step"):
            tvars = tf.trainable_variables()
            grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 100.0)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.9)
            self.train_op = optimizer.apply_gradients(zip(grad, tvars))

    def build(self):
        timer = Timer()
        timer.start("Building model...")

        self._add_placeholders()
        self._add_word_embeddings_op()
        self._add_logits_op()
        self._add_loss_op()
        self._add_train_op()

        timer.stop()
        # f = tf.summary.FileWriter("summary_relation")
        # f.add_graph(tf.get_default_graph())
        # f.close()
        # exit(0)

    def _loss(self, sess, feed_dict):
        feed_dict = feed_dict
        feed_dict[self.dropout_embedding] = 1.0
        feed_dict[self.dropout] = 1.0
        feed_dict[self.is_training] = False

        return sess.run(self.loss, feed_dict=feed_dict)

    def _next_batch(self, data, num_batch):
        start = 0
        idx = 0
        while idx < num_batch:
            word_ids = data['words'][start:start + self.batch_size]
            pos_ids = data['poses'][start:start + self.batch_size]
            labels = data['labels'][start:start + self.batch_size]
            relation_ids = data['relations'][start:start + self.batch_size]

            relation_ids, sequence_lengths_re = pad_sequences(relation_ids, pad_tok=0, max_sent_length=self.max_length)

            # Word - relation - word
            word_ids, sequence_lengths = pad_sequences(word_ids, pad_tok=0, max_sent_length=self.max_length)
            relation_wd_ids = self.embeddings.shape[0] + relation_ids
            word_relation_ids = np.zeros((word_ids.shape[0], word_ids.shape[1] + relation_wd_ids.shape[1]))
            word_relation_ids[:, ::2] = word_ids
            word_relation_ids[:, 1::2] = relation_wd_ids

            # Pos - relation - pos
            pos_ids, sequence_lengths = pad_sequences(pos_ids, pad_tok=0, max_sent_length=self.max_length)
            relation_pos_ids = self.num_of_pos + 1 + relation_ids
            pos_relation_ids = np.zeros((pos_ids.shape[0], pos_ids.shape[1] + relation_pos_ids.shape[1]))
            pos_relation_ids[:, ::2] = pos_ids
            pos_relation_ids[:, 1::2] = relation_pos_ids

            start += self.batch_size
            idx += 1
            yield word_relation_ids, pos_relation_ids, labels, relation_ids, sequence_lengths, sequence_lengths_re

    def _train(self, epochs, early_stopping=True, patience=10, verbose=True):
        Log.verbose = verbose
        if not os.path.exists(self.trained_models):
            os.makedirs(self.trained_models)

        saver = tf.train.Saver(max_to_keep=2)
        best_loss = 0
        n_epoch_no_improvement = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            num_batch_train = len(self.dataset_train.labels) // self.batch_size + 1

            for e in range(epochs):
                words_shuffled, poses_shuffled, labels_shuffled, relations_shuffled = shuffle(
                    self.dataset_train.words,
                    self.dataset_train.poses,
                    self.dataset_train.labels,
                    self.dataset_train.relations
                )

                data = {
                    'words': words_shuffled,
                    'poses': poses_shuffled,
                    'labels': labels_shuffled,
                    'relations': relations_shuffled
                }

                for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_train)):
                    words, poses, labels, relations, sequence_lengths, sequence_lengths_re = batch
                    feed_dict = {
                        self.word_ids: words,
                        self.pos_ids: poses,
                        self.labels: labels,
                        self.relations: relations,
                        self.sequence_lens: sequence_lengths,
                        self.sequence_lens_re: sequence_lengths_re,
                        self.dropout_embedding: 0.5,
                        self.dropout: 0.5,
                        self.is_training: True
                    }

                    _, _, loss_train = sess.run([self.train_op, self.extra_update_ops, self.loss], feed_dict=feed_dict)
                    # all, _all = sess.run([self.static_embeddings, self._static_embedding], feed_dict=feed_dict)
                    if idx % 20 == 0:
                        Log.log("Iter {}, Loss: {} ".format(idx, loss_train))

                Log.log("End epochs {}".format(e + 1))

                # stop by validation loss
                if early_stopping:
                    num_batch_val = len(self.dataset_validation.labels) // self.batch_size + 1
                    total_loss = []

                    data = {
                        'words': self.dataset_validation.words,
                        'poses': self.dataset_validation.poses,
                        'labels': self.dataset_validation.labels,
                        'relations': self.dataset_validation.relations
                    }

                    for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_val)):
                        words, poses, labels, relations, sequence_lengths, sequence_lengths_re = batch

                        acc, f1 = self._accuracy(sess, feed_dict={
                            self.word_ids: words,
                            self.pos_ids: poses,
                            self.labels: labels,
                            self.relations: relations,
                            self.sequence_lens: sequence_lengths,
                            self.sequence_lens_re: sequence_lengths_re,
                            self.is_training: False
                        })
                        total_loss.append(f1)

                    val_loss = np.mean(total_loss)
                    Log.log("F1: {}".format(val_loss))
                    print(best_loss)
                    if val_loss > best_loss:
                        saver.save(sess, self.model_name)
                        Log.log('Save the model at epoch {}'.format(e + 1))
                        best_loss = val_loss
                        n_epoch_no_improvement = 0
                    else:
                        n_epoch_no_improvement += 1
                        Log.log("Number of epochs with no improvement: {}".format(n_epoch_no_improvement))
                        if n_epoch_no_improvement >= patience:
                            print("Best loss: {}".format(best_loss))
                            break

            if not early_stopping:
                saver.save(sess, self.model_name)

    def _accuracy(self, sess, feed_dict):
        feed_dict = feed_dict
        feed_dict[self.dropout_embedding] = 1.0
        feed_dict[self.dropout] = 1.0
        feed_dict[self.is_training] = False

        logits = sess.run(self.logits, feed_dict=feed_dict)
        accuracy = []
        f1 = []
        predict = []
        exclude_label = []
        for logit, label in zip(logits, feed_dict[self.labels]):
            logit = np.argmax(logit)
            exclude_label.append(label)
            predict.append(logit)
            accuracy += [logit == label]

        f1.append(f1_score(predict, exclude_label, average='macro'))
        return accuracy, np.mean(f1)

    def load_data(self, train, validation):
        """
        :param dataset.Dataset train:
        :param dataset.Dataset validation:
        :return:
        """
        timer = Timer()
        timer.start("Loading data")

        self.dataset_train = train
        self.dataset_validation = validation

        print("Number of training examples:", len(self.dataset_train.labels))
        print("Number of validation examples:", len(self.dataset_validation.labels))
        timer.stop()

    def run_train(self, epochs, early_stopping=True, patience=10):
        timer = Timer()
        timer.start("Training model...")
        self._train(epochs=epochs, early_stopping=early_stopping, patience=patience)
        timer.stop()

    # test
    def predict(self, test):
        """

        :param dataset.Dataset test:
        :return:
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            Log.log("Testing model over test set")
            # a = tf.train.latest_checkpoint(self.model_name)
            saver.restore(sess, self.model_name)

            y_pred = []
            num_batch = len(test.labels) // self.batch_size + 1

            data = {
                'words': test.words,
                'poses': test.poses,
                'labels': test.labels,
                'relations': test.relations
            }

            for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch)):
                words, poses, labels, relations, sequence_lengths, sequence_lengths_re = batch
                feed_dict = {
                    self.word_ids: words,
                    self.pos_ids: poses,
                    self.relations: relations,
                    self.sequence_lens: sequence_lengths,
                    self.sequence_lens_re: sequence_lengths_re,
                    self.dropout_embedding: 1.0,
                    self.dropout: 1.0,
                    self.is_training: False
                }
                logits = sess.run(self.logits, feed_dict=feed_dict)

                for logit in logits:
                    decode_sequence = np.argmax(logit)
                    y_pred.append(decode_sequence)

        return y_pred
