import pickle

import constants
from data_utils import *
from evaluate.bc5 import evaluate_bc5
from models.model_cnn import CnnModel
from dataset import Dataset


def main():

    print('Build data')
    # load vocabs
    vocab_words = load_vocab(constants.ALL_WORDS)
    vocab_poses = load_vocab(constants.ALL_POSES)
    vocab_depends = load_vocab(constants.ALL_DEPENDS)
    train = Dataset('data/raw_data/norm_corenlp_sdp_data.train.txt', vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_depends=vocab_depends)
    pickle.dump(train, open(constants.PICKLE_DATA + 'train.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
    test = Dataset('data/raw_data/norm_corenlp_sdp_data.test.txt', vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_depends=vocab_depends)
    pickle.dump(test, open(constants.PICKLE_DATA + 'test.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
    # exit(0)


    # print('Re-Load data')
    # train = pickle.load(open(constants.PICKLE_DATA + 'train.pickle', 'rb'))
    # test = pickle.load(open(constants.PICKLE_DATA + 'test.pickle', 'rb'))
    # train = open(constants.RAW_DATA + 'norm_corenlp_sdp_data.train.txt', 'rb')
    # test = open(constants.RAW_DATA + 'norm_corenlp_sdp_data.test.txt', 'rb')
    validation = test

    # get pre trained embeddings
    embeddings = get_trimmed_w2v_vectors(constants.TRIMMED_W2V)
    model = CnnModel(
        model_name=constants.MODEL_NAMES.format('cnn', constants.JOB_IDENTITY),
        embeddings=embeddings,
        batch_size=128
    )

    model.build()

    model.load_data(train=train, validation=validation)
    model.run_train(epochs=constants.EPOCHS, early_stopping=constants.EARLY_STOPPING, patience=constants.PATIENCE)

    answer = {}
    identities = test.identities
    y_pred = model.predict(test)
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            if identities[i][0] not in answer:
                answer[identities[i][0]] = []

            if identities[i][1] not in answer[identities[i][0]]:
                answer[identities[i][0]].append(identities[i][1])

    # for identity in identities:
        # for i in range(len(y_pred)):
            # print(answer[identities[i][0]])
    print(
        'result: abstract: ', evaluate_bc5(answer)
    )


if __name__ == '__main__':
    main()
