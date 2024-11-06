import sys, os, random
import numpy as np
import keras
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers.merge import concatenate
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.metrics import accuracy_score


if len(sys.argv) < 4:
    print('Needs 3 arguments - \n'
          '1. Batch size during training\n'
          '2. Batch size during testing\n'
          '3. No. of epochs\n')
    exit(0)

seed_value = 12321
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
np.random.seed(seed_value)

# todo : command line both for train and test
DATADIR = './model-aware-qpp/input_data_t100/'   # (1)
DATADIR_idf = './model-aware-qpp/exp_hist_with_idf/input_100/'
DATADIR_idf_countidf = './model-aware-qpp/exp_hist_with_idf/input_idfcount_10/'
DATADIR_test = './Model-Aware-QPP/TrainDrmmWithAP/tets_data/'   # (1)

NUMCHANNELS = 10  # todo : command line
# HIDDEN_LAYER_DIM = 16    # (2)
# Num top docs (Default: 10)
K = 10   # (5)
# Num of bottom docs
# L = 5
# M: bin-size (Default: 30)
M = 120  # (6)
K1 = 4
M1 = 30
BATCH_SIZE_TRAIN = int(sys.argv[1])   # (7 - depends on the total no. of ret docs)
BATCH_SIZE_TEST = int(sys.argv[2])
EPOCHS = int(sys.argv[3])  # (8)


class InteractionData:
    # Interaction data of query qid with K top docs -
    # each row vector is a histogram of interaction data for a document

    def __init__(self, qid, dataPathBase=DATADIR_idf):
        self.qid = qid
        histFile = "{}/{}.hist".format(dataPathBase, self.qid)
        # df = pd.read_csv(histFile, delim_whitespace=True, header=None)
        # self.matrix = df.to_numpy()
        histogram = np.genfromtxt(histFile, delimiter=" ")
        self.matrix = histogram[:, 4:]


class PairedInstance:
    def __init__(self, line):
        l = line.strip().split('\t')
        if len(l) > 2:
            self.qid_a = l[0]
            self.qid_b = l[1]
            self.class_label = int(l[2])
        else:
            self.qid_a = l[0]
            self.qid_b = l[1]

    def __str__(self):
        return "({}, {})".format(self.qid_a, self.qid_b)

    def getKey(self):
        return "{}-{}".format(self.qid_a, self.qid_b)


# Separate instances for training/test sets etc. Load only the id pairs.
# Data is loaded later in batches with a subclass of Keras generator
class PairedInstanceIds:
    '''
    Each line in this file should comprise three tab separated fields
    <id1> <id2> <label (1/0)>
    '''

    def __init__(self, idpairLabelsFile):
        self.data = {}

        with open(idpairLabelsFile) as f:
            content = f.readlines()

        # remove whitespace characters like `\n` at the end of each line
        for x in content:
            instance = PairedInstance(x)
            self.data[instance.getKey()] = instance

allPairs_train = PairedInstanceIds(DATADIR + 'train_input/qid_ap.pairs')   # (3)
allPairsList_train = list(allPairs_train.data.values())

allPairs_test = PairedInstanceIds(DATADIR + 'test_input/qid_ap.pairs')    # (4)
allPairsList_test = list(allPairs_test.data.values())

print ('{}/{} pairs for training'.format(len(allPairsList_train), len(allPairsList_train)))
print ('{}/{} pairs for testing'.format(len(allPairsList_test), len(allPairsList_test)))

'''
The files need to be residing in the folder data/
Each file is a matrix of values that's read using 
'''

class PairCmpDataGeneratorTrain(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paired_instances_ids, dataFolder=DATADIR_idf, batch_size=BATCH_SIZE_TRAIN, dim=(4, 30, 1)):
        'Initialization'
        self.paired_instances_ids = paired_instances_ids
        self.dim = dim
        self.batch_size = batch_size
        self.dataDir = dataFolder
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paired_instances_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs = [self.paired_instances_ids[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs)

        return X

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.paired_instances_ids))

    def __data_generation(self, list_IDs):
        'Generates data pairs containing batch_size samples'
        # Initialization
        X = [np.empty((self.batch_size, *self.dim)) for i in range(20)]
        # X_bottom = [np.empty((self.batch_size, *self.dim_bottom)) for i in range(2)]
        # Y_top = [np.empty((self.batch_size, *self.dim_top)) for i in range(2)]
        # Y_bottom = [np.empty((self.batch_size, *self.dim_bottom)) for i in range(2)]
        Z = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.qid_a
            # print('ID-1 : ', a_id)
            b_id = paired_instance.qid_b
            # print('ID-2 : ', b_id)

            # read from the data file and construct the instances
            a_data = InteractionData(a_id, self.dataDir)
            b_data = InteractionData(b_id, self.dataDir)
            for val in range(9):
                X[val][i,] = a_data.matrix[val, :].reshape(4, 30, 1)
                X[val + 10][i,] = b_data.matrix[val, :].reshape(4, 30, 1)
            Z[i] = paired_instance.class_label

        return X, Z


class PairCmpDataGeneratorTest(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paired_instances_ids, dataFolder=DATADIR_idf, batch_size=BATCH_SIZE_TRAIN, dim=(4, 30, 1)):
        'Initialization'
        self.paired_instances_ids = paired_instances_ids
        self.dim = dim
        self.batch_size = batch_size
        self.dataDir = dataFolder
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paired_instances_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs = [self.paired_instances_ids[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs)

        return X

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.paired_instances_ids))

    def __data_generation(self, list_IDs):
        'Generates data pairs containing batch_size samples'
        # Initialization
        X = [np.empty((self.batch_size, *self.dim)) for i in range(20)]
        # X_bottom = [np.empty((self.batch_size, *self.dim_bottom)) for i in range(2)]
        # Y_top = [np.empty((self.batch_size, *self.dim_top)) for i in range(2)]
        # Y_bottom = [np.empty((self.batch_size, *self.dim_bottom)) for i in range(2)]
        # Z = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.qid_a
            # print('ID-1 : ', a_id)
            b_id = paired_instance.qid_b
            # print('ID-2 : ', b_id)

            # read from the data file and construct the instances
            a_data = InteractionData(a_id, self.dataDir)
            b_data = InteractionData(b_id, self.dataDir)
            for val in range(9):
                X[val][i,] = a_data.matrix[val, :].reshape(4, 30, 1)
                X[val + 10][i,] = b_data.matrix[val, :].reshape(4, 30, 1)
            # Z[i] = paired_instance.class_label

        return X


def build_siamese(input_shape_init):
    input_a1 = Input(shape=input_shape_init, dtype='float32')
    input_a2 = Input(shape=input_shape_init, dtype='float32')
    input_a3 = Input(shape=input_shape_init, dtype='float32')
    input_a4 = Input(shape=input_shape_init, dtype='float32')
    input_a5 = Input(shape=input_shape_init, dtype='float32')
    input_a6 = Input(shape=input_shape_init, dtype='float32')
    input_a7 = Input(shape=input_shape_init, dtype='float32')
    input_a8 = Input(shape=input_shape_init, dtype='float32')
    input_a9 = Input(shape=input_shape_init, dtype='float32')
    input_a10 = Input(shape=input_shape_init, dtype='float32')

    input_b1 = Input(shape=input_shape_init, dtype='float32')
    input_b2 = Input(shape=input_shape_init, dtype='float32')
    input_b3 = Input(shape=input_shape_init, dtype='float32')
    input_b4 = Input(shape=input_shape_init, dtype='float32')
    input_b5 = Input(shape=input_shape_init, dtype='float32')
    input_b6 = Input(shape=input_shape_init, dtype='float32')
    input_b7 = Input(shape=input_shape_init, dtype='float32')
    input_b8 = Input(shape=input_shape_init, dtype='float32')
    input_b9 = Input(shape=input_shape_init, dtype='float32')
    input_b10 = Input(shape=input_shape_init, dtype='float32')

    encoded_list_a = []
    # for i in range(10):
    # input_a = Input(shape=input_shape_init, dtype='float32')
    matrix_encoder = Sequential(name='sequence_a')
    matrix_encoder.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape_init))
    matrix_encoder.add(MaxPooling2D(padding='same'))
    matrix_encoder.add(Dropout(0.2))
    matrix_encoder.add(Flatten())
    encoded_list_a.append(matrix_encoder(input_a1))
    encoded_list_a.append(matrix_encoder(input_a2))
    encoded_list_a.append(matrix_encoder(input_a3))
    encoded_list_a.append(matrix_encoder(input_a4))
    encoded_list_a.append(matrix_encoder(input_a5))
    encoded_list_a.append(matrix_encoder(input_a6))
    encoded_list_a.append(matrix_encoder(input_a7))
    encoded_list_a.append(matrix_encoder(input_a8))
    encoded_list_a.append(matrix_encoder(input_a9))
    encoded_list_a.append(matrix_encoder(input_a10))

    encoded_list_b = []
    matrix_encoder_1 = Sequential(name='sequence_b')
    matrix_encoder_1.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape_init))
    matrix_encoder_1.add(MaxPooling2D(padding='same'))
    matrix_encoder_1.add(Dropout(0.2))
    matrix_encoder_1.add(Flatten())
    encoded_list_b.append(matrix_encoder_1(input_b1))
    encoded_list_b.append(matrix_encoder_1(input_b2))
    encoded_list_b.append(matrix_encoder_1(input_b3))
    encoded_list_b.append(matrix_encoder_1(input_b4))
    encoded_list_b.append(matrix_encoder_1(input_b5))
    encoded_list_b.append(matrix_encoder_1(input_b6))
    encoded_list_b.append(matrix_encoder_1(input_b7))
    encoded_list_b.append(matrix_encoder_1(input_b8))
    encoded_list_b.append(matrix_encoder_1(input_b9))
    encoded_list_b.append(matrix_encoder_1(input_b10))

    merged_vector_a = concatenate(list(encoded_list_a), axis=-1, name='concatenate_a')
    merged_vector_b = concatenate(list(encoded_list_b), axis=-1, name='concatenate_b')
    merged_vector = concatenate([merged_vector_a, merged_vector_b], axis=-1, name='concatenate_final')

    dense1 = Dense(10, activation='relu')(merged_vector)
    predictions = Dense(1, activation='sigmoid')(dense1)

    siamese_net = Model([input_a1, input_a2, input_a3, input_a4, input_a5, input_a6, input_a7, input_a8, input_a9, input_a10,
                         input_b1, input_b2, input_b3, input_b4, input_b5, input_b6, input_b7, input_b8, input_b9, input_b10], outputs=predictions)

    return siamese_net

siamese_model = build_siamese((K1, M1, 1))
siamese_model.compile(loss = keras.losses.BinaryCrossentropy(),
                      optimizer = keras.optimizers.Adam(),
                      metrics=['accuracy'])
siamese_model.summary()

training_generator = PairCmpDataGeneratorTrain(allPairsList_train, dataFolder=DATADIR_idf+'train_input/')
siamese_model.fit_generator(generator=training_generator,
                            use_multiprocessing=True,
                            epochs=EPOCHS,
                            workers=4)
                            # validation_split=0.2,
                            # verbose=1)

# siamese_model.save_weights('/store/causalIR/model-aware-qpp/foo.weights')
test_generator = PairCmpDataGeneratorTest(allPairsList_test, dataFolder=DATADIR_idf+'test_input/')
predictions = siamese_model.predict(test_generator)  # just to test, will rerank LM-scored docs
# print('predict ::: ', predictions)
# print('predict shape ::: ', predictions.shape)
with open(DATADIR + "16april.test.res", 'w') as outFile:     # (9)
    i = 0
    for entry in test_generator.paired_instances_ids:
        if predictions[i][0] >= 0.45:
            outFile.write(entry.qid_a + '\t' + entry.qid_b + '\t' + str(round(predictions[i][0], 4)) + '\t' + '1\n')
        else:
            outFile.write(entry.qid_a + '\t' + entry.qid_b + '\t' + str(round(predictions[i][0], 4)) + '\t' + '0\n')
        i += 1
outFile.close()

# measure accuracy
gt_file = np.genfromtxt(DATADIR + 'test_input/trec8_ap.pairs.gt', delimiter='\t')    # (10)
actual = gt_file[:, 2:]
# print(actual)

predict_file = np.genfromtxt(DATADIR + '16april.test.res', delimiter='\t')
# predict_file = np.genfromtxt('/store/causalIR/model-aware-qpp/exp_hist_with_idf/nqc.pairs', delimiter='\t')
predict = predict_file[:, 3:]
# print(predict)

score = accuracy_score(actual, predict)
print('Accuracy : ', round(score, 4))
