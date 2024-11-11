class Config:
    DATA_DIR = 'data/'
    X_TRAIN_FILE = DATA_DIR + 'tox21_dense_train.csv'
    Y_TRAIN_FILE = DATA_DIR + 'tox21_labels_train.csv'
    X_TEST_FILE = DATA_DIR + 'tox21_dense_test.csv'
    Y_TEST_FILE = DATA_DIR + 'tox21_labels_test.csv'
    NUM_HIDDEN_UNITS = 1024
    DROPOUT_RATE = 0.5
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    NUM_EPOCHS = 2