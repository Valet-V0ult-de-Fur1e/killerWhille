from enum import IntEnum


class DatasetType(IntEnum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


DATA_PATH = r'Dataset v2'
OUTPUT_PATH = r'results'
WEIGHTS_PATH = r'weights/vggish_audioset_weights_without_fc2.h5'
WEIGHTS_PATH_TOP = r'weights/vggish_audioset_weights.h5'
BEST_WEIGHT_FILE_NAME = 'weights/weights.best.hdf5'
TENSORBOARD_BASE_FOLDER = 'tensorboard'

# Classification params
OTHER_CLASS = 'Other'
OTHER_CLASSES = []
# REMOVE_CLASSES = [
#     'AtlanticSpottedDolphin'
#     , 'BeardedSeal'
#     , 'Beluga_WhiteWhale'
#     , 'BottlenoseDolphin'
#     , 'Boutu_AmazonRiverDolphin'
#     , 'BowheadWhale'
#     , 'ClymeneDolphin'
#     , 'CommonDolphin'
#     , 'DallsPorpoise'
#     , 'DuskyDolphin'
#     , 'FalseKillerWhale'
#     , 'Fin_FinbackWhale'
#     , 'FrasersDolphin'
#     , 'Grampus_RissosDolphin'
#     , 'GrayWhale'
#     , 'HarborPorpoise'
#     , 'HarpSeal'
#     , 'HumpbackWhale'
#     , 'LeopardSeal'
#     , 'LongBeakedPacificCommonDolphin'
#     , 'Long_FinnedPilotWhale'
#     , 'MelonHeadedWhale'
#     , 'Narwhal'
#     , 'NorthernRightWhale'
#     , 'PantropicalSpottedDolphin'
#     , 'RossSeal'
#     , 'Rough_ToothedDolphin'
#     , 'Short_FinnedPacificPilotWhale'
#     , 'SouthernRightWhale'
#     , 'SpinnerDolphin'
#     , 'StripedDolphin'
#     , 'Walrus'
#     , 'WeddellSeal'
#     , 'WestIndianManatee'
#     , 'White_beakedDolphin'
#     , 'White_sidedDolphin'
#     , 'HeavisidesDolphin'
#     , 'MinkeWhale'
#     , 'RibbonSeal'
#     , 'RingedSeal'
#     , 'SpottedSeal'
#     , 'TucuxiDolphin'
# ]

REMOVE_CLASSES = [#'BeardedSeal',  # (NEED TO UNDERSTAND)
                  'BlueWhale',  # 3 training samples
                  'CommersonsDolphin',  # 1 training sample
                  'FinlessPorpoise',  # 2 training samples
                  'GraySeal',  # 7 training samples
                  'HarbourSeal',  # 1 training sample
                  'HeavisidesDolphin',  # 14 training samples
                  'HoodedSeal',  # 2 training samples
                  'IrawaddyDolphin',  # 5 training samples
                  'JuanFernandezFurSeal',  # 4 training samples
                  #'LeopardSeal',  # (NEED TO UNDERSTAND)
                  'MinkeWhale',  # 24 training samples
                  'NewZealandFurSeal',  # 2 training samples
                  'RibbonSeal',  # 45 training samples
                  'RingedSeal',  # 46 training samples
                  'SeaOtter',  # 2 training samples
                  #'Short_FinnedPacificPilotWhale',  # (NEED TO UNDERSTAND)
                  'SpottedSeal',  # 22 training samples
                  'StellerSeaLion',  # 6 training samples
                  'TucuxiDolphin',  # 12 training samples
    'HarpSeal',
    'BeardedSeal',          # BAD f1-score
    'LeopardSeal',
    'Narwhal'
                 ]

LOSS = 'categorical_crossentropy'
EPOCHS = 100
BATCH_SIZE = 64
MODEL_NAMES = ['vggish', 'logreg']
DEFAULT_MODEL_NAME = 'vggish'

# Model training - VGGish params
OPTIMIZER = 'adam'
LEARNING_RATE = 0.0004  # SGD default LR=0.01; Adam default LR=0.001 #TODO: before 12.04 15:28 was 0.001
DROPOUT = 0.4 #TODO: test 0.2?? was 0.4 (мб батч нормализацию выкинуть?)
FINAL_DENSE_NODES = 256
L2_REG_RATE = 0.01  # used for all Dense and Conv2D layers

# Model training - LogReg params
LOGREG_OPTIMIZER = 'adam'
LOGREG_LEARNING_RATE = 0.005

# EarlyStopping
EARLY_STOPPING_PATIENCE = 10

# Mel spectrogram hyperparameters
FILE_MAX_SIZE_SECONDS = 10.00
FILE_SAMPLING_SIZE_SECONDS = 0.98
