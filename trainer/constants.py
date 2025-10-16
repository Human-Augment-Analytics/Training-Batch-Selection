import os

# ========== Directory Paths ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION_DATA_DIR = os.path.join(BASE_DIR, 'trainer/data/vision')
OUTPUT_DIR = os.path.join(BASE_DIR, 'trainer/pipelines/vision/output')

# ========== Data Files ==========
TRAIN_CSV = os.path.join(VISION_DATA_DIR, 'mnist_train.csv')
TEST_CSV  = os.path.join(VISION_DATA_DIR, 'mnist_test.csv')

# ========== Model ==========
INPUT_DIM = 784
HIDDEN_DIM = 128
NUM_CLASSES = 10

# ========== Training ==========
EPOCHS = 5
BATCH_SIZE = 64
N_RUNS = 5
DEVICE = 'cpu'

# ========== Smart Batch ==========
MOVING_AVG_DECAY = 0.9
EXPLORE_FRAC = 0.5
TOP_K_FRAC = 0.2

# ========== Random Seed ==========
RANDOM_SEED = 2024