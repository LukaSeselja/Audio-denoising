WINDOW_SIZE         = 512
HOP_SIZE            = 256
INPUT_DIM           = WINDOW_SIZE // 2 + 1

HIDDEN_DIM          = 64

EPOCHS              = 250
LEARNING_RATE       = 0.02
BATCH_SIZE          = 16
BETA                = 0.9

NOIZEUS_DIR         = "noizeus"
NOIZEUS_N_TRAIN     = 20
NOIZEUS_N_TEST      = 10
NOIZEUS_TOTAL       = 30
 
NOIZEUS_NOISE_TYPES = [
    "car", "babble", "restaurant", "street",
    "airport", "train", "exhibition", "station"
]

MODEL_PATH          = "model_tezine.npz"

RANDOM_SEED         = 42