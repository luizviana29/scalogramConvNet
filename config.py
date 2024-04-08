import os
#Dataset original
ORIG_INPUT_DATASET = os.path.join("C:\\Users\\Acer\\Documents\Mestrado\\Dissertacao\\Database", "training_mexh_4_review")
METRONOMOS_DATASET = os.path.join("C:\\Users\\Acer\\Documents\Mestrado\\Dissertacao\\Database", "metronomos_review")
#TEST_PATH = os.path.join("C:\\Users\\Acer\\Documents\Mestrado\\Dissertacao\\Database", "test_mexh_4_correct")
ACM_MIRUM_PATH = os.path.join("C:\\Users\\Acer\\Documents\Mestrado\\Dissertacao\\Database\\test_mexh_4_review", "ACM_MIRUM")
BALLROOM_PATH= os.path.join("C:\\Users\\Acer\\Documents\Mestrado\\Dissertacao\\Database\\test_mexh_4_review", "BALLROOM")
GIANTSTEPS_PATH= os.path.join("C:\\Users\\Acer\\Documents\Mestrado\\Dissertacao\\Database\\test_mexh_4_review", "GIANTSTEPS")
GTZAN_GENRES_PATH= os.path.join("C:\\Users\\Acer\\Documents\Mestrado\\Dissertacao\\Database\\test_mexh_4_review", "GTZAN_GENRES")
HAINSWORTH_PATH= os.path.join("C:\\Users\\Acer\\Documents\Mestrado\\Dissertacao\\Database\\test_mexh_4_review", "HAINSWORTH")
ISMIR2004_PATH= os.path.join("C:\\Users\\Acer\\Documents\Mestrado\\Dissertacao\\Database\\test_mexh_4_review", "ISMIR2004")
SMC_MIRUM_PATH= os.path.join("C:\\Users\\Acer\\Documents\Mestrado\\Dissertacao\\Database\\test_mexh_4_review", "SMC_MIRUM")
COMBINADOS_PATH= os.path.join("C:\\Users\\Acer\\Documents\Mestrado\\Dissertacao\\Database\\test_mexh_4_review", "COMBINADOS")
TRAIN_DATASET = os.path.join("C:\\Users\\Acer\\Documents\\Mestrado\\Dissertacao\\Database\\training_mexh_4_review_split", "treino5")
VAL_DATASET = os.path.join("C:\\Users\\Acer\\Documents\\Mestrado\\Dissertacao\\Database\\training_mexh_4_review_split", "k1")
#Dataset organizado
BASE_PATH = os.path.join("C:\\Users\\Acer\\Documents\\Mestrado\\Dissertacao\\Database", "training_mexh_4_review_split")
k1_PATH = os.path.sep.join([BASE_PATH, "k1"])
k2_PATH = os.path.sep.join([BASE_PATH, "k2"])
k3_PATH = os.path.sep.join([BASE_PATH, "k3"])
k4_PATH = os.path.sep.join([BASE_PATH, "k4"])
k5_PATH = os.path.sep.join([BASE_PATH, "k5"])
SPLIT=0.2
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 5
INIT_LR = 1e-2
BS = 64 #128
# TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
# VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
# TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
