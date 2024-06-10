model_weights_path = "/home/tako4/capstone/backend/Backend/ai_model/model_weights"


class LDAModelConfig:
    NUM_OF_CATEGORY = range(3, 21)
    NUM_OF_TOPICS_BY_GROUP = range(5, 11)
    PASSES = 100


class RegressionModelConfig:
    RIDGE_PARAMETERS = {"ridge__alpha": [1e-4, 1e-2, 1, 10, 100], "ridge__random_state": [25]}
    LASSO_PARAMETERS = {"lasso__alpha": [1e-4, 1e-2, 1, 10, 100], "lasso__random_state": [25]}


class BaseConfig:
    TEST_SIZE = 0.3
    RANDOM_STATE = 25


class VolaConfig:
    TIME_INTERVALS = [1, 5, 15, 60, 1440]
    VOLA_COLUMNS = [f"vola_{time}m" for time in TIME_INTERVALS]
