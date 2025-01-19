
def case_dir_path_error(logger):
    
    error_message = "Case directory path is mandatory. Please provide a valid path."
    if logger:
        logger.error(error_message)
    raise ValueError(error_message)


def preprocess_file_path_error(logger):
    
    error_message = "preprocess_file_path is mandatory. Please provide a valid path."
    if logger:
        logger.error(error_message)
    raise ValueError(error_message)


def classifiers_path_error(logger):
    
    error_message = "classifiers_path is mandatory. Please provide a valid path."
    if logger:
        logger.error(error_message)
    raise ValueError(error_message)

