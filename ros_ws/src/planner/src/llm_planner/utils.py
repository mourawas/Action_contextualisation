import os
import glob

LOG_DIR_NAME = 'log'


def get_log_folder():
    # Get the current directory
    current_dir = os.path.abspath(__file__)

    # Move one directory up
    parent_dir = os.path.dirname(current_dir)

    # Navigate to the log directory
    log_dir = os.path.join(parent_dir, LOG_DIR_NAME)

    return log_dir


def create_experiment_log(extension = ""):

    log_dir = get_log_folder()

    # Create the log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Find existing experiment folders
    experiment_folders = glob.glob(os.path.join(log_dir, f'experiment_{extension}_*'))

    # Determine the next experiment number
    if experiment_folders:
        experiment_numbers = [int(folder.split('_')[-1]) for folder in experiment_folders]
        next_experiment_number = max(experiment_numbers) + 1
    else:
        next_experiment_number = 0

    # Create the experiment folder
    experiment_folder = os.path.join(log_dir, f'experiment_{extension}_{next_experiment_number}')
    os.makedirs(experiment_folder)

    return experiment_folder

