import os

path_t, file_t = os.path.split(os.path.abspath(__file__))
file_settings = os.path.join(path_t, 'settings')


def get_path_file_figures():
    with open(file_settings) as f:
        pff = f.readlines()[0].strip()
    return pff


def get_path_file_tables():
    with open(file_settings) as f:
        pft = f.readlines()[1].strip()
    return pft

