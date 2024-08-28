import os


def create_path(path):
    os.makedirs(path, exist_ok=True)
    return path


