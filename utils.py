import shutil


def copy_file(source, destination):
    shutil.copy2(source, destination)
    return
