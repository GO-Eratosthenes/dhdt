import os


def get_list_files(dat_path, ending):
    '''
    generate a list with files, with a given ending/extension
    '''
    c = len(ending)
    file_list = [x for x in os.listdir(dat_path) if x[-c:]==ending]
    return file_list
