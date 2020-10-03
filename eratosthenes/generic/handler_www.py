import tarfile
import urllib.request

def url_exist(file_url):
    try: 
        urllib.request.urlopen(file_url).code == 200
        return True
    except:
        return False

def get_tar_file(tar_url, dump_dir):
    ftp_stream = urllib.request.urlopen(tar_url)
    tar_file = tarfile.open(fileobj=ftp_stream, mode="r|gz")
    tar_file.extractall(path=dump_dir)
    tar_names = tar_file.getnames()
    return tar_names

def change_url_resolution(url_string,new_res):
    print('aan de slag')
    
    # get resolution
    props = url_string.split('_')
    for i in range(1,len(props)):
        if props[i][-1] == 'm':
            old_res = props[i]
            props[i] = str(new_res)+'m'
    
    if (old_res=='2m') & (new_res==10):
        # the files are subdivided in quads
        props = props[:-4]+props[-2:]
    
    url_string_2 = '_'.join(props)
    
    folders = url_string_2.split('/')
    for i in range(len(folders)):
        print
        if folders[i] == old_res:
            folders[i] = str(new_res)+'m'
    
    gran_url_new = '/'.join(folders)
    return gran_url_new