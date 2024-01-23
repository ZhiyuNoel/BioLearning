import os
import os.path
import sys

dir_path = '../UI_files'
tar_path = '../src'

def transPyFile(filename):
    return os.path.splitext(filename)[0] + '.py'


def checkRepeat(tar_filename):
    files = os.listdir(tar_path)
    for filename in files:
        if filename == tar_filename:
            return False

    return True

# 调用系统命令把ui转换成py
def runMain(filename):
    pyfile = transPyFile(filename)
    tar_output = tar_path + '/' + pyfile
    tar_input = dir_path + '/' + filename
    if checkRepeat(pyfile):
        cmd = 'pyuic6 -o {pyfile} {uifile}'.format(pyfile=tar_output, uifile=tar_input)
        print('file created')
        os.system(cmd)
    else:
        os.remove(tar_output)
        cmd = 'pyuic6 -o {pyfile} {uifile}'.format(pyfile=tar_output, uifile=tar_input)
        print('repeat')
        os.system(cmd)

if __name__ == "__main__":
    runMain(sys.argv[1])