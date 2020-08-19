from IPython.display import display

import wfdb
from ecg_tmp import *


def get_ecg_data(datfile): 
    ## convert .dat/q1c to numpy arrays
    recordname=os.path.basename(datfile).split(".dat")[0]
    recordpath=os.path.dirname(datfile)
    cwd=os.getcwd()
    os.chdir(recordpath) ## somehow it only works if you chdir. 

    annotator='atr'
    annotation = wfdb.rdann(recordname, extension=annotator, sampfrom=0,sampto = None, pbdir=None)
    record = wfdb.rdsamp(recordname, sampfrom=0,sampto = None) #wfdb.showanncodes()

    Vctrecord=np.transpose(record.p_signals)
    VctAnnotationHot=np.zeros( (2,len(Vctrecord[1])), dtype=np.int)
    VctAnnotationHot[1] = 1;
    
    #print("ecg, 2 lead of shape" , Vctrecord.shape) 
    #print("VctAnnotationHot of shape" , VctAnnotationHot.shape) 
    #print('plotting extracted signal with annotation')
    #wfdb.plotrec(record, annotation=annotation2, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits = 'seconds')

    VctAnnotations=list(zip(annotation.sample,annotation.symbol)) ## zip coordinates + annotations (N),(t) etc)
    #print(VctAnnotations)
    for i in range(len(VctAnnotations)):
        if( VctAnnotations[i][1] == 'N' or 
            VctAnnotations[i][1] == 'L' or  
            VctAnnotations[i][1] == 'R' or 
           VctAnnotations[i][1] == 'B' or 
           VctAnnotations[i][1] == 'A' or
           VctAnnotations[i][1] == 'a' or 
           VctAnnotations[i][1] == 'J' or 
           VctAnnotations[i][1] == 'S' or 
           VctAnnotations[i][1] == 'V' or  
           VctAnnotations[i][1] == 'r' or  
           VctAnnotations[i][1] == 'F' or 
           VctAnnotations[i][1] == 'e' or 
           VctAnnotations[i][1] == 'j' or
           VctAnnotations[i][1] == 'n' or 
           VctAnnotations[i][1] == 'E' or 
           VctAnnotations[i][1] == '/' or 
           VctAnnotations[i][1] == 'f' or
           VctAnnotations[i][1] == 'Q' or  
           VctAnnotations[i][1] == '?'):
            VctAnnotationHot[0][VctAnnotations[i][0]] = 1;  
            VctAnnotationHot[1][VctAnnotations[i][0]] = 0;  
    VctAnnotationHot=np.transpose(VctAnnotationHot)
    Vctrecord=np.transpose(Vctrecord) # transpose to (timesteps,feat)

    os.chdir(cwd)
    return Vctrecord, VctAnnotationHot


class Record:
    def __init__(self):
        self.file_name = "C:\\Users\\Dell\\Desktop\\Technion\\DeepLearning\\project_data\\mit-bih\\files\\04043"
        self.record = wfdb.rdrecord(self.file_name)
        self.ann = wfdb.rdann(self.file_name,extension="atr")


    def visualization(self):
        # wfdb.plot_wfdb(record=self.record, title='test1')
        # display(self.record.__dict__)
        wfdb.plot_all_records("C:\\Users\\Dell\\Desktop\\Technion\\DeepLearning\\project_data\\mit-bih\\files")


if __name__ == '__main__':
    # r = Record()
    # r.visualization()

    # get_ecg_data("C:\\Users\\Dell\\Desktop\\Technion\\DeepLearning\\project_data\\mit-bih\\files\\04043.dat")

    get_ecg_data("C:\\Users\\Dell\\Desktop\\Technion\\DeepLearning\\project_data\\mit-bih\\files\\04043")

    aaa=2