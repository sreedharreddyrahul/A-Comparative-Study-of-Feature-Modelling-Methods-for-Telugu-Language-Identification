import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import os
fe=[]
for i in range(0,7):
  path="C:/Users/sreed/VSCODE_CoDING/AISP/Languages/"
  path="C:/Users/sreed/VSCODE_CoDING/AISP/Languages/"+os.listdir(path)[i]
  print('* '+path+': *')
  for i1 in range(0,6):
      print()
      print('*'+path+"/"+os.listdir(path)[i1]+'*')
      frequency_sampling, audio_signal = wavfile.read(path+"/"+os.listdir(path)[i1])

      audio_signal = audio_signal[:15000]

      features_mfcc = mfcc(audio_signal, frequency_sampling)
      fe.append(features_mfcc)
      
          # f.write(features_mfcc)

      print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
      print('Length of each feature =', features_mfcc.shape[1])
      # print(features_mfcc)


      features_mfcc = features_mfcc.T
      plt.matshow(features_mfcc)
      plt.title('MFCC')

with open('C:/Users/sreed/VSCODE_CoDING/AISP/Languages/text.txt','wb') as f:
  np.savetxt(f, fe, fmt='%.2f')