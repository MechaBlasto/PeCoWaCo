

from scipy.signal import blackman
from scipy.fftpack import fft, ifft, fftshift
from scipy.fftpack import fftfreq
import numpy as np
from scipy.signal import find_peaks
from numpy import mean, sqrt, square

def show_peak(onedimg, frequ, veto_frequ):

    peaks, _ = find_peaks(onedimg)


    above_threshfrequ = []
    maxval = 0
    reqpeak =0
    for x in peaks:
      if(frequ[x] > veto_frequ):
        above_threshfrequ.append(x)
    for i in range(0,len(above_threshfrequ)):
      if onedimg[above_threshfrequ[i]] > maxval:
        maxval = onedimg[above_threshfrequ[i]]
        reqpeak = frequ[above_threshfrequ[i]] 
        
    frqY = reqpeak  
    
    return frqY

def CrossCorrelationStrip(imageA, imageB):
    
    PointsSample = imageA.shape[1] 
    stripA = imageA[:,0]
    stripB = imageB[:,0]
    stripCross = np.conjugate(stripA)* stripB
    Initial = 0
    x = []
    for i in range(stripA.shape[0]):
      x.append(i)
    for i in range(imageA.shape[1]):
        
        stripB = imageB[:,i]
        stripCross = np.conjugate(stripA)* stripB
        PointsSample += stripCross
        
    return PointsSample, x 

def SimpleCrossCorrelation(imageA, imageB):
    
    
   

    stripCross = np.conjugate(imageA)* imageB
    Initial = 0
    x = []
    for i in range(imageA.shape[0]):
        x.append(i)
        PointsSample = np.conjugate(imageA)* imageB
        
        
    return PointsSample, x 

  
def RMSStrip(imageA, cal):
    rmstotal = np.empty(imageA.shape[0])
    PointsSample = imageA.shape[1]
    peri = range(0, int(np.round(imageA.shape[0] * cal)))
    for i in range(imageA.shape[0] - 1):
        stripA = imageA[i,:]
        RMS = sqrt(mean(square(stripA)))
        rmstotal[i] = RMS
        
    return [rmstotal, peri]    

def FFTStrip(imageA):
    ffttotal = np.empty(imageA.shape)
    PointsSample = imageA.shape[1] 
    for i in range(imageA.shape[0]):
        stripA = imageA[i,:]
       
        fftstrip = fftshift(fft(stripA))
        ffttotal[i,:] = np.abs(fftstrip)
    return ffttotal 

def PhaseDiffStrip(imageA):
    diff = np.empty(imageA.shape)
    value = np.empty(imageA.shape)
    for i in range(imageA.shape[0] - 1):
       
        diff[i, :] = imageA[i,:] - imageA[i + 1, :]
    return diff
    
def PhaseStrip(imageA):
    ffttotal = np.empty(imageA.shape)
    PointsSample = imageA.shape[1] 
    for i in range(imageA.shape[0]):
        stripA = imageA[i,:]
        
        fftstrip = (fft(stripA))
        ffttotal[i,:] = np.angle(fftstrip)
    return ffttotal


def sumProjection(image):
    sumPro = 0
    time = range(0, image.shape[1])
    time = np.asarray(time)
    for i in range(image.shape[0]):
        strip = image[i,:]
        sumPro += np.abs(strip) 
 
   
    return [sumPro, time]

def maxProjection(image):
    time = range(0, image.shape[1])
    time = np.asarray(time)
         

    return [np.amax(image, axis = 0), time]
    
#FFT along a strip
def doFilterFFT(image,Time_unit, filter):
   addedfft = 0 
   PointsSample = image.shape[1] 
   for i in range(image.shape[0]):
      if filter == True:   
       w = blackman(PointsSample)
      if filter == False:
       w = 1
      strip = image[i,:]
       
      fftresult = fft(w * strip)
      addedfft += np.abs(fftresult)  
   #addedfft/=image.shape[0]
   
   
   xf = fftfreq(PointsSample, Time_unit)
   
   
   return addedfft[1:int(PointsSample//2)], xf[1:int(PointsSample//2)]



def do2DFFT(image, Space_unit, Time_unit, filter):
    fftresult = fft(image)
    PointsT = image.shape[1]
    PointsY = image.shape[0]
    Tomega = fftfreq(PointsT, Time_unit)
    Spaceomega = fftfreq(PointsY, Space_unit)
    
    return fftresult

def do2DInverseFFT(image, Space_unit, Time_unit, filter):
    fftresult = ifft(image)
    PointsT = image.shape[1]
    PointsY = image.shape[0]
    Tomega = fftfreq(PointsT, Time_unit)
    Spaceomega = fftfreq(PointsY, Space_unit)
    
    return fftresult
def CrossCorrelation(imageA, imageB):
    crosscorrelation = imageA
    fftresultA = fftshift(fft(imageA))
    fftresultB = fftshift(fft(imageB))
    multifft = fftresultA * np.conj(fftresultB)
    crosscorrelation = fftshift(ifft(multifft))
    return np.abs(crosscorrelation) 


def Similarity(image, calibration, umblock, time):
    
   
    block = int(umblock/calibration)
    sliceimage = image[:,time]
    startregionslice = sliceimage[0:block]
    mul = []
    for i in range(0, sliceimage.shape[0], block):
        
        startindex = i
        endindex = block + startindex
        if endindex -startindex == block:
          regionslice = sliceimage[startindex:endindex]
          for j in range(0, regionslice.shape[0]):
              mul.append(startregionslice[j] * regionslice[j]/ (startregionslice[0] * regionslice[0]))
        
     
    return mul    
    
    