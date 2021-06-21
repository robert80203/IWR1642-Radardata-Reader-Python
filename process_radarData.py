import numpy as np
from scipy.fftpack import fft,ifft
from cmath import sqrt
import scipy.io
from plot_utils import PlotMaps, PlotPose, PlotHeatmaps
from PIL import Image
import json

class RadarObject():
    def __init__(self, visualization=True):
        self.radarDataFileNameGroup = [
            ['raw_data/iwr1642/20210609/single_1/hori', 'raw_data/iwr1642/20210609/single_1/verti']
        ]
        self.saveDirNameGroup =  [
            'processed_data/iwr1642/20210609/single_1'
        ]
        self.rgbFileNameGroup = [
            'frames/20210609/single_1/processed/images'
        ]
        self.jointsFileNameGroup = [
            'processed_data/iwr1642/20210609/single_1/annot/hrnet_annot.json'
        ]
        self.numADCSamples = 128
        self.numADCBits = 1
        self.numRX = 4
        self.numLanes = 2
        self.framePerSecond = 30
        self.duration = 60
        self.numFrame = self.framePerSecond * self.duration
        self.numChirp = 128
        self.numKeypoints = 17
        self.xIndices = [-45, -30, -15, 0, 15, 30, 45]
        self.yIndices = [i * 10 for i in range(10)]
        self.visualization = visualization
        self.showPose = True
        self.showHeatmap = True
    
    def getadcDataFromDCA1000(self, fileName):
        with open(fileName, 'rb') as fp:
            adcData = np.fromfile(fp, np.int16)
            adcData = np.array(adcData, dtype="uint16")

        fileSize = adcData.shape[0]
        adcData = adcData.reshape(-1, self.numLanes*2).transpose()

        # # for complex data
        fileSize = int(fileSize/2)
        LVDS = np.zeros((2, fileSize))  # seperate each LVDS lane into rows

        temp = np.empty((adcData[0].size + adcData[1].size), dtype=adcData[0].dtype)
        temp[0::2] = adcData[0]
        temp[1::2] = adcData[1]
        LVDS[0] = temp
        temp = np.empty((adcData[2].size + adcData[3].size), dtype=adcData[2].dtype)
        temp[0::2] = adcData[2]
        temp[1::2] = adcData[3]
        LVDS[1] = temp

        # organize data by receiver
        # fileSize / 2 = 102400000
        # numADCSamples * (numRX / 2) = 400
        adcData = np.zeros((self.numRX, int(fileSize/self.numRX)), dtype = 'complex_')
        iter = 0
        for i in range(0, fileSize, self.numADCSamples * 4):
            adcData[0][iter:iter+self.numADCSamples] = LVDS[0][i:i+self.numADCSamples] + np.sqrt(-1+0j)*LVDS[1][i:i+self.numADCSamples]
            adcData[1][iter:iter+self.numADCSamples] = LVDS[0][i+self.numADCSamples:i+self.numADCSamples*2] + np.sqrt(-1+0j)*LVDS[1][i+self.numADCSamples:i+self.numADCSamples*2]
            adcData[2][iter:iter+self.numADCSamples] = LVDS[0][i+self.numADCSamples*2:i+self.numADCSamples*3] + np.sqrt(-1+0j)*LVDS[1][i+self.numADCSamples*2:i+self.numADCSamples*3]
            adcData[3][iter:iter+self.numADCSamples] = LVDS[0][i+self.numADCSamples*3:i+self.numADCSamples*4] + np.sqrt(-1+0j)*LVDS[1][i+self.numADCSamples*3:i+self.numADCSamples*4]
            iter = iter + self.numADCSamples

        #correct reshape
        adcDataReshape = adcData.reshape(self.numRX, -1, self.numADCSamples)
        print('Shape of radar data:', adcDataReshape.shape)
        return adcDataReshape

    def getadcDataFromTSW400(self, fileName):
        with open(fileName, 'rb') as fp:
            adcData = np.fromfile(fp, np.int16)
            adcData = np.array(adcData, dtype="uint16")
            adcData = np.array(adcData - 32768, dtype="int16")
        fileSize = adcData.shape[0]
        adcData = adcData.reshape(-1, self.numLanes*2).transpose()

        # # for complex data
        fileSize = int(fileSize/2)
        LVDS = np.zeros((2, adcData.shape[1]), dtype = 'complex_')  # seperate each LVDS lane into rows
        LVDS[0] = adcData[0] + np.sqrt(-1+0j)*adcData[1]
        LVDS[1] = adcData[2] + np.sqrt(-1+0j)*adcData[3]

        # organize data by receiver
        # fileSize / 2 = 102400000
        # numADCSamples * (numRX / 2) = 400
        adcData = np.zeros((self.numRX, int(fileSize/self.numRX)), dtype = 'complex_')
        iter = 0
        for i in range(0, int(fileSize / 2), self.numADCSamples * 2):
            adcData[0][iter:iter+self.numADCSamples] = LVDS[0][i:i+self.numADCSamples]
            adcData[1][iter:iter+self.numADCSamples] = LVDS[1][i:i+self.numADCSamples]
            adcData[2][iter:iter+self.numADCSamples] = LVDS[0][i+self.numADCSamples:i+self.numADCSamples*2]
            adcData[3][iter:iter+self.numADCSamples] = LVDS[1][i+self.numADCSamples:i+self.numADCSamples*2]
            iter = iter + self.numADCSamples

        
        #correct reshape
        adcDataReshape = adcData.reshape(self.numRX, -1, self.numADCSamples)
        print('Shape of radar data:', adcDataReshape.shape)
        return adcDataReshape

    def postProcessFFT(self, dataFFT, rescale=False):
        dataFFT = np.fft.fftshift(dataFFT, axes=(0,))
        if self.visualization:
            dataFFT = np.abs(dataFFT)
        if rescale:
            dataFFT = 10 * np.log10(dataFFT)
        dataFFT = np.transpose(dataFFT)
        dataFFT = np.flip(dataFFT, axis=0)
        dataFFT = np.flip(dataFFT, axis=1)
        return dataFFT

    def generateRangeAzimuthMap(self, frame):
        dataRangeAzimuth = np.zeros((self.numRX*2, self.numChirp//2, self.numADCSamples), dtype='complex_')
        for idxRX in range(self.numRX):
            for idxChirp in range(self.numChirp//2):
                dataRangeAzimuth[idxRX, idxChirp] = (frame[idxRX, idxChirp*2] + frame[idxRX, idxChirp*2+1])/2
                dataRangeAzimuth[idxRX+4, idxChirp] = (frame[idxRX, idxChirp*2] - frame[idxRX, idxChirp*2+1])/2
        for idxChirp in range(self.numChirp//2):
            dataRangeAzimuth[:, idxChirp, :] = np.fft.fft2(dataRangeAzimuth[:, idxChirp, :])
        #dataFFT = dataRangeAzimuth[:,:,0:self.numADCSamples].max(axis=1)
        dataFFT = dataRangeAzimuth[:,0,0:self.numADCSamples//2]
        dataFFT = self.postProcessFFT(dataFFT)
        return dataFFT

    def generateRangeDopplerMap(self, frame):
        dataRangeDoppler = np.zeros((self.numRX*2, self.numChirp//2, self.numADCSamples), dtype='complex_')
        for idxRX in range(self.numRX):
            for idxChirp in range(self.numChirp//2):
                dataRangeDoppler[idxRX, idxChirp] = (frame[idxRX, idxChirp*2] + frame[idxRX, idxChirp*2+1])/2
                dataRangeDoppler[idxRX+4, idxChirp] = (frame[idxRX, idxChirp*2] - frame[idxRX, idxChirp*2+1])/2
        for idxRX in range(self.numRX*2):
            dataRangeDoppler[idxRX, :, :] = np.fft.fft2(dataRangeDoppler[idxRX, :, :])
        #dataFFT = dataRangeDoppler[:,:,0:self.numADCSamples//2].max(axis=0)
        dataFFT = dataRangeDoppler[0,:,0:self.numADCSamples//2]
        dataFFT = postProcessFFT(dataFFT, rescale=True)
        return dataFFT

    def saveDataAsFigure(self, img, joints, 
        mapRangeAzimuthHori, mapRangeAzimuthVert, 
        visDirName, idxFrame):
        pose = None
        heatmap = None
        if self.showPose:
            pose = PlotPose(img, joints)
        if self.showHeatmap:
            heatmap = PlotHeatmaps(joints, self.numKeypoints)
        PlotMaps(visDirName, self.xIndices, self.yIndices, 
            idxFrame, mapRangeAzimuthHori, mapRangeAzimuthVert, 
            pose, heatmap)

    def saveRadarData(self, matrix, dirName):
        np.savetxt(dirName, np.column_stack([matrix.real, matrix.imag]))

    def processRadarData(self):
        for idxName in range(len(self.radarDataFileNameGroup)):
            adcDataHori = self.getadcDataFromDCA1000(self.radarDataFileNameGroup[idxName][0]+'/adc_data.bin')
            adcDataVert = self.getadcDataFromDCA1000(self.radarDataFileNameGroup[idxName][1]+'/adc_data.bin')
            
            if self.visualization:
                with open(self.jointsFileNameGroup[idxName], "r") as fp:
                    annotGroup = json.load(fp)
            
            for idxFrame in range(0,self.numFrame):
                frameHori = adcDataHori[:, self.numChirp*(idxFrame):self.numChirp*(idxFrame+1), 0:self.numADCSamples]
                frameVert = adcDataVert[:, self.numChirp*(idxFrame):self.numChirp*(idxFrame+1), 0:self.numADCSamples]
                mapRangeAzimuthHori = self.generateRangeAzimuthMap(frameHori)
                mapRangeAzimuthVert = self.generateRangeAzimuthMap(frameVert)
                
                if self.visualization:
                    visDirName = self.saveDirNameGroup[idxName] + '/visualization' + ('/%09d.png' % idxFrame)
                    img = np.array(Image.open(self.rgbFileNameGroup[idxName] + "/%09d.jpg" % idxFrame).convert('RGB'))
                    joints = annotGroup[idxFrame]['joints']
                    self.saveDataAsFigure(img, joints, 
                        mapRangeAzimuthHori, mapRangeAzimuthVert, 
                        visDirName, idxFrame)
                else:
                    horiDirName = self.saveDirNameGroup[idxName] + '/hori' + ('/%09d' % idxFrame) + '.txt'
                    self.saveRadarData(mapRangeAzimuthHori, horiDirName)
                    vertDirName = self.saveDirNameGroup[idxName] + '/verti' + ('/%09d' % idxFrame) + '.txt'
                    self.saveRadarData(mapRangeAzimuthVert, vertDirName)

                print('Finished frame %d' % idxFrame)

if __name__ == "__main__":
    radarObject = RadarObject(visualization=True)
    radarObject.processRadarData()