import numpy as np
from scipy.fftpack import fft,ifft
from cmath import sqrt
import scipy.io
from plot_utils import PlotMaps, PlotPose, PlotHeatmaps
from PIL import Image
import json
import torch
import torch.nn.functional as F
#import mmwave as mm
#from mmwave.dataloader import DCA1000

class RadarObject():
    def __init__(self, visualization=False):
        numGroup = 30 + 1
        self.root = '20211220' #'20210712'
        self.saveRoot = '20211220_Pad_CltRm_RDA' #'20210712_Pad_CltRm_RDA'
        self.radarDataFileNameGroup = []
        self.saveDirNameGroup =  []
        self.rgbFileNameGroup = []
        self.jointsFileNameGroup = []
        self.numADCSamples = 256 # 0916: 128, 1008: 256, 1106: 128, 1220: 256, 0107: 128
        self.adcRatio = 4
        self.numAngleBins = self.numADCSamples//self.adcRatio # half of range/numADCsamples
        self.numADCBits = 1
        self.numRX = 4
        self.numLanes = 2
        self.framePerSecond = 30
        self.duration = 60
        self.numFrame = self.framePerSecond * self.duration
        self.numChirp = 128 #0916: 256, 1008: 128, 1106: 256, 1220: 128, 0107: 256 # 2 times of the num of chirp you set in TI tool
        self.idxProcChirp = 64 #0916: 128, 1008: 64, 1106: 128, 1220: 64, 0107: 128 # manually decided how many chirps you want to process
        self.numGroupChirp = 4 #1106:8 1220:8, 0107:8 # manually decided the chirp duration, e.g.  64/4 = totally has 16 chirp
        self.numKeypoints = 14 
        self.xIndices = [-45, -30, -15, 0, 15, 30, 45]
        self.yIndices = [i * 10 for i in range(10)]
        self.visualization = visualization
        #self.saveWithChirp = saveWithChirp
        self.showPose = False #always set it to false
        self.showHeatmap = True
        self.useVelocity = True
        #self.downscaleRange = True

        #leftVIndex = {i - self.idxProcChirp//2: -i for i in range(self.idxProcChirp//2)}
        leftVIndex = {i - self.idxProcChirp//2: -(i+1) for i in range(self.idxProcChirp//2)}
        rightVIndex = {i:self.idxProcChirp//2 - i for i in range(self.idxProcChirp//2)}
        self.mergeDict = {**leftVIndex, **rightVIndex}
        #self.dca = DCA1000()
        self.initialize(numGroup)

    def initialize(self, numGroup):
        #for i in range(1, numGroup + 1):
        for i in range(16, 31):
            radarDataFileName = ['raw_data/iwr1642/' + self.root + '/single_' + str(i) + '/hori', 
                                 'raw_data/iwr1642/' + self.root + '/single_' + str(i) + '/vert']
            #saveDirName = 'processed_data/iwr1642/' + self.saveRoot + '/single_' + str(i)
            saveDirName = 'radar_skeleton_estimation/data/' + self.saveRoot + '/single_' + str(i)
            rgbFileName = 'frames/' + self.root + '/single_' + str(i) + '/processed/images'
            #jointsFileName = 'processed_data/iwr1642/' + self.saveRoot + '/single_' + str(i) + '/annot/hrnet_annot.json'
            jointsFileName = 'radar_skeleton_estimation/data/' + self.saveRoot + '/single_' + str(i) + '/annot/hrnet_annot.json'
            self.radarDataFileNameGroup.append(radarDataFileName)
            self.saveDirNameGroup.append(saveDirName)
            self.rgbFileNameGroup.append(rgbFileName)
            self.jointsFileNameGroup.append(jointsFileName)

    def getadcDataFromDCA1000(self, fileName):
        #with open(fileName, 'rb') as fp:
            #adcData = np.fromfile(fp, np.int16)
            #adcData = np.array(adcData, dtype="uint16")
        adcData = np.fromfile(fileName, dtype=np.int16)

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
    
    def getadcDataFromDCA1000V2(self, fileName):
        #with open(fileName, 'rb') as fp:
            #adcData = np.fromfile(fp, np.int16)
            #adcData = np.array(adcData, dtype="uint16")
        adcData = np.fromfile(fileName, dtype=np.int16)
        fileSize = adcData.shape[0]
        #adcData = adcData.reshape(-1, self.numLanes*2).transpose()

        # # for complex data
        fileSize = int(fileSize/2)
        LVDS = np.zeros((1, fileSize), dtype = 'complex_')  # seperate each LVDS lane into rows

        counter = 0
        for i in range(0, fileSize, 4):
            #print(LVDS[0, counter], adcData[i])
            LVDS[0, counter] = adcData[i] + np.sqrt(-1+0j) * adcData[i+2]
            LVDS[0, counter+1] = adcData[i+1] + np.sqrt(-1+0j) * adcData[i+3]
            counter = counter + 2

        #LVDS = LVDS.reshape(self.numADCSamples * self.numRX, self.numChirp)
        LVDS = LVDS.reshape(self.numADCSamples * self.numRX, self.numChirp * self.numFrame)
        LVDS = LVDS.transpose()

        adcData = np.zeros((self.numRX, self.numChirp * self.numFrame * self.numADCSamples), dtype = 'complex_')

        for row in range(self.numRX):
            for i in range(self.numChirp * self.numFrame):
                adcData[row, i * self.numADCSamples:(i + 1) * self.numADCSamples] = \
                LVDS[i, row * self.numADCSamples:(row + 1) * self.numADCSamples]
        
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
    
    def clutterRemoval(self, input_val, axis=0):
        """Perform basic static clutter removal by removing the mean from the input_val on the specified doppler axis.
        Args:
            input_val (ndarray): Array to perform static clutter removal on. Usually applied before performing doppler FFT.
                e.g. [num_chirps, num_vx_antennas, num_samples], it is applied along the first axis.
            axis (int): Axis to calculate mean of pre-doppler.
        Returns:
            ndarray: Array with static clutter removed.
        """
        # Reorder the axes
        reordering = np.arange(len(input_val.shape))
        reordering[0] = axis
        reordering[axis] = 0
        input_val = input_val.transpose(reordering)

        # Apply static clutter removal
        mean = input_val.transpose(reordering).mean(0)
        #output_val = input_val - mean
        output_val = input_val - np.expand_dims(mean, axis=0)
        out = output_val.transpose(reordering)
        return out 
    
    def postProcessFFT(self, dataFFT, rescale=False):
        dataFFT = np.fft.fftshift(dataFFT, axes=(0,))
        if self.visualization:
            dataFFT = np.abs(dataFFT)
        if rescale:
            dataFFT = 10 * np.log10(dataFFT)
        dataFFT = np.transpose(dataFFT)
        dataFFT = np.flip(dataFFT, axis=0)
        return dataFFT

    def generateRangeAzimuthMap(self, frame):
        dataRangeAzimuth = np.zeros((self.numRX*2, self.numChirp//2, self.numADCSamples), dtype='complex_')
        # Process radar data with BPM-MIMO
        # for idxRX in range(self.numRX):
        #     for idxChirp in range(self.numChirp//2):
        #         dataRangeAzimuth[idxRX, idxChirp] = (frame[idxRX, idxChirp*2] + frame[idxRX, idxChirp*2+1])/2
        #         dataRangeAzimuth[idxRX+4, idxChirp] = (frame[idxRX, idxChirp*2] - frame[idxRX, idxChirp*2+1])/2
        
        # Process radar data with TDM-MIMO
        for idxRX in range(self.numRX):
            for idxChirp in range(self.numChirp):
                if idxChirp % 2 == 0:
                    #dataRangeAzimuth[idxRX*2, idxChirp//2] = frame[idxRX, idxChirp]
                    dataRangeAzimuth[idxRX, idxChirp//2] = frame[idxRX, idxChirp]
                else:
                    #dataRangeAzimuth[idxRX*2+1, idxChirp//2] = frame[idxRX, idxChirp]
                    dataRangeAzimuth[idxRX+4, idxChirp//2] = frame[idxRX, idxChirp]

        # perform static clutter removal, along with chirps
        dataRangeAzimuth = np.transpose(dataRangeAzimuth, (1, 0, 2))
        dataRangeAzimuth = self.clutterRemoval(dataRangeAzimuth, axis=0)
        dataRangeAzimuth = np.transpose(dataRangeAzimuth, (1, 0, 2))

        padding = ((0, self.numAngleBins - dataRangeAzimuth.shape[0]), (0,0), (0,0))
        #padding = ((0, 512 - 8), (0,0), (0,512 - 128))
        dataRangeAzimuth = np.pad(dataRangeAzimuth, padding, mode='constant')

        for idxChirp in range(self.numChirp//2):
            dataRangeAzimuth[:, idxChirp, :] = np.fft.fft2(dataRangeAzimuth[:, idxChirp, :])
        #dataFFT = dataRangeAzimuth[:,:,0:self.numADCSamples].max(axis=1)
        #if self.saveWithChirp:
        if not self.visualization:
            dataFFTGroup = np.zeros((self.idxProcChirp//self.numGroupChirp, self.numADCSamples//self.adcRatio, self.numAngleBins), dtype='complex_')
            #dataFFTGroup = np.zeros((self.idxProcChirp//self.numGroupChirp, self.numADCSamples, self.numAngleBins), dtype='complex_')
            #dataFFTGroup = np.zeros((self.idxProcChirp, self.numADCSamples//2, self.numAngleBins), dtype='complex_')
            i = 0
            for idxChirp in range(0, self.idxProcChirp, self.numGroupChirp):
                #for idxChirp in range(32, 64+32, 8):
                dataFFT = dataRangeAzimuth[:,idxChirp,0:self.numADCSamples//self.adcRatio]
                dataFFTGroup[i, :, :] = self.postProcessFFT(dataFFT)
                i += 1
            return dataFFTGroup
        else:
            dataFFT = dataRangeAzimuth[:,0,0:self.numADCSamples//self.adcRatio]
            #dataFFT = dataRangeAzimuth[:,0,0:self.numADCSamples]
            dataFFT = self.postProcessFFT(dataFFT)
            return dataFFT

    def generateRangeDopplerAzimuthMap(self, frame):
        dataRangeAzimuth = np.zeros((self.numRX*2, self.numChirp//2, self.numADCSamples), dtype='complex_')

        # Process radar data with TDM-MIMO
        for idxRX in range(self.numRX):
            for idxChirp in range(self.numChirp):
                if idxChirp % 2 == 0:
                    dataRangeAzimuth[idxRX, idxChirp//2] = frame[idxRX, idxChirp]
                else:
                    dataRangeAzimuth[idxRX+4, idxChirp//2] = frame[idxRX, idxChirp]

        # perform static clutter removal, along with chirps
        dataRangeAzimuth = np.transpose(dataRangeAzimuth, (1, 0, 2))
        dataRangeAzimuth = self.clutterRemoval(dataRangeAzimuth, axis=0)
        dataRangeAzimuth = np.transpose(dataRangeAzimuth, (1, 0, 2))

        for idxRX in range(self.numRX * 2):
            dataRangeAzimuth[idxRX, :, :] = np.fft.fft2(dataRangeAzimuth[idxRX, :, :])

        padding = ((0, self.numAngleBins - dataRangeAzimuth.shape[0]), (0,0), (0,0))
        #padding = ((0, 512 - 8), (0,0), (0,512 - 128))
        dataRangeAzimuth = np.pad(dataRangeAzimuth, padding, mode='constant')

        for idxChirp in range(self.numChirp//2):
            for idxADC in range(self.numADCSamples):
                dataRangeAzimuth[:, idxChirp, idxADC] = np.fft.fft(dataRangeAzimuth[:, idxChirp, idxADC])
        
        #specific range
        idxADCSpecific = [i for i in range(76, 140)]
        
        if not self.visualization:
            # change velocity index
            #-----------------------------
            #| q | 0 |  |  |  |  | 0 | k |
            #-----------------------------
            #             ||
            #             \/
            #-----------------------------
            #|  |  | 0 | q | k | 0 |  |  |
            #-----------------------------
            dataTemp = np.zeros((self.idxProcChirp, self.numADCSamples//self.adcRatio, self.numAngleBins), dtype='complex_')
            dataFFTGroup = np.zeros((self.idxProcChirp//self.numGroupChirp, self.numADCSamples//self.adcRatio, self.numAngleBins), dtype='complex_')
            for idxRX in range(self.numAngleBins):
                for idxADC in range(self.numADCSamples//self.adcRatio):
                    #dataTemp[:, idxADC, idxRX] = dataRangeAzimuth[idxRX, :, idxADC]
                    #dataTemp[:, idxADC, idxRX] = np.fft.fftshift(dataTemp[:, idxADC, idxRX], axes=(0))
                    dataTemp[:, idxADC, idxRX] = dataRangeAzimuth[idxRX, :, idxADCSpecific[idxADC]]
                    dataTemp[:, idxADC, idxRX] = np.fft.fftshift(dataTemp[:, idxADC, idxRX], axes=(0))
            chirpPad = self.idxProcChirp//self.numGroupChirp
            i = 0
            for idxChirp in range(self.idxProcChirp//2 - chirpPad//2, self.idxProcChirp//2 + chirpPad//2):
                dataFFTGroup[i, :, :] = self.postProcessFFT(np.transpose(dataTemp[idxChirp, :, :]))
                i += 1
            return dataFFTGroup
        else:       
            dataTemp = np.zeros((self.idxProcChirp, self.numADCSamples//self.adcRatio, self.numAngleBins), dtype='complex_')
            for idxRX in range(self.numAngleBins):
                for idxADC in range(self.numADCSamples//self.adcRatio):
                    #dataTemp[:, idxADC, idxRX] = dataRangeAzimuth[idxRX, :, idxADC]
                    #dataTemp[:, idxADC, idxRX] = np.fft.fftshift(dataTemp[:, idxADC, idxRX], axes=(0))
                    dataTemp[:, idxADC, idxRX] = dataRangeAzimuth[idxRX, :, idxADCSpecific[idxADC]]
                    dataTemp[:, idxADC, idxRX] = np.fft.fftshift(dataTemp[:, idxADC, idxRX], axes=(0))
            dataFFT = self.postProcessFFT(np.transpose(np.sum(dataTemp, axis=0)))
            return dataFFT

    def saveDataAsFigure(self, img, joints, 
        mapRangeAzimuthHori, mapRangeAzimuthVert, 
        visDirName, idxFrame):
        pose = img
        heatmap = None
        if self.showPose:
            pose = PlotPose(img, joints)
        if self.showHeatmap:
            heatmap = PlotHeatmaps(joints, self.numKeypoints)
        PlotMaps(visDirName, self.xIndices, self.yIndices, 
            idxFrame, mapRangeAzimuthHori, mapRangeAzimuthVert, 
            pose, heatmap)

    def saveRadarData(self, matrix, dirName, idxFrame):
        dirSave = dirName + ('/%09d' % idxFrame) + '.npy'
        np.save(dirSave, matrix)

    def processRadarData(self):
        for idxName in range(len(self.radarDataFileNameGroup)):
            adcDataHori = self.getadcDataFromDCA1000(self.radarDataFileNameGroup[idxName][0]+'/adc_data.bin')
            adcDataVert = self.getadcDataFromDCA1000(self.radarDataFileNameGroup[idxName][1]+'/adc_data.bin')
            #adcDataHori = np.fromfile(self.radarDataFileNameGroup[idxName][0]+'/adc_data.bin', dtype=dt)
            #adcDataHori = self.dca.organize(adcDataHori)
            
            #adcDataVert = np.fromfile(self.radarDataFileNameGroup[idxName][1]+'/adc_data.bin', dtype=dt)
            #adcDataVert = self.dca.organize(adcDataVert)

            if self.visualization and (self.showPose or self.showHeatmap):
                with open(self.jointsFileNameGroup[idxName], "r") as fp:
                    annotGroup = json.load(fp)
            
            for idxFrame in range(0,self.numFrame):
                frameHori = adcDataHori[:, self.numChirp*(idxFrame):self.numChirp*(idxFrame+1), 0:self.numADCSamples]
                frameVert = adcDataVert[:, self.numChirp*(idxFrame):self.numChirp*(idxFrame+1), 0:self.numADCSamples]
                
                if self.useVelocity:
                    mapRangeAzimuthHori = self.generateRangeDopplerAzimuthMap(frameHori)
                    mapRangeAzimuthVert = self.generateRangeDopplerAzimuthMap(frameVert)
                else:
                    mapRangeAzimuthHori = self.generateRangeAzimuthMap(frameHori)
                    mapRangeAzimuthVert = self.generateRangeAzimuthMap(frameVert)
                
                if self.visualization:
                    visDirName = self.saveDirNameGroup[idxName] + '/visualization' + ('/%09d.png' % idxFrame)
                    if self.showPose or self.showHeatmap:
                        img = np.array(Image.open(self.rgbFileNameGroup[idxName] + "/%09d.jpg" % idxFrame).convert('RGB'))
                        #img = None
                        joints = annotGroup[idxFrame]['joints']
                    else:
                        img = None
                        joints = None
                    self.saveDataAsFigure(img, joints, 
                        mapRangeAzimuthHori, mapRangeAzimuthVert, 
                        visDirName, idxFrame)
                else:
                    self.saveRadarData(mapRangeAzimuthHori, self.saveDirNameGroup[idxName] + '/hori', idxFrame)
                    self.saveRadarData(mapRangeAzimuthVert, self.saveDirNameGroup[idxName] + '/verti', idxFrame)

                print('%s, finished frame %d' % (self.radarDataFileNameGroup[idxName][0], idxFrame), end='\r')
                print(mapRangeAzimuthHori.shape, end='\r')
if __name__ == "__main__":
    radarObject = RadarObject(visualization=False)
    radarObject.processRadarData()