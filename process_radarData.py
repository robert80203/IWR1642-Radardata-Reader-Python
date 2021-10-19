import numpy as np
from scipy.fftpack import fft,ifft
from cmath import sqrt
import scipy.io
from plot_utils import PlotMaps, PlotPose, PlotHeatmaps
from PIL import Image
import json
#import mmwave as mm
#from mmwave.dataloader import DCA1000

class RadarObject():
    def __init__(self, visualization=True, saveWithChirp=False):
        numGroup = 1#30 + 1
        self.root = '20211008'#'20210712'
        self.saveRoot = '20211008_Pad_CltRm'#'20210712_Pad_CltRm_RDA'
        self.radarDataFileNameGroup = []
        self.saveDirNameGroup =  []
        self.rgbFileNameGroup = []
        self.jointsFileNameGroup = []
        self.numADCSamples = 256 # 0916: 128, 1008: 256
        self.numAngleBins = self.numADCSamples//2 # half of range/numADCsamples
        self.numADCBits = 1
        self.numRX = 4
        self.numLanes = 2
        self.framePerSecond = 30
        self.duration = 60
        self.numFrame = self.framePerSecond * self.duration
        self.numChirp = 128 #0916: 256, 1008: 128, 2 times of the num of chirp you set in TI tool
        self.idxProcChirp = 64 #0916: 128, 1008: 64 # manually decided how many chirps you want to process
        self.numGroupChirp = 4 #0916: 8, 1008: 8 # manually decided the chirp duration, e.g.  64/4 = totally has 16 chirp
        self.numKeypoints = 16#17
        self.xIndices = [-45, -30, -15, 0, 15, 30, 45]
        self.yIndices = [i * 10 for i in range(10)]
        self.visualization = visualization
        self.saveWithChirp = saveWithChirp
        self.showPose = False
        self.showHeatmap = False

        #self.dca = DCA1000()
        self.initialize(numGroup)

    def initialize(self, numGroup):
        for i in range(1, numGroup + 1):
        #for i in range(5, 6):
            radarDataFileName = ['raw_data/iwr1642/' + self.root + '/single_' + str(i) + '/hori', 
                                 'raw_data/iwr1642/' + self.root + '/single_' + str(i) + '/vert']
            saveDirName = 'processed_data/iwr1642/' + self.saveRoot + '/single_' + str(i)
            #saveDirName = 'radar_skeleton_estimation/data/' + self.saveRoot + '/single_' + str(i)
            rgbFileName = 'frames/' + self.root + '/single_' + str(i) + '/processed/images'
            jointsFileName = 'processed_data/iwr1642/' + self.saveRoot + '/single_' + str(i) + '/annot/hrnet_annot.json'
            #jointsFileName = 'radar_skeleton_estimation/data/' + self.saveRoot + '/single_' + str(i) + '/annot/hrnet_annot.json'
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
        if self.saveWithChirp:
            dataFFTGroup = np.zeros((self.idxProcChirp//self.numGroupChirp, self.numADCSamples//2, self.numAngleBins), dtype='complex_')
            #dataFFTGroup = np.zeros((self.idxProcChirp//self.numGroupChirp, self.numADCSamples, self.numAngleBins), dtype='complex_')
            #dataFFTGroup = np.zeros((self.idxProcChirp, self.numADCSamples//2, self.numAngleBins), dtype='complex_')
            i = 0
            for idxChirp in range(0, self.idxProcChirp, self.numGroupChirp):
                #for idxChirp in range(32, 64+32, 8):
                dataFFT = dataRangeAzimuth[:,idxChirp,0:self.numADCSamples//2]
                dataFFTGroup[i, :, :] = self.postProcessFFT(dataFFT)
                i += 1
            return dataFFTGroup
        else:
            dataFFT = dataRangeAzimuth[:,0,0:self.numADCSamples//2]
            #dataFFT = dataRangeAzimuth[:,0,0:self.numADCSamples]
            dataFFT = self.postProcessFFT(dataFFT)
            return dataFFT

    def generateRangeAzimuthMapV2(self, frame):
        dataRangeAzimuth = np.zeros((self.numRX*2, self.numChirp//2, self.numADCSamples), dtype='complex_')
        # for idxRX in range(self.numRX):
        #     for idxChirp in range(self.numChirp//2):
        #         dataRangeAzimuth[idxRX, idxChirp] = (frame[idxRX, idxChirp*2] + frame[idxRX, idxChirp*2+1])/2
        #         dataRangeAzimuth[idxRX+4, idxChirp] = (frame[idxRX, idxChirp*2] - frame[idxRX, idxChirp*2+1])/2
        
        # Process radar data with TDM-MIMO
        for idxRX in range(self.numRX):
            for idxChirp in range(self.numChirp):
                if idxChirp % 2 == 0:
                    dataRangeAzimuth[idxRX, idxChirp//2] = frame[idxRX, idxChirp]
                else:
                    dataRangeAzimuth[idxRX+4, idxChirp//2] = frame[idxRX, idxChirp]

        # perform static clutter removal, along with chirps
        #dataRangeAzimuth = np.transpose(dataRangeAzimuth, (1, 0, 2))
        #dataRangeAzimuth = self.clutterRemoval(dataRangeAzimuth, axis=0)
        #dataRangeAzimuth = np.transpose(dataRangeAzimuth, (1, 0, 2))

        # Range FFT
        for idxRX in range(self.numRX*2):
            for idxChirp in range(self.numChirp//2):
                dataRangeAzimuth[idxRX, idxChirp, :] = np.fft.fft(dataRangeAzimuth[idxRX, idxChirp, :])
        # Doppler FFT
        for idxRX in range(self.numRX*2):
            for idxADC in range(self.numADCSamples):
                dataRangeAzimuth[idxRX, :, idxADC] = np.fft.fft(dataRangeAzimuth[idxRX, :, idxADC])
        padding = ((0, self.numAngleBins - dataRangeAzimuth.shape[0]), (0,0), (0,0))
        dataRangeAzimuth = np.pad(dataRangeAzimuth, padding, mode='constant')
        # Azimuth FFT
        for idxChirp in range(self.numChirp//2):
            for idxADC in range(self.numADCSamples):
                dataRangeAzimuth[:, idxChirp, idxADC] = np.fft.fft(dataRangeAzimuth[:, idxChirp, idxADC])
        #dataFFT = dataRangeAzimuth[:,:,0:self.numADCSamples].max(axis=1)
        if self.saveWithChirp:
            pass
            # dataFFTGroup = np.zeros((self.idxProcChirp, self.numADCSamples//2, self.numAngleBins), dtype='complex_')
            # for idxChirp in range(self.idxProcChirp):
            #     dataFFT = dataRangeAzimuth[:,idxChirp,0:self.numADCSamples//2]
            #     dataFFTGroup[idxChirp, :, :] = self.postProcessFFT(dataFFT)
            # return dataFFTGroup
        else:
            #dataFFT = dataRangeAzimuth[:,0,0:self.numADCSamples//2]
            dataFFT = np.mean(dataRangeAzimuth[:, :, 0:self.numADCSamples//2], 1)
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

    def saveRadarData(self, matrix, dirName, idxFrame):
        dirSave = dirName + ('/%09d' % idxFrame) + '.npy'
        np.save(dirSave, matrix)
        
        #if self.saveWithChirp:
        #    for idxChirp in range(self.idxProcChirp):
        #        idxChirpFrame = (idxFrame * (self.idxProcChirp)) + idxChirp
        #        dirSave = dirName + ('/%09d' % idxChirpFrame) + '.txt'
        #        np.savetxt(dirSave, np.column_stack([matrix[idxChirp].real, matrix[idxChirp].imag]))
        #else:
        #    dirSave = dirName + ('/%09d' % idxFrame) + '.txt'
        #    np.savetxt(dirSave, np.column_stack([matrix.real, matrix.imag]))

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
                mapRangeAzimuthHori = self.generateRangeAzimuthMap(frameHori)
                mapRangeAzimuthVert = self.generateRangeAzimuthMap(frameVert)
                if self.visualization:
                    if self.saveWithChirp:
                        mapRangeAzimuthHori = mapRangeAzimuthHori[0,:,:]#only take the first chirp
                        mapRangeAzimuthVert = mapRangeAzimuthVert[0,:,:]
                    visDirName = self.saveDirNameGroup[idxName] + '/visualization' + ('/%09d.png' % idxFrame)
                    
                    if self.showPose or self.showHeatmap:
                        img = np.array(Image.open(self.rgbFileNameGroup[idxName] + "/%09d.jpg" % idxFrame).convert('RGB'))
                        joints = annotGroup[idxFrame]['joints']
                    else:
                        img = None
                        joints = None
                    self.saveDataAsFigure(img, joints, 
                        mapRangeAzimuthHori, mapRangeAzimuthVert, 
                        visDirName, idxFrame)
                else:
                    horiDirName = self.saveDirNameGroup[idxName] + '/hori' #+ ('/%09d' % idxFrame) + '.txt'
                    self.saveRadarData(mapRangeAzimuthHori, horiDirName, idxFrame)
                    vertDirName = self.saveDirNameGroup[idxName] + '/verti' #+ ('/%09d' % idxFrame) + '.txt'
                    self.saveRadarData(mapRangeAzimuthVert, vertDirName, idxFrame)

                print('%s, finished frame %d' % (self.radarDataFileNameGroup[idxName][0], idxFrame), end='\r')
if __name__ == "__main__":
    radarObject = RadarObject(visualization=False, saveWithChirp=True)
    radarObject.processRadarData()