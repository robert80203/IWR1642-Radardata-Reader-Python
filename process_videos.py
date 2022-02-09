import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

def GenerateFramesfromVideo(video_path, save_path, isFliped=False):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if isFliped:
            image = cv2.flip(image, 0)
        cv2.imwrite(save_path+"/%09d.jpg" % count, image)     # save frame as JPEG file      
        success, image = vidcap.read()
        print('%s, save a new frame: %d' % (video_path, count), end="\r")
        count += 1
    return count

def PreprocessFrames(num_frames, load_path, save_path):
    transform_fn = transforms.Compose([transforms.ToTensor()])
    toimg = transforms.ToPILImage()
    for i in range(num_frames):
        img = Image.open(load_path+"/%09d.jpg" % i).convert('RGB')
        img = transform_fn(img)
        #print(img.size())
        img = F.interpolate(img.unsqueeze(0), size=[512, 512]) # 256, 256
        img = toimg(img.squeeze(0))
        img.save(save_path+"/%09d.jpg" % i)
        print('Save a new frame: %d' % i)
        #img.show()

def PreprocessSpecificFrames(load_path, save_path, p_frame_num):
    #transform_fn = transforms.Compose([transforms.CenterCrop(512), transforms.ToTensor()])
    transform_fn = transforms.Compose([transforms.ToTensor()])
    toimg = transforms.ToPILImage()
    i = 0
    for idx in range(p_frame_num, p_frame_num+1800):
        img = Image.open(load_path+"/%09d.jpg" % idx).convert('RGB')
        img = transform_fn(img)
        #img = toimg(img)
        #print(img.size())
        img = F.interpolate(img.unsqueeze(0), size=[512, 512]) # 256, 256
        img = toimg(img.squeeze(0))
        img.save(save_path+"/%09d.jpg" % i)
        print('%s, save a old frame %d as a new frame %d' % (save_path, idx, i), end='\r')
        i += 1

def GenerateFramesfromVideoProcessFrames(video_path, save_path, p_frame_num, isFliped=False):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    idx = 0
    while success:
        if count >= p_frame_num and count < (p_frame_num+1800):
            if isFliped:
                image = cv2.flip(image, 0)
                #image = cv2.flip(image, -1)
            image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_LINEAR)
            cv2.imwrite(save_path+"/%09d.jpg" % idx, image)     # save frame as JPEG file 
            print('%s, save a new frame: %d' % (video_path, count), end="\r")
            idx += 1
        success, image = vidcap.read()
        count += 1
    return count

def GenerateVideofromFrames(frame_path, save_path):
    #img = cv2.imread(frame_path+'/RAmap0.png')
    #print(frame_path)
    img = cv2.imread(frame_path + ('/%09d.png' % 0))
    fps = 30
    size = (img.shape[1],img.shape[0])
    #print(size)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    videoWrite = cv2.VideoWriter(save_path, fourcc,fps,size)

    
    for i in range(0,1800):
        #fileName = frame_path+'/RAmap%d.png'%i
        fileName = frame_path+'/%09d.png' % i
        img = cv2.imread(fileName)
        videoWrite.write(img)
        print(i, end='\r')




#for videos collected after 20210628
#the video clips should be flipped (isFliped=True)

if __name__ == "__main__":
    
    nameGroup = ['single_%d' % i for i in range(1, 1+1)]
    #nameGroup = ['single_%d' % i for i in range(1, 15+1)]
    root = '20211220'
    
    # for idxName in range(len(nameGroup)):
    #    name = nameGroup[idxName]
    #    pathVideo = "videos/" + root + "/" + name + ".mp4" # ".avi"
    #    pathRawFrame = "frames/" + root + "/" + name + "/raw"
    #    numFrames = GenerateFramesfromVideo(pathVideo, pathRawFrame, True)
    #    #PreprocessFrames(numFrames, pathRawFrame, pathProcessedFrame)

    # startFrameGroup = [
    #     166, 
    #     91, 
    #     123, 
    #     102, 
    #     103, 
    #     84, 
    #     80, 
    #     96, 
    #     83, 
    #     76, 
    #     87, 
    #     83, 
    #     76, 
    #     74, 
    #     83
    # ]

    #0916
    # startFrameGroup = [
    #     101, 
    #     101, 
    #     89, 
    #     111, 
    #     112, 
    #     248, 
    #     121, 
    #     136, 
    #     79, 
    #     84, 
    #     91, 
    #     80, 
    #     123, 
    #     99, 
    #     87
    # ]
    # ''' 20211106_long_range '''
    # startFrameGroup = [
    #     68, 
    #     99, 
    #     78, 
    #     81, 
    #     79, 
    #     69, 
    #     60, 
    #     66, 
    #     100, 
    #     54, 
    #     51, 
    #     68, 
    #     72, 
    #     50, 
    #     52, 
    #     63, 
    #     47, 
    #     59, 
    #     60, 
    #     62, 
    #     89, 
    #     79, 
    #     122, 
    #     82, 
    #     160, 
    #     72, 
    #     89, 
    #     92, 
    #     65, 
    #     78, 
    #     71, 
    #     70, 
    #     76, 
    #     96, 
    #     78, 
    #     88, 
    #     94, 
    #     56, 
    #     71, 
    #     89, 
    #     51, 
    #     54, 
    #     46, 
    #     66, 
    #     55, 
    #     70, 
    #     50, 
    #     68, 
    #     54, 
    #     48, 
    #     61, 
    #     64, 
    #     55, 
    #     57, 
    #     55, 
    #     62, 
    #     66, 
    #     51, 
    #     66, 
    #     56, 
    #     55, 
    #     54, 
    #     65, 
    #     56, 
    #     60, 
    #     54, 
    #     59, 
    #     74, 
    #     56, 
    #     56, 
    #     61, 
    #     56, 
    #     60, 
    #     64, 
    #     71, 
    #     61, 
    #     56, 
    #     53, 
    #     46, 
    #     57, 
    #     64, 
    #     76, 
    #     116, 
    #     62, 
    #     45, 
    #     89, 
    #     88, 
    #     61, 
    #     73, 
    #     43, 
    #     61, 
    #     78, 
    #     67, 
    #     54, 
    #     34, 
    #     46, 
    #     77, 
    #     80, 
    #     35, 
    #     85, 
    #     58, 
    #     61, 
    #     55, 
    #     64, 
    #     53, 
    #     55, 
    #     56, 
    #     56, 
    #     61, 
    #     60, 
    #     80, 
    #     56, 
    #     65, 
    #     62, 
    #     57, 
    #     54, 
    #     49, 
    #     61, 
    #     61, 
    #     53
    # ]

    # for idxName in range(len(nameGroup)):
    #    name = nameGroup[idxName]
    #    pathRawFrame = "frames/" + root + "/" + name + "/raw"
    #    pathProcessedFrame = "frames/" + root + "/" + name + "/processed/images"
    #    PreprocessSpecificFrames(pathRawFrame, pathProcessedFrame, startFrameGroup[idxName])
    
    # 20211220_ShortRangeConfig
    startFrameGroup = [
        89, 114, 101, 94, 80, 
        94, 81, 94, 86, 84, 
        52, 56, 75, 92, 70, 
        55, 62, 69, 86, 63, 
        52, 53, 66, 52, 67, 
        64, 99, 61, 67, 64
    ]
    # 20220107_LongRangeConfig
    # startFrameGroup = [
	# 	121, 96, 70, 135, 76, 
	# 	81, 95, 136, 71, 86, 
	# 	95, 156, 162, 118, 117, 
	# 	117, 140, 108, 98, 100, 
	# 	83, 130, 129, 120, 88, 
	# 	108, 124, 96, 168, 96
    # ]

    #for specific
    # nameGroup = ['single_1', 'single_7', 'single_13', 'single_19', 'single_25']
    # nameIdx = [1, 7, 13, 19, 25]
    # for idxName in range(len(nameIdx)):
    #     name = nameGroup[idxName]
    #     pathVideo = "videos/" + root + "/" + name + ".mp4" # ".avi"
    #     pathProcessedFrame = "frames/" + root + "/" + name + "/processed/images"
    #     GenerateFramesfromVideoProcessFrames(pathVideo, pathProcessedFrame, startFrameGroup[nameIdx[idxName]-1], True)
    
    #for all
    # for idxName in range(len(nameGroup)):
    #     name = nameGroup[idxName]
    #     pathVideo = "videos/" + root + "/" + name + ".mp4" # ".avi"
    #     pathProcessedFrame = "frames/" + root + "/" + name + "/processed/images"
    #     GenerateFramesfromVideoProcessFrames(pathVideo, pathProcessedFrame, startFrameGroup[idxName], True)


    #pathVideoFrame = "/work/robert80203/Radar_detection/radar_skeleton_estimation/data/20211220_Pad_CltRm_RDA/single_6/visualization"
    pathVideoFrame = "radar_skeleton_estimation/visualization/20220107/eccv2022/stunet_df/single_19"
    #pathVideo = "./videos/20211220/gt/single_25.avi"
    pathVideo = "./videos/20220107/eccv2022/stunet_df_single_19.avi"
    GenerateVideofromFrames(pathVideoFrame, pathVideo)
