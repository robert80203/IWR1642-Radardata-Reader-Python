import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

def GenerateFramesfromVideo(video_path, save_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(save_path+"/%09d.jpg" % count, image)     # save frame as JPEG file      
        success, image = vidcap.read()
        print('Save a new frame: %d' % count)
        count += 1
    return count

def PreprcessFrames(num_frames, load_path, save_path):
    transform_fn = transforms.Compose([transforms.ToTensor()])
    toimg = transforms.ToPILImage()
    for i in range(num_frames):
        img = Image.open(load_path+"/%09d.jpg" % i).convert('RGB')
        img = transform_fn(img)
        #print(img.size())
        img = F.interpolate(img.unsqueeze(0), size=[256, 256])
        img = toimg(img.squeeze(0))
        img.save(save_path+"/%09d.jpg" % i)
        print('Save a new frame: %d' % i)
        #img.show()

def PreprcessSpecificFrames(load_path, save_path, p_frame_num):
    transform_fn = transforms.Compose([transforms.ToTensor()])
    toimg = transforms.ToPILImage()
    i = 0
    for idx in range(p_frame_num, p_frame_num+1800):
        img = Image.open(load_path+"/%09d.jpg" % idx).convert('RGB')
        img = transform_fn(img)
        #print(img.size())
        img = F.interpolate(img.unsqueeze(0), size=[256, 256])
        img = toimg(img.squeeze(0))
        img.save(save_path+"/%09d.jpg" % i)
        print('Save a old frame %d as a new frame %d' % (idx, i))
        i += 1

def GenerateVideofromFrames(frame_path, save_path):
    #img = cv2.imread(frame_path+'/RAmap0.png')
    img = cv2.imread(frame_path + ('/%09d.png' % 0))
    fps = 30
    size = (img.shape[1],img.shape[0])
    print(size)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    videoWrite = cv2.VideoWriter(save_path, fourcc,fps,size)


    for i in range(0,1800):
        #fileName = frame_path+'/RAmap%d.png'%i
        fileName = frame_path+'/%09d.png' % i
        img = cv2.imread(fileName)
        videoWrite.write(img)
        print(i)


if __name__ == "__main__":
    nameGroup = [
    'single_1',
    #'single_2',
    #'single_3',
    #'single_4',
    #'single_5',
    #'single_6',
    ]
    startFrameGroup = [
        #139,
        #75,
        #63,
        147,
        #103,
        #247
    ]
    for idxName in range(len(nameGroup)):
        name = nameGroup[idxName]
        pathVideo = "videos/20210609/" + name + ".avi"
        #pathRawFrame = "frames/20210609/" + name + "/raw"
        #pathProcessedFrame = "frames/20210609/" + name + "/processed/images"
        
        #numFrames = GenerateFramesfromVideo(pathVideo, pathRawFrame)
        #PreprcessFrames(numFrames, pathRawFrame, pathProcessedFrame)
        
        #PreprcessSpecificFrames(pathRawFrame, pathProcessedFrame, startFrameGroup[idxName])
        
        pathVideoFrame = "processed_data/iwr1642/20210609/" + name + "/visualization"
        GenerateVideofromFrames(pathVideoFrame, pathVideo)