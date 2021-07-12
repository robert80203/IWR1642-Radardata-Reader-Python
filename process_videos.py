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
        print('%s, save a new frame: %d' % (video_path, count))
        count += 1
    return count

def PreprcessFrames(num_frames, load_path, save_path):
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

def PreprcessSpecificFrames(load_path, save_path, p_frame_num):
    transform_fn = transforms.Compose([transforms.ToTensor()])
    toimg = transforms.ToPILImage()
    i = 0
    for idx in range(p_frame_num, p_frame_num+1800):
        img = Image.open(load_path+"/%09d.jpg" % idx).convert('RGB')
        img = transform_fn(img)
        #print(img.size())
        img = F.interpolate(img.unsqueeze(0), size=[512, 512]) # 256, 256
        img = toimg(img.squeeze(0))
        img.save(save_path+"/%09d.jpg" % i)
        print('%s, save a old frame %d as a new frame %d' % (save_path, idx, i))
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

#for 20210609
#startFrameGroup = [
#        139,
#        75,
#        63,
#        147,
#        103,
#        247
#    ]

# fpr 20210628
# startFrameGroup = [
#         156,
#         96,
#         22,
#         167,
#         275,
#         12,
#         21,
#         8,
#         17,
#         16,
#         423,
#         192,
#         207,
#         182,
#         233,
#         205,
#         386,
#         231,
#         171,
#         158,
#         6,
#         8,# should be 14, but total length is not enough
#         6,
#         8,
#         0,
#         8,
#         3,
#         11,
#         7,
#         12
#     ]


#for 20210628
#the video clips should be flipped (isFliped=True)

if __name__ == "__main__":
    
    nameGroup = ['single_%d' % i for i in range(1,18)]
    root = '20210712'
    
    for idxName in range(len(nameGroup)):
        name = nameGroup[idxName]
        pathVideo = "videos/" + root + "/" + name + ".mp4" # ".avi"
        pathRawFrame = "frames/" + root + "/" + name + "/raw"
        numFrames = GenerateFramesfromVideo(pathVideo, pathRawFrame, True)
        #PreprcessFrames(numFrames, pathRawFrame, pathProcessedFrame)

    # startFrameGroup = [
    # ]
    # for idxName in range(len(nameGroup)):
    #     name = nameGroup[idxName]
    #     pathRawFrame = "frames/" + root + "/" + name + "/raw"
    #     pathProcessedFrame = "frames/" + root + "/" + name + "/processed/images"
    #     PreprcessSpecificFrames(pathRawFrame, pathProcessedFrame, startFrameGroup[idxName])
    
    # pathVideoFrame = "/media/pc3426/B2E67301E672C4DF/radar_skeleton_estimation/radar_skeleton_estimation_network/visualization/for20210706_presentation/multiFramesChirps_tmv2_20210707_val_2"
    # pathVideo = "./videos/20210628/multiFramesChirps_tmv2_20210707_val_2.avi"
    # GenerateVideofromFrames(pathVideoFrame, pathVideo)
