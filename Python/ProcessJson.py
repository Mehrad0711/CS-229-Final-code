import pandas as pd
import numpy as np
import os
import pprint
from matplotlib import pyplot as plt
import tarfile
#%matplotlib notebook
from matplotlib import animation, rc
from IPython.display import HTML
from IPython.core.debugger import Tracer
dir_name='./Lukaz_dropbox_Folder/output_some/05965701/json/'
plt.rcParams['animation.ffmpeg_path'] = 'D:\\Users\\ouassimk\\Downloads\\ffmpeg-3.4-win64-static\\ffmpeg-3.4-win64-static\\bin\\ffmpeg.exe'


def get_keypoints(dir_name, nframes):

 ''' This function reads the Json files for given video and extracts up to 4 views (recordings)
     it stores them in 4 matrices
 '''   
 nframes= len([name for name in os.listdir(dir_name)if name.endswith(".json")])


 framesMat=np.empty([nframes, 54])
 framesMat2=np.empty([nframes, 54]) 
 framesMat3=np.empty([nframes, 54])
 framesMat4=np.empty([nframes, 54])
 framesMat[:] = framesMat2[:]=framesMat3[:]=framesMat4[:]=np.NAN
 ind=0
 for  ind, filename in enumerate(os.listdir(dir_name)):
    if filename.endswith(".json"): 
        #Tracer()() 
        frame= pd.read_json(os.path.join(dir_name, filename))
        if frame.shape[0]>1:
            people=frame['people']
            valuesPeople=pd.DataFrame.from_records(people.values)
            c=valuesPeople.drop(['face_keypoints','hand_left_keypoints','hand_right_keypoints'], axis=1)
            nPeople=c.shape[0]
            if nPeople==4:
                framesMat[ind,:]=np.array(c.iloc[0]['pose_keypoints'])
                framesMat2[ind,:]=np.array(c.iloc[1]['pose_keypoints'])
                framesMat3[ind,:]=np.array(c.iloc[2]['pose_keypoints'])
                framesMat4[ind,:]=np.array(c.iloc[3]['pose_keypoints'])                    
            elif nPeople==3:
                framesMat[ind,:]=np.array(c.iloc[0]['pose_keypoints'])
                framesMat2[ind,:]=np.array(c.iloc[1]['pose_keypoints'])
                framesMat3[ind,:]=np.array(c.iloc[2]['pose_keypoints'])
            elif nPeople==2:
                framesMat[ind,:]=np.array(c.iloc[0]['pose_keypoints'])
                framesMat2[ind,:]=np.array(c.iloc[1]['pose_keypoints'])
            elif nPeople==1:
                framesMat[ind,:]=np.array(c.iloc[0]['pose_keypoints'])               
           # ind+=1
        #print(os.path.join(dir_name, filename))
        continue
    else:
        continue
 return framesMat, framesMat2, framesMat3,framesMat4, nframes

def plot_keypoints_i(framesMat,framesMat2,framesMat3,framesMat4,nframes, joint_i):
   ''' joint_i= represents either the x or y of a joint
       nframes= number of frames in the video
   '''
   time=np.arange(1,nframes+1)
   fig, axes = plt.subplots(2, 2)
   axes[0,0].plot(time, framesMat[:,joint_i],'r*',time, framesMat2[:,joint_i],'k.',time,framesMat3[:,joint_i],'k.',time, framesMat4[:,joint_i],'k.')
   axes[0,1].plot(time, framesMat[:,joint_i], 'k.',time, framesMat2[:,joint_i],'r*', time,framesMat3[:,joint_i],'k.', time, framesMat4[:,joint_i],'k.')
   axes[1,0].plot(time, framesMat[:,joint_i],'k.',time,framesMat2[:,joint_i] ,'k.', time,framesMat3[:,joint_i], 'r*',time, framesMat4[:,joint_i],'k.')
   axes[1,1].plot(time, framesMat[:,joint_i],'k.', time, framesMat2[:,joint_i], 'k.',time,framesMat3[:,joint_i],'k.', time, framesMat4[:,joint_i],'r*')

   axes[0,0].legend(['Per1', 'Per2', 'Per3', 'Per4'])
   axes[0,1].legend(['Per1', 'Per2', 'Per3', 'Per4'])
   axes[1,0].legend(['Per1', 'Per2', 'Per3', 'Per4'])
   axes[1,1].legend(['Per1', 'Per2', 'Per3', 'Per4'])

   return fig
def plot_keypoints_i_2(framesMat,framesMat2,framesMat3,framesMat4,nframes, joint_i):
   ''' joint_i= represents either the x or y of a joint
       nframes= number of frames in the video
   '''
   time=np.arange(1,nframes+1)
   fig, axes = plt.subplots()
   axes.plot(time, framesMat[:,joint_i],'.',time, framesMat2[:,joint_i],'.',time,framesMat3[:,joint_i],'.',time, framesMat4[:,joint_i],'.')
   
   axes.legend(['Entry 1', 'Entry 2', 'Entry 3', 'Entry 4'])
   axes.set_xlabel('Frame')
   axes.set_ylabel('x')
   return fig

def plot_extract_views(Per1,Per2,nframes):
  ''' This is a ploting function for a specific joint
      time series in both views
  '''  
  time=np.arange(1,nframes+1)
  f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
  ax1.plot(time, Per1[:,0], '+',time, Per2[:,0],'*')
  ax1.legend(['View1-Nose- x ', 'View2-Nose- x ']);
  ax2.plot(time, Per1[:,0],'+', time, Per2[:,0],'*')
  ax2.legend(['View1-Nose- y ', 'View2-Nose- y ']);
  ax1.set_xlabel('Frame')
  ax2.set_xlabel('Frame')
  ax1.set_ylabel('x')
  ax2.set_ylabel('y')

  return f
def plot_extract_views_2(Per1,Per2,nframes,joint_i):
  ''' This is a ploting function for a specific joint
      time series in both views
  '''  
  time=np.arange(1,nframes+1)
  f, ax1= plt.subplots()
  ax1.plot(time, Per1[:,joint_i], '.',time, Per2[:,joint_i],'.')
  ax1.legend(['View1-Nose- x ', 'View2-Nose- x ']);
  ax1.set_xlabel('Frame')
  ax1.set_ylabel('x')

  return f
def process_Json_keypoints(framesMat,framesMat2,framesMat3,framesMat4,nfiles):
 ''' This function processes the four matrices of view 1, 2,3,4 to extract the 
     correct left and right views

     it returns view1 and view2 (here named per1 and per2) 
 '''
 meanFrames=np.empty([nfiles,2])
 meanFrames2=np.empty([nfiles,2])
 meanFrames3=np.empty([nfiles,2])
 meanFrames4=np.empty([nfiles,2])
 
 # to take the mean of the body we only consider te points: 0,1,2,5,8,11,14,15,16, 17 
 # which represents: nose, neck, LSHO, RSHO, LHIP, RHIP, LEYE,REYE, LEAR, REAR
 # this should give us a more stable body center
 
 # xInd is the x for the above points
 xInd=np.array([0,3,6,15,24,33,42,45,48,51])
 # yInd is the y for the above points

 framesMat[framesMat[:]<=0]=np.nan
 framesMat2[framesMat2[:]<=0]=np.nan
 framesMat3[framesMat3[:]<=0]=np.nan
 framesMat4[framesMat4[:]<=0]=np.nan
 yInd=xInd+1

 xframeMat=framesMat[:,xInd];  yframeMat= framesMat[:,yInd]; indFrameMat= np.logical_and(~np.isnan(xframeMat),~np.isnan(yframeMat) )
 xframeMat2=framesMat2[:,xInd];  yframeMat2= framesMat2[:,yInd]; indFrameMat2= np.logical_and(~np.isnan(xframeMat2),~np.isnan(yframeMat2) )
 xframeMat3=framesMat3[:,xInd];  yframeMat3= framesMat3[:,yInd]; indFrameMat3= np.logical_and(~np.isnan(xframeMat3),~np.isnan(yframeMat3) )
 xframeMat4=framesMat4[:,xInd];  yframeMat4= framesMat4[:,yInd]; indFrameMat4= np.logical_and(~np.isnan(xframeMat4),~np.isnan(yframeMat4) )

 for i in range(xframeMat.shape[0]):
  meanFrames[i,0] =np.nanmean(xframeMat[i,indFrameMat[i,:]]);  meanFrames[i,1] =np.nanmean(yframeMat[i,indFrameMat[i,:]])
  meanFrames2[i,0] =np.nanmean(xframeMat2[i,indFrameMat2[i,:]]);  meanFrames2[i,1] =np.nanmean(yframeMat2[i,indFrameMat2[i,:]])
  meanFrames3[i,0] =np.nanmean(xframeMat3[i,indFrameMat3[i,:]]);  meanFrames3[i,1] =np.nanmean(yframeMat3[i,indFrameMat3[i,:]])
  meanFrames4[i,0] =np.nanmean(xframeMat4[i,indFrameMat4[i,:]]);  meanFrames4[i,1] =np.nanmean(yframeMat4[i,indFrameMat4[i,:]])

 Per1= np.empty(framesMat.shape)
 Per2= np.empty(framesMat.shape)
 Per1[:]=Per2[:]= np.NAN
 Per1[0,:]=framesMat[0,:]
 Per2[0,:]=framesMat2[0,:]
 for i in np.arange(1,nfiles):
    list=np.stack((meanFrames[i,0:2],meanFrames2[i,0:2],meanFrames3[i,0:2],meanFrames4[i,0:2]))
    list=list[~np.isnan(list).any(axis=1)]
    
    listMat=np.stack((framesMat[i,:],framesMat2[i,:],framesMat3[i,:],framesMat4[i,:]))
    listMat=listMat[~np.isnan(listMat).any(axis=1)]

    mPer1= np.nanmean(Per1[:i,:],axis=0)
    meanPer1=np.empty([1,2])
    meanPer1[0,0] =np.nanmean(mPer1[xInd]);  meanPer1[0,1] =np.nanmean(mPer1[yInd])

    mPer2= np.nanmean(Per2[:i,:],axis=0)
    meanPer2=np.empty([1,2])
    meanPer2[0,0] =np.nanmean(mPer2[xInd]);  meanPer2[0,1] =np.nanmean(mPer2[yInd])
    if list.shape[0]>0:
        min1=np.min(np.linalg.norm(meanPer1-list,axis=1))
        min2= np.min(np.linalg.norm(meanPer2-list,axis=1))
        ind1=np.argmin(np.linalg.norm(meanPer1-list,axis=1))
        ind2= np.argmin(np.linalg.norm(meanPer2-list,axis=1))
        if ind1==ind2:
            if min1<min2:
                if list.shape[0]==1:               
                     Per1[i,:]=listMat[ind1,:]
                     continue
                elif list.shape[0]>1:
                   ind2 = np.argmin(np.linalg.norm(meanPer2-list[np.arange(len(list))!=ind1],axis=1))
            elif min2<min1:
                if list.shape[0]==1:
                     Per2[i,:]=listMat[ind2,:]
                     continue
                elif list.shape[0]>1:
                    ind1= np.argmin(np.linalg.norm(meanPer2-list[np.arange(len(list))!=ind2],axis=1))
                
        Per1[i,:]=listMat[ind1]
        Per2[i,:]=listMat[ind2]
    else:
        Per1[i,:]=np.nan
        Per2[i,:]=np.nan

 # separate view1 (Per1) and view2 (Per2) using the middle of the frame 320
 Per1[np.nanmax(Per1[:,::3],axis=1)<320,:]=np.NAN
 Per2[np.nanmax(Per2[:,::3],axis=1)>320,:]=np.NAN

  

 return Per1, Per2
def check_views(view1,view2):
    ''' This is simple data quality check for the views 1 and 2
    '''
    ind1_noNan=~np.isnan(view1[:,0])
    ind2_noNan=~np.isnan(view2[:,0])

    fracV1= view1[ind1_noNan,0].shape[0]/view1[:,0].shape[0]
    fracV2= view2[ind2_noNan,0].shape[0]/view2[:,0].shape[0]

    if fracV1<0.5 or fracV2<0.5:
        return False
    return True


def main():
 parent_floder= './Lukaz_dropbox_Folder/output_some'
 out_folder= './Lukaz_dropbox_Folder/view_test'
 fig_folder='./Lukaz_dropbox_Folder/fig_test'
 for folder in os.listdir(parent_floder):
     folderPath= os.path.join(parent_floder,folder, 'json')
     nframes= len([name for name in os.listdir(folderPath)if name.endswith(".json")])

     if nframes >0:
        # process the video's json files
        framesMat, framesMat2, framesMat3,framesMat4, nframes= get_keypoints(folderPath,nframes)
        # extract view 1 and 2
        Per1, Per2 = process_Json_keypoints(framesMat,framesMat2,framesMat3,framesMat4,nframes)
        newpath= os.path.join(out_folder, folder)
        if not os.path.exists(newpath):
           os.makedirs(newpath)

        # check if data separation is good
        if check_views(Per1,Per2):
         view1= os.path.join(newpath, 'view1.npy')   
         np.save(view1, Per1)
         view2= os.path.join(newpath, 'view2.npy')   
         np.save(view2, Per2)

         #save plots
         fig1=plot_keypoints_i_2(framesMat,framesMat2,framesMat3,framesMat4,nframes, 24)
         fig2=plot_extract_views_2(Per1,Per2,nframes,24)

         fig1_path=os.path.join(fig_folder, folder+'_fig1.png')
         fig2_path=os.path.join(fig_folder, folder+'_fig2.png')

         fig1.savefig(fig1_path)
         fig2.savefig(fig2_path)

if __name__=='__main__':
    main()