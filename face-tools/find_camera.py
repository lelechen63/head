import sys
import soft_renderer as sr
import imageio
from skimage.transform import warp
from skimage.transform import AffineTransform
import numpy as np
import cv2

import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
import time
res = 224
import os
import pickle
root  = '/home/cxu-serve/p1/lchen63/voxceleb/'
import shutil

def recover(rt):
    rots = []
    trans = []
    for tt in range(rt.shape[0]):
        ret = rt[tt,:3]
        r = R.from_rotvec(ret)
        ret_R = r.as_dcm()
        ret_t = rt[tt, 3:]
        ret_t = ret_t.reshape(3,1)
        rots.append(ret_R)
        trans.append(ret_t)
    return (np.array(rots), np.array(trans))

def load_obj(obj_file):
    vertices = []

    triangles = []
    colors = []

    with open(obj_file) as infile:
        for line in infile.read().splitlines():
            if len(line) > 2 and line[:2] == "v ":
                ts = line.split()
                x = float(ts[1])
                y = float(ts[2])
                z = float(ts[3])
                r = float(ts[4])
                g = float(ts[5])
                b = float(ts[6])
                vertices.append([x,y,z])
                colors.append([r,g,b])
            elif len(line) > 2 and line[:2] == "f ":
                ts = line.split()
                fx = int(ts[1]) - 1
                fy = int(ts[2]) - 1
                fz = int(ts[3]) - 1
                triangles.append([fx,fy,fz])
    
    return (np.array(vertices), np.array(triangles).astype(np.int), np.array(colors))

def setup_renderer():    
    renderer = sr.SoftRenderer(camera_mode="look", viewing_scale=2/res, far=10000, perspective=False, image_size=res, camera_direction=[0,0,-1], camera_up=[0,1,0], light_intensity_ambient=1)
    renderer.transform.set_eyes([res/2, res/2, 6000])
    return renderer
def get_np_uint8_image(mesh, renderer):
    images = renderer.render_mesh(mesh)
    image = images[0]
    image = torch.flip(image, [1,2])
    image = image.detach().cpu().numpy().transpose((1,2,0))
    image = np.clip(image, 0, 1)
    image = (255*image).astype(np.uint8)
    return image


def demo():
    key_id = 95 # index of the frame used to do the 3D face reconstruction (key frame)
    model_id = "00218"
    itvl = 1000.0/25.0 # 25fps
    


    # extract the frontal facial landmarks for key frame
    lmk3d_all = np.load("{}/{}_front.npy".format(model_id, model_id))
    lmk3d_target = lmk3d_all[key_id]

    # load the 3D facial landmarks on the PRNet 3D reconstructed face
    lmk3d_origin = np.load("{}/{}_prnet.npy".format(model_id, model_id))
    # lmk3d_origin[:,1] = res - lmk3d_origin[:,1]

    # load RTs for all frame
    rots, trans = recover(np.load("{}/{}_sRT.npy".format(model_id, model_id)))

    # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
    lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
    p_affine = np.linalg.lstsq(lmk3d_origin_homo, lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
    pr = p_affine[:,:3] # 3x3
    pt = p_affine[:,3:] # 3x1

    # load the original 3D face mesh then transform it to align frontal face landmarks
    vertices_org, triangles, colors = load_obj("{}/{}_original.obj".format(model_id, model_id)) # get unfrontalized vertices position
    vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

    # set up the renderer
    renderer = setup_renderer()

    if overlay:
        real_video = mmcv.VideoReader("{}/{}.mp4".format(model_id, model_id))

    fig = plt.figure()
    ims = []

    for i in range(rots.shape[0]):
        # get rendered frame
        vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
        face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
        image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8

        if overlay:
            frame = mmcv.bgr2rgb(real_video[i]) # RGB, (224,224,3), np.uint8

        if not overlay:
            im = plt.imshow(image_render, animated=True)
        else:
            im = plt.imshow((frame[:,:,:3] * 0.5 + image_render[:,:,:3] * 0.5).astype(np.uint8), animated=True)
        
        ims.append([im])
        print("[{}/{}]".format(i+1, rots.shape[0])) 
    

    ani = animation.ArtistAnimation(fig, ims, interval=itvl, blit=True, repeat_delay=1000)
    if not overlay:
        ani.save('{}/{}_render.mp4'.format(model_id, model_id))
    else:
        ani.save('{}/{}_overlay.mp4'.format(model_id, model_id))
    plt.show()


def get(batch = 0 ):
    # key_id = 58 #
    # model_id = "00025"

    _file = open(os.path.join(root, 'txt',  "front_rt.pkl"), "rb")
    data = pickle._Unpickler(_file)
    data.encoding = 'latin1'

    data = data.load()
    _file.close()
    flage = False
    for k, item in enumerate(data[2000* batch:2000* (batch + 1)]):
        


        key_id = item[-1]
        # if  k == 5689:
        #     flage = True

        # if flage == False:
        #     continue
        print ('++++++++++++++++++++++++++++++++%d'%(k + 2000* batch))
        video_path = os.path.join(root, 'unzip', item[0] + '.mp4') 

        reference_img_path = video_path[:-4] + '_%05d.png'%key_id

        reference_prnet_lmark_path = video_path[:-4] +'_prnet.npy'

        original_obj_path = video_path[:-4] + '_original.obj'

        rt_path  = video_path[:-4] + '_sRT.npy'
        lmark_path  = video_path[:-4] +'_front.npy'


        # if os.path.exists( video_path[:-4] + '_ani.mp4'):
        #     print ('=====')
        #     continue

        if  not os.path.exists(original_obj_path) or not os.path.exists(reference_prnet_lmark_path) or not os.path.exists(lmark_path) or not os.path.exists(rt_path):
            
            print (original_obj_path)
            print ('++++')
            continue

        # extract the frontal facial landmarks for key frame
        lmk3d_all = np.load(lmark_path)
        lmk3d_target = lmk3d_all[key_id]


        # load the 3D facial landmarks on the PRNet 3D reconstructed face
        lmk3d_origin = np.load(reference_prnet_lmark_path)
        # lmk3d_origin[:,1] = res - lmk3d_origin[:,1]
        
        

        # load RTs
        rots, trans = recover(np.load(rt_path))

        # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
        lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
        p_affine = np.linalg.lstsq(lmk3d_origin_homo, lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
        pr = p_affine[:,:3] # 3x3
        pt = p_affine[:,3:] # 3x1

        # load the original 3D face mesh then transform it to align frontal face landmarks
        vertices_org, triangles, colors = load_obj(original_obj_path) # get unfrontalized vertices position
        vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

        # set up the renderer
        renderer = setup_renderer()
        # generate animation

        temp_path = './tempo_%05d'%batch

        # generate animation
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.mkdir(temp_path)
        # writer = imageio.get_writer('rotation.gif', mode='I')
        for i in range(rots.shape[0]):
                # get rendered frame
            vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
            face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
            image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
            
            #save rgba image as bgr in cv2
            rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
            cv2.imwrite( temp_path +  "/%05d.png"%i, rgb_frame)  
        command = 'ffmpeg -framerate 25 -i '  + temp_path + '/%5d.png  -c:v libx264 -y -vf format=yuv420p ' +   video_path[:-4] + '_ani.mp4'
        os.system(command)
        # break
import mmcv
def vis_single(video_path, key_id, save_name):
    overlay = True
    #key_id = 79 #
    #video_path = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/test_video/id04276/k0zLls_oen0/00341.mp4'
    reference_img_path = video_path[:-4] + '_%05d.png'%key_id
    reference_prnet_lmark_path = video_path[:-4] +'_prnet.npy'

    original_obj_path = video_path[:-4] + '_original.obj'

    rt_path  = video_path[:-4] + '_sRT.npy'

    lmark_path  = video_path[:-4] +'_front.npy'



    # extract the frontal facial landmarks for key frame
    lmk3d_all = np.load(lmark_path)
    lmk3d_target = lmk3d_all[key_id]


    # load the 3D facial landmarks on the PRNet 3D reconstructed face
    lmk3d_origin = np.load(reference_prnet_lmark_path)
    # lmk3d_origin[:,1] = res - lmk3d_origin[:,1]


    # load RTs
    rots, trans = recover(np.load(rt_path))

     # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
    lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
    p_affine = np.linalg.lstsq(lmk3d_origin_homo, lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
    pr = p_affine[:,:3] # 3x3
    pt = p_affine[:,3:] # 3x1

    # load the original 3D face mesh then transform it to align frontal face landmarks
    vertices_org, triangles, colors = load_obj(original_obj_path) # get unfrontalized vertices position
    vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

    # set up the renderer
    renderer = setup_renderer()
    # generate animation

    if os.path.exists('./tempo1'):
        shutil.rmtree('./tempo1')
    os.mkdir('./tempo1')
    if overlay:
        real_video = mmcv.VideoReader(video_path)

    for i in range(rots.shape[0]):
        t = time.time()
        

        # get rendered frame
        vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
        face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
        image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
        print (image_render.shape)
        print (image_render.max())
        print (image_render.min())
        #save rgba image as bgr in cv2
        rgb_frame =  (image_render ).astype(int)[:,:,:-1][...,::-1]
        overla_frame = (0.5* rgb_frame + 0.5 * real_video[i]).astype(int)
        cv2.imwrite("./tempo1/%05d.png"%i, overla_frame)


        print (time.time() - t)
        # writer.append_data((255*warped_image).astype(np.uint8))

        print("[{}/{}]".format(i+1, rots.shape[0]))    
        # if i == 5:
        #     breakT
    t = time.time()
    ani_mp4_file_name = save_name  # './fuck.mp4'
    command = 'ffmpeg -framerate 25 -i ./tempo1/%5d.png  -c:v libx264 -y -vf format=yuv420p ' + ani_mp4_file_name 
    os.system(command)
    print (time.time() - t)
    
import random    
def gg():
    _file = open(os.path.join(root, 'txt',  "front_rt.pkl"), "rb")
    data = pickle._Unpickler(_file)
    data.encoding = 'latin1'

    data = data.load()
    _file.close()
    print (len(data))
    # random.shuffle(data)
    for k, item in enumerate(data):



        key_id = item[-1]
        
        video_path = os.path.join(root, 'unzip', item[0] + '_ani.mp4')
        # save_name = './tempo2/' + item[0].replace('/', '_') + '.mp4' 
        if k % 10 ==0 and k > 1800 and k < 1900:
            print (video_path)
        # vis_single(video_path, key_id, save_name)
        # if k == 10:
        #     break
# demo()
# gg()
get(6)
