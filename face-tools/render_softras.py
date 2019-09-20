import soft_renderer as sr
import soft_renderer.cuda.create_texture_image as create_texture_image_cuda
import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

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
                

def render_mesh_rt(vertices, triangles, colors, rot=None, tran=None):
    ''' Apply the rt(rotation and translation) to a mesh then render it to an image
    Args:
        vertices:  torch.Tensor, shape = [nver, 3] (on gpu)
        triangles: torch.Tensor, shape = [ntri, 3] (on gpu)
        colors:    torch.Tensor, shape = [nver, 3] (on gpu)
        r: torch.Tensor, shape=[3,3] (on gpu)
        t: torch.Tensor, shape=[3,1] (on gpu)
    Return:
        image: torch.Tensor, shape=[4, width, height] (on gpu)
    '''
    # vertices = torch.Tensor(vertices).cuda()
    # triangles = torch.Tensor(triangles).cuda().int()
    # colors = torch.Tensor(colors).cuda()
    # rot = torch.Tensor(rot).cuda()
    # tran = torch.Tensor(trans).cuda()

    # apply r and t
    if type(rot) == torch.Tensor and type(tran) == torch.Tensor:
        # new_vertices = (rot @ vertices.transpose(1,0) + tran).transpose(1,0)
        new_vertices = (rot.T @ (vertices.transpose(1,0) - tran)).transpose(1,0)
    else:
        new_vertices = vertices

    # prepare the mesh
    face_mesh = sr.Mesh(new_vertices, triangles, colors, texture_type="vertex")

    # prepare the renderer
    # renderer = sr.SoftRenderer(camera_mode="look_at", far=10000, image_size=512, camera_direction=[0,1,0], light_directions=[0,0,1], dist_func="hard") # the dist_func can be 'hard', 'barycentric' or 'euclidean'
    # renderer.transform.set_eyes_from_angles(-400, 0, 0) # this three number can be trainable parameters as well
    renderer = sr.SoftRenderer(camera_mode="look", far=10000, image_size=224, viewing_angle=15, camera_direction=[0,0,-1], camera_up=[-0.3,1,0], light_intensity_ambient=1, light_color_ambient=[1,1,1], dist_func="hard")
    renderer.transform.set_eyes([0,-50,680])
    # do the rendering
    images = renderer.render_mesh(face_mesh)
    image = images[0]
    image = torch.flip(image, [2])
    return image



if __name__ == "__main__":
    face_id = "00336"
    mesh_file = "00025/00025.obj"
    rt_file = "00025/00025_sRT.npy"

    face_mesh = sr.Mesh.from_obj(mesh_file, load_texture=True, texture_type="vertex")
    renderer = sr.SoftRenderer(camera_mode="look", far=10000, image_size=224, viewing_angle=15, camera_direction=[0,0,-1], camera_up=[-0.3,1,0], light_intensity_ambient=1, light_color_ambient=[1,1,1], dist_func="hard")
    renderer.transform.set_eyes([0,-50,680])
    # do the rendering
    images = renderer.render_mesh(face_mesh)
    image = images[0]
    image = torch.flip(image, [2])
    image = image.detach().cpu().numpy()
    image = image.transpose((1,2,0))
    plt.imshow(image)
    plt.show()
"""
    # load mesh and rt from files to np.array
    vertices, triangles, colors = load_obj(mesh_file)
    rots, trans = recover(np.load(rt_file))

    # convert them to torch.Tensor and send to gpu
    vertices_d = torch.Tensor(vertices).cuda()
    triangles_d = torch.Tensor(triangles).cuda().int()
    colors_d = torch.Tensor(colors).cuda()
    rots_d = torch.Tensor(rots).cuda()
    trans_d = torch.Tensor(trans).cuda()


    ### Example one: render one frame by applying the first rt and generate the redered image
    image = render_mesh_rt(vertices_d, triangles_d, colors_d, rots_d[58], trans_d[58])
    # image = render_mesh_rt(vertices_d, triangles_d, colors_d)
    # send the image to cpu and plot it
    image = image.detach().cpu().numpy()
    image = image.transpose((1,2,0))
    plt.imshow(image)
    plt.show()
"""

"""
    ### Example two: render all frames and produce a video
    
    fig = plt.figure()
    ims = []
    
    # render the first frame without applying r and t
    # image = render_mesh_rt(vertices_d, triangles_d, colors_d).detach().cpu().numpy().transpose((1,2,0))
    # im = plt.imshow(image, animated=True)
    # ims.append([im])

    # render the rest frames
    for i in range(rots.shape[0]):
        image = render_mesh_rt(vertices_d, triangles_d, colors_d, rots_d[i], trans_d[i]).detach().cpu().numpy().transpose((1,2,0))
        im = plt.imshow(image, animated=True)
        ims.append([im])
        print("[{}/{}]".format(i+1, rots.shape[0]))

    itvl = 8000 / (rots.shape[0]-1)
    ani = animation.ArtistAnimation(fig, ims, interval=itvl, blit=True, repeat_delay=1000)
    # ani.save('dynamic_images.mp4')
    plt.show()
"""