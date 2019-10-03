"""
This package performs the pre-processing of the VoxCeleb dataset in order to have it ready for training, speeding the
process up.
"""
import logging
import os
from datetime import datetime
import pickle as pkl
import random
from multiprocessing import Pool

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
from face_alignment import FaceAlignment, LandmarksType
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import mmcv


# region DATASET PREPARATION

def preprocess_dataset(source, output, device='cpu', size=0, overwrite=False):
    """
    Starts the pre-processing of the VoxCeleb dataset used for the Talking Heads models. This process has the following
    steps:

    * Extract all frames of each video in the dataset. Frames of videos that are split in several files are joined
    together.
    * Select K+1 frames of each video that will be kept. K frames will be used to train the embedder network, while
    the other one will be used to train the generator network. The value of K can be configured in the config.py file.
    * Landmarks will be extracted for the face in each of the frames that are being kept.
    * The frames and the corresponding landmarks for each video will be saved in files (one for each video) in the
    output directory.

    We originally tried to process several videos simultaneously using multiprocessing, but this seems to actually
    slow down the process instead of speeding it up.


    :param source: Path to the raw VoxCeleb dataset.
    :param output: Path where the pre-processed videos will be stored.
    :param device: Device used to run the landmark extraction model.
    :param size: Size of the dataset to generate. If 0, the entire raw dataset will be processed, otherwise, as many
    videos will be processed as specified by this parameter.
    :param overwrite: f True, files that have already been processed will be overwritten, otherwise, they will be
    ignored and instead, different files will be loaded.
    """
    logging.info('===== DATASET PRE-PROCESSING =====')
    logging.info(f'Running on {device.upper()}.')
    logging.info(f'Saving K+1 random frames from each video (K = {config.K}).')
    fa = FaceAlignment(LandmarksType._2D, device=device)

    video_list = get_video_list(source, size, output, overwrite=overwrite)

    logging.info(f'Processing {len(video_list)} videos...')
    # pool = Pool(processes=4, initializer=init_pool, initargs=(fa, output))
    # pool.map(process_video_folder, video_list)

    init_pool(fa, output)
    counter = 1
    for v in video_list:
        start_time = datetime.now()
        process_video_folder(v)
        logging.info(f'{counter}/{len(video_list)}\t{datetime.now()-start_time}')
        counter += 1

    logging.info(f'All {len(video_list)} videos processed.')


def get_video_list(source, size, output, overwrite=True):
    """
    Extracts a list of paths to videos to pre-process during the current run.

    :param source: Path to the root directory of the dataset.
    :param size: Number of videos to return.
    :param output: Path where the pre-processed videos will be stored.
    :param overwrite: If True, files that have already been processed will be overwritten, otherwise, they will be
    ignored and instead, different files will be loaded.
    :return: List of paths to videos.
    """
    already_processed = []
    if not overwrite:
        already_processed = [
            os.path.splitext(video_id)[0]
            for root, dirs, files in os.walk(output)
            for video_id in files
        ]

    video_list = []
    counter = 0
    for root, dirs, files in os.walk(source):
        if len(files) > 0 and os.path.basename(os.path.normpath(root)) not in already_processed:
            assert contains_only_videos(files) and len(dirs) == 0
            video_list.append((root, files))
            counter += 1
            if 0 < size <= counter:
                break

    return video_list


def init_pool(face_alignment, output):
    global _FA
    _FA = face_alignment
    global _OUT_DIR
    _OUT_DIR = output


def process_video_folder(video):
    """
    Extracts all frames from a video, selects K+1 random frames, and saves them along with their landmarks.
    :param video: 2-Tuple containing (1) the path to the folder where the video segments are located and (2) the file
    names of the video segments.
    """
    folder, files = video

    try:
        assert contains_only_videos(files)
        frames = np.concatenate([extract_frames(os.path.join(folder, f)) for f in files])

        save_video(
            frames=select_random_frames(frames),
            video_id=os.path.basename(os.path.normpath(folder)),
            path=_OUT_DIR,
            face_alignment=_FA
        )
    except Exception as e:
        logging.error(f'Video {os.path.basename(os.path.normpath(folder))} could not be processed:\n{e}')


def contains_only_videos(files, extension='.mp4'):
    """
    Checks whether the files provided all end with the specified video extension.
    :param files: List of file names.
    :param extension: Extension that all files should have.
    :return: True if all files end with the given extension.
    """
    return len([x for x in files if os.path.splitext(x)[1] != extension]) == 0


def extract_frames(video):
    """
    Extracts all frames of a video file. Frames are extracted in BGR format, but converted to RGB. The shape of the
    extracted frames is [height, width, channels]. Be aware that PyTorch models expect the channels to be the first
    dimension.
    :param video: Path to a video file.
    :return: NumPy array of frames in RGB.
    """
    cap = cv2.VideoCapture(video)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = np.empty((n_frames, h, w, 3), np.dtype('uint8'))

    fn, ret = 0, True
    while fn < n_frames and ret:
        ret, img = cap.read()
        frames[fn] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fn += 1

    cap.release()
    return frames


def select_random_frames(frames):
    """
    Selects K+1 random frames from a list of frames.
    :param frames: Iterator of frames.
    :return: List of selected frames.
    """
    S = []
    while len(S) <= 8 :
        s = random.randint(0, len(frames)-1)
        if s not in S:
            S.append(s)

    return [frames[s] for s in S]


def save_video(path, video_id, frames, face_alignment):
    """
    Generates the landmarks for the face in each provided frame and saves the frames and the landmarks as a pickled
    list of dictionaries with entries {'frame', 'landmarks'}.

    :param path: Path to the output folder where the file will be saved.
    :param video_id: Id of the video that was processed.
    :param frames: List of frames to save.
    :param face_alignment: Face Alignment model used to extract face landmarks.
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    data = []
    for i in range(len(frames)):
        x = frames[i]
        y = face_alignment.get_landmarks_from_image(x)[0]
        data.append({
            'frame': x,
            'landmarks': y,
        })

    filename = f'{video_id}.vid'
    pkl.dump(data, open(os.path.join(path, filename), 'wb'))
    logging.info(f'Saved file: {filename}')

# endregion

# region DATASET RETRIEVAL


class VoxCelebDataset(Dataset):
    """ Dataset object used to access the pre-processed VoxCelebDataset """

    def __init__(self, root, extension='.vid', shuffle=False, transform=None, shuffle_frames=False, subset_size=None):
        """
        Instantiates the Dataset.

        :param root: Path to the folder where the pre-processed dataset is stored.
        :param extension: File extension of the pre-processed video files.
        :param shuffle: If True, the video files will be shuffled.
        :param transform: Transformations to be done to all frames of the video files.
        :param shuffle_frames: If True, each time a video is accessed, its frames will be shuffled.
        """
        self.root = root
        self.transform = transform

        self.files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(root)
            for filename in files
            if filename.endswith(extension)
        ]
        self.files.sort()
        if subset_size is not None:
            self.files = self.files[:subset_size]
        self.length = len(self.files)
        self.indexes = [idx for idx in range(self.length)]

        if shuffle:
            random.shuffle(self.indexes)

        self.shuffle_frames = shuffle_frames

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = self.indexes[idx]
        path = self.files[real_idx]
        data = pkl.load(open(path, 'rb'))
        if self.shuffle_frames:
            random.shuffle(data)

        data_array = []
        for d in data:
            x = PIL.Image.fromarray(d['frame'], 'RGB')
            y = plot_landmarks(d['frame'], d['landmarks'])
            if self.transform:
                x = self.transform(x)
                y = self.transform(y)
            assert torch.is_tensor(x), "The source images must be converted to Tensors."
            data_array.append(torch.stack((x, y)))
        data_array = torch.stack(data_array)

        return real_idx, data_array


class Lmark2rgbDataset(Dataset):
    """ Dataset object used to access the pre-processed VoxCelebDataset """

    def __init__(self, dataset_dir, train = 'train'):
        """
        Instantiates the Dataset.

        :param root: Path to the folder where the pre-processed dataset is stored.
        :param extension: File extension of the pre-processed video files.
        :param shuffle: If True, the video files will be shuffled.
        :param transform: Transformations to be done to all frames of the video files.
        :param shuffle_frames: If True, each time a video is accessed, its frames will be shuffled.
        """
        self.train = train
        self.output_shape   = tuple([256, 256])
        self.num_frames = 64  
        self.root  = dataset_dir     
        if self.train =='train':
            _file = open(os.path.join(dataset_dir, 'txt',  "front_rt2.pkl"), "rb")
            # self.data = pkl.load(_file)
            self.data = pkl._Unpickler(_file)
            self.data.encoding = 'latin1'
            self.data = self.data.load()
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, 'txt', "front_rt2.pkl"), "rb")
            self.data = pkl._Unpickler(_file)
            self.data.encoding = 'latin1'
            self.data = self.data.load()
            # self.data = pkl.load(_file)
            _file.close()
        print (len(self.data))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.data)   

    def __getitem__(self, index):
        v_id = self.data[index][0]
        reference_id = self.data[index][1]

        video_path = os.path.join(self.root, 'unzip', v_id + '.mp4')
        
        ani_video_path = os.path.join(self.root, 'unzip', v_id + '_ani.mp4')


        lmark_path = os.path.join(self.root, 'unzip', v_id + '.npy')


        lmark = np.load(lmark_path)[:,:,:-1]

        v_length = lmark.shape[0]

        real_video  = mmcv.VideoReader(video_path)
        ani_video = mmcv.VideoReader(ani_video_path)

        # sample frames for embedding network
        input_indexs  = set(random.sample(range(0,64), 8))

        # we randomly choose a target frame 
        while True:
            target_id =  np.random.choice([0, v_length - 1])
            if target_id != reference_id and target_id not in input_indexs:
                break
        reference_frames = []
        for t in input_indexs:
            rgb_t =  mmcv.bgr2rgb(real_video[t]) 
            lmark_t = lmark[t]
            lmark_rgb = plot_landmarks( lmark_t)
            lmark_rgb = np.array(lmark_rgb) 

            # resize 224 to 256
            rgb_t  = cv2.resize(rgb_t, self.output_shape)
            lmark_rgb  = cv2.resize(lmark_rgb, self.output_shape)
            
            # to tensor
            rgb_t = self.transform(rgb_t)
            lmark_rgb = self.transform(lmark_rgb)


            reference_frames.append(torch.stack([rgb_t, lmark_rgb]))  # (6, 256, 256)   

        reference_frames = torch.stack(reference_frames)
        ############################################################################
        target_rgb = real_video[target_id]
        reference_rgb = real_video[reference_id]
        reference_ani = ani_video[reference_id]
        target_ani = ani_video[target_id]
        target_lmark = lmark[target_id]

        target_rgb = mmcv.bgr2rgb(target_rgb)
        target_rgb = cv2.resize(target_rgb, self.output_shape)
        target_rgb = self.transform(target_rgb)

        target_ani = mmcv.bgr2rgb(target_ani)
        target_ani = cv2.resize(target_ani, self.output_shape)
        target_ani = self.transform(target_ani)

        # reference_rgb = mmcv.bgr2rgb(reference_rgb)
        # reference_rgb = cv2.resize(reference_rgb, self.output_shape)
        # reference_rgb = self.transform(reference_rgb)

        # reference_ani = mmcv.bgr2rgb(reference_ani)
        # reference_ani = cv2.resize(reference_ani, self.output_shape)
        # reference_ani = self.transform(reference_ani)

        target_lmark = plot_landmarks(target_lmark)
        target_lmark = np.array(target_lmark) 
        target_lmark  = cv2.resize(target_lmark, self.output_shape)
        target_lmark  = cv2.resize(target_lmark, self.output_shape)
        target_lmark = self.transform(target_lmark)


        input_dic = {'v_id' : v_id, 'target_lmark': target_lmark, 'reference_frames': reference_frames,
        'target_rgb': target_rgb, 'target_ani': target_ani
        }
        return input_dic


def plot_landmarks( landmarks):
    """
    Creates an RGB image with the landmarks. The generated image will be of the same size as the frame where the face
    matching the landmarks.

    The image is created by plotting the coordinates of the landmarks using matplotlib, and then converting the
    plot to an image.

    Things to watch out for:
    * The figure where the landmarks will be plotted must have the same size as the image to create, but matplotlib
    only accepts the size in inches, so it must be converted to pixels using the DPI of the screen.
    * A white background is printed on the image (an array of ones) in order to keep the figure from being flipped.
    * The axis must be turned off and the subplot must be adjusted to remove the space where the axis would normally be.

    :param frame: Image with a face matching the landmarks.
    :param landmarks: Landmarks of the provided frame,
    :return: RGB image with the landmarks as a Pillow Image.
    """
    dpi = 100
    fig = plt.figure(figsize=(224/ dpi,224 / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.imshow(np.ones((224,224)))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Head
    ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='green', lw=2)
    # Eyebrows
    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='orange', lw=2)
    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='orange', lw=2)
    # Nose
    ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='blue', lw=2)
    ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='blue', lw=2)
    # Eyes
    ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='red', lw=2)
    ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='red', lw=2)
    # Mouth
    ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='purple', lw=2)

    fig.canvas.draw()
    data = PIL.Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB', 0, 1)
    plt.close(fig)
    return data

# # endregion
# import torchvision


# import time
# dataset = Lmark2rgbDataset('/home/cxu-serve/p1/lchen63/voxceleb/', 'train')
# data_loader = torch.utils.data.DataLoader(dataset,
#                           batch_size=1,
#                           num_workers=1,
#                           shuffle=False, drop_last=True)  
# t1 = time.time()
# print (len(data_loader))
# for (step, gg)  in enumerate(data_loader):
#     print (time.time() -  t1)
#     print  (gg['v_id'])
#     print (gg['reference_frames'].shape)
#     inputs = [gg['target_rgb'], gg['target_ani'], gg['target_lmark'], gg['target_rgb']]
#     fake_im = torch.stack(inputs, dim = 1)
#     fake_store = fake_im.data.contiguous().view(4*1,3,256,256)
#     torchvision.utils.save_image(fake_store, 
#         "./tmp/vis_%05d.png"%step,normalize=True)
    
#     if step == 10:
        # break
   