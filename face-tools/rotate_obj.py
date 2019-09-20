from scipy.spatial.transform import Rotation as R
import numpy as np

def apply_rt(obj_input, rt, obj_output):
    infile = open(obj_input)
    outfile = open(obj_output, "w+")
    R = rt[0]
    T = rt[1]
    for line in infile.read().splitlines():
        if line[0] == "v":
            tokens = line.split()
            x = float(tokens[1])
            y = float(tokens[2])
            z = float(tokens[3])
            v = np.array([x,y,z]).reshape(3,1)
            v_rt = (np.matmul(R, v) + T).reshape(3,)
            tokens[1] = str(v_rt[0])
            tokens[2] = str(v_rt[1])
            tokens[3] = str(v_rt[2])
            outfile.write(" ".join(tokens))
        else:
            outfile.write(line)
        outfile.write("\n")
    
    infile.close()
    outfile.close()

def recover(rt):
    new_rt = []
    for tt in range(rt.shape[0]):
        ret = rt[tt,:3]
        r = R.from_rotvec(ret)
        ret_R = r.as_dcm()
        ret_t = rt[tt, 3:]
        ret_t = ret_t.reshape(3,1)
        new_rt.append([ret_R, ret_t])
    return new_rt


frame_id = "00336"
rt_file = frame_id + "_sRT.npy"
input_obj_file = frame_id + ".obj"
rt = recover(np.load("00336_sRT.npy"))

for i in range(len(rt)):
    out_obj_file = frame_id + "_rt" + str(i) + ".obj"
    apply_rt(input_obj_file, rt[i], out_obj_file)
    print("[{}/{}]".format(i+1, len(rt)))

