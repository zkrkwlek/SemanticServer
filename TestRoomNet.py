import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from roomnet.net import *
#from roomnet.get_res import get_im
roomnet_model_dir = "./roomnet/model"

####result image visualization
import numpy as np
import cv2
l_list=[0,8,14,20,24,28,34,38,42,44,46,48]
l_list=np.array(l_list, dtype = np.uint32)
colors=np.array([[255,0,0],[0,0,255],[0,255,0],[255,255,0],[255,0,255], [0,255,255],[139,126,102], [118,238,198]])

def guassian_2d(x_mean, y_mean, dev=2.0):
  x, y = np.meshgrid(np.arange(40), np.arange(40))
  #z=(1.0/(2.0*np.pi*dev*dev))*np.exp(-((x-x_mean)**2+ (y-y_mean)**2)/(2.0*dev**2))
  z=np.exp(-((x-x_mean)**2+ (y-y_mean)**2)/(2.0*dev**2))
  return z
def get_im(ims, layout, label, j):

    #print(l_list[label],l_list[label + 1])
    lay = layout[:, :, int(l_list[label]):int(l_list[label + 1])]
    num = lay.shape[2]
    #print(num)
    outim = np.zeros((40, 40, 3))
    pts = []
    for i in range(num):
        position = np.where(lay[:, :, i] == np.max(lay[:, :, i]))
        x1 = position[0][0]
        x2 = position[1][0]
        pts.append([x2, x1])
        im2 = guassian_2d(x2, x1).reshape(40, 40, 1)
        outim += im2 * colors[i]
    outim = cv2.resize(outim, (320, 320))
    #    res=cv2.addWeighted(ims*255, 0.7, outim, 0.5, 0)
    pt = np.array(pts, np.int32) * 8
    pt = tuple(map(tuple, pt))
    outim = np.array(outim, np.uint8)
    l = 3
    #    for i  in range(num):
    #      cv2.circle(outim, pts[i], 5, colors[i], -1)
    print(pt)
    if label == 0:
        cv2.line(outim, pt[1], pt[0], (255, 0, 0), l)
        cv2.line(outim, pt[0], pt[6], (255, 0, 0), l)
        cv2.line(outim, pt[6], pt[7], (255, 0, 0), l)
        cv2.line(outim, pt[0], pt[2], (255, 0, 0), l)
        cv2.line(outim, pt[6], pt[4], (255, 0, 0), l)
        cv2.line(outim, pt[2], pt[3], (255, 0, 0), l)
        cv2.line(outim, pt[2], pt[4], (255, 0, 0), l)
        cv2.line(outim, pt[4], pt[5], (255, 0, 0), l)
    if label == 1:
        cv2.line(outim, pt[0], pt[3], (255, 0, 0), l)
        cv2.line(outim, pt[1], pt[4], (255, 0, 0), l)
        cv2.line(outim, pt[0], pt[1], (255, 0, 0), l)
        cv2.line(outim, pt[3], pt[4], (255, 0, 0), l)
        cv2.line(outim, pt[0], pt[2], (255, 0, 0), l)
        cv2.line(outim, pt[3], pt[5], (255, 0, 0), l)

    if label == 2:
        cv2.line(outim, pt[0], pt[1], (255, 0, 0), l)
        cv2.line(outim, pt[3], pt[4], (255, 0, 0), l)
        cv2.line(outim, pt[0], pt[3], (255, 0, 0), l)
        cv2.line(outim, pt[0], pt[2], (255, 0, 0), l)
        cv2.line(outim, pt[3], pt[5], (255, 0, 0), l)
    if label == 3 or label == 4:
        cv2.line(outim, pt[0], pt[1], (255, 0, 0), l)
        cv2.line(outim, pt[0], pt[2], (255, 0, 0), l)
        cv2.line(outim, pt[0], pt[3], (255, 0, 0), l)
    if label == 5:
        cv2.line(outim, pt[3], pt[5], (255, 0, 0), l)
        cv2.line(outim, pt[3], pt[4], (255, 0, 0), l)
        cv2.line(outim, pt[0], pt[1], (255, 0, 0), l)
        cv2.line(outim, pt[0], pt[2], (255, 0, 0), l)
        cv2.line(outim, pt[0], pt[3], (255, 0, 0), l)

    if label == 6 or label == 7:
        cv2.line(outim, pt[0], pt[1], (255, 0, 0), l)
        cv2.line(outim, pt[2], pt[3], (255, 0, 0), l)
    if label == 8 or label == 9 or label == 10:
        cv2.line(outim, pt[0], pt[1], (255, 0, 0), l)
    #    outim=np.array(outim, np.float32)
    outim = cv2.resize(outim, (320, 320))
    ims = np.array(ims * 255, np.uint8)
    res = cv2.addWeighted(ims, 0.5, outim, 0.5, 0)

    return res
####result image visualization

##LOAD Roomnet
roomnet_config = tf.ConfigProto()
roomnet_config.gpu_options.allow_growth = True
roomnet_config.allow_soft_placement = True
roomnet_sess = tf.Session(config=roomnet_config)
#device = '/gpu:1'
net = RoomnetVanilla()
#net = RcnnNet()
net.build_model()
start_step = net.restore_model(roomnet_sess, roomnet_model_dir)
print('restored')

from PIL import Image
#image = Image.open("./roomnet/test/4.jpg")

im = cv2.imread("./roomnet/test/wean2.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im, (320, 320) )
#image.show()

im=np.array(im, dtype=np.float32)/255.0
im_in = im.reshape(1,320,320,3)

#im2=np.array(im_in[0]*255.0, dtype=np.uint8)
#img_in = Image.fromarray(im2)
#img_in.show()

layout=np.zeros((1,40, 40, 48))
label=np.zeros((1,11))
net.set_feed(im_in, layout, label,0)
pred_class, pred_lay=net.run_result(roomnet_sess)
c_out=np.argmax(pred_class, axis=1)
#pred_lay = np.array(pred_lay, dtype = pred_lay.dtype)
#pred_lay=pred_lay.astype(np.int64)
outim = get_im(im, pred_lay[0], c_out, 0)
img_out = Image.fromarray(outim)
img_out.show()
print('predict')

##LOAD Roomnet