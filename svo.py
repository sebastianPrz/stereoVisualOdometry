import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy import array

img = cv2.imread("C:\\users\\seb\\desktop\\left\\0000100.png", 0)

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header
'''

def write_ply(fn, verts):
    verts = verts.reshape(-1, 3)
    verts = np.hstack([verts])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f')


# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create(threshold=20)

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, None,color=(255,0,0))

print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Total Keypoints with nonmaxSuppression: ", len(kp))
window_size = 3
cv2.imwrite("C:\\users\\seb\\desktop\\right\\0000100.png",img2)

kp = fast.detect(img,None)

print(type(kp))
print "Total Keypoints without nonmaxSuppression: ", len(kp)
img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

numpyKP = np.array(kp)

nextPts, status, error = cv2.calcOpticalFlowPyrLK(img, img2, numpyKP, None, **lk_params)

img4 = cv2.drawKeypoints(img2, nextPts, None, color=(255,0,0))

plt.imshow(img4, "gray")
plt.show()

min_disp = 16
num_disp = 32-min_disp
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=16,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
    )

imgR = cv2.pyrDown(cv2.imread('C:\\Users\\Seb\\Desktop\\right\\0000100.png'))  # downscale images for faster processing
imgL = cv2.pyrDown(cv2.imread('C:\\Users\\Seb\\Desktop\\left\\0000100.png'))
disp = stereo.compute(imgR, imgL).astype(np.float32) / 16.0

f = 1585.56                          # guess for focal length
Q = np.float32([[1, 0, 0, -643.03232],
                [0,1, 0, -472.15968], # turn points 180 deg around x-axis,
                [0, 0, 0, -f], # so that y-axis looks up
                [0, 0, 0.2399, 0]])
points = cv2.reprojectImageTo3D(disp, Q)
mask = disp > disp.min()
out_points = points[mask]
out_fn = 'out1.ply'
write_ply(out_fn, out_points)
print('%s saved' % 'out.ply')