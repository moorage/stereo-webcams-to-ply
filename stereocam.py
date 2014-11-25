#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

import numpy as np
import cv2

MAX_IMAGES = 5

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

# do_stereo( cv2.imread('../gpu/aloeL.jpg'), cv2.imread('../gpu/aloeR.jpg'), 'out.py')
def do_stereo(inputL,inputR,outfile):
    print 'loading images...'
    imgL = cv2.pyrDown( inputL  )  # downscale images for faster processing
    imgR = cv2.pyrDown( inputR  )

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM(minDisparity = min_disp,
        numDisparities = num_disp,
        SADWindowSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 1,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        fullDP = False
    )

    print 'computing disparity...'
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print 'generating 3d point cloud...',
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    write_ply(outfile, out_points, out_colors)
    print '%s saved' % outfile

    # cv2.imshow('left', imgL)
    # cv2.imshow('disparity', (disp-min_disp)/num_disp)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    webcam0 = cv2.VideoCapture(1)
    webcam1 = cv2.VideoCapture(2)

    if webcam0.isOpened() and webcam1.isOpened(): # try to get the first frame
      rval0, frame0 = webcam0.read()
      rval1, frame1 = webcam1.read()
    else:
      rval0 = False
    
    i = 0
    while rval0 and i < MAX_IMAGES:
        i += 1
        do_stereo( frame0, frame1, 'out%i.ply' % i)

        # Tee up next frame
        rval0, frame0 = webcam0.read()
        rval1, frame1 = webcam1.read()
        
        key = cv2.waitKey(20)
        if key in [27, ord('Q'), ord('q')]: # exit on ESC
          break
    
    print "DONE!"