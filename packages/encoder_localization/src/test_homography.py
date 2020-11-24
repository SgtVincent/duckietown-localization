import cv2
import numpy as np
import os 

# [ fx 0  cx ] [ R00  R01  TX ]    [ H00 H01 H02 ]
# [  0 fy cy ] [ R10  R11  TY ] =  [ H10 H11 H12 ]
# [  0  0  1 ] [ R20  R21  TZ ] =  [ H20 H21 H22 ]
#

def homography2transformation(H, K):
    # @H: homography, 3x3 matrix 
    # @K: intrinsic matrix, 3x3 matrix 
    # apply inv(K)
    fx = K[0,0]; fy = K[1,1]; cx = K[0,2]; cy = K[1,2]
    K_inv = np.array(
        [[1/fx,  0,      -cx/fx],
        [0,     1/fy,   -cy/fy],
        [0,     0,      1     ]]
    ) 
    Rt = K_inv @ H

    # normalize Rt matrix so that R columns have unit length 
    norm_1 = np.linalg.norm(Rt[:, 0])
    norm_2 = np.linalg.norm(Rt[:, 1])
    norm = np.sqrt(norm_1 * norm_2) # get the average norm of first two columns
    # WARNING: check if camera is above baseline !!!
    
    norm_Rt = Rt/norm
    
    r1 = norm_Rt[:, 0]; r2 = norm_Rt[:, 1]
    r3 = np.cross(r1, r2)
    R = np.stack((r1, r2, r3), axis=1)
    
    # no idea why, this SVD will ruin the whole transformation!!! 
    print("R (before polar decomposition):\n",R,"\ndet(R): ", np.linalg.det(R))
    u, s, vh = np.linalg.svd(R)
    R = u@vh
    print("R (after polar decomposition):\n", R, "\ndet(R): ", np.linalg.det(R))

    T = np.zeros((4,4))
    T[0:3, 0:3] = R
    T[0:3, 3] = norm_Rt[:, 2]
    T[3,3] = 1.0

    return T

if __name__ == "__main__":

    H = [
        4.901931583237535e-06,
        -0.0002441938726410032,
        -0.13815159874374158,
        0.0008089507472746776,
        8.25475002676469e-05,
        -0.26026178771790215,
        -6.421924308602405e-05,
        -0.006135407471049536,
        0.9999999999999999]
    K = [319.2923244124961, 0.0, 305.3834748901065, 0.0, 316.6905964742383, 239.5466696838447, 0.0, 0.0, 1.0]
    
    # scale = 1
    # test_planar_points = [np.array([scale,scale, 1.0]),
    #                 np.array([-scale,scale, 1.0]),
    #                 np.array([-scale,-scale, 1.0]),
    #                 np.array([scale,-scale, 1.0])]
    # test_3d_points = [np.array([scale,scale,0, 1.0]),
    #                 np.array([-scale,scale,0, 1.0]),
    #                 np.array([-scale,-scale,0, 1.0]),
    #                 np.array([scale,-scale,0, 1.0])]

    test_planar_points = [np.array([0.315,0.093, 1.0]),
                    np.array([0.315,-0.093, 1.0]),
                    np.array([0.191,-0.093, 1.0]),
                    np.array([0.191,0.093, 1.0])]

    test_3d_points = [np.array([0.315,0.093,0, 1.0]),
                    np.array([0.315,-0.093,0, 1.0]),
                    np.array([0.191,-0.093,0, 1.0]),
                    np.array([0.191,0.093,0, 1.0])]


    num_pts = len(test_planar_points)

    H_mat = np.array(H).reshape((3,3))
    H_mat = np.linalg.inv(H_mat)
    K_mat = np.array(K).reshape((3,3))

    true_projection = [H_mat @ pt for pt in test_planar_points]    
    true_projection = [pixel[:2]/pixel[2] for pixel in true_projection]

    # _, rotations, translations, normals =cv2.decomposeHomographyMat(H_mat, K_mat)
    # # pick the first one
    # rotation = rotations[0]
    # translation = translations[0]
    
    # T1 = np.zeros((4,4))
    # T1[:3,:3] = np.array(rotation).reshape((3,3))
    # T1[:3, 3] = np.array(translation).reshape(3)
    # T1[ 3, 3] = 1.0
    # T1_projection = [np.concatenate((K_mat, np.zeros((3,1))), axis=1) @ T1 @ pt for pt in test_3d_points]
    # T1_projection = [pixel/pixel[2] for pixel in T1_projection]
    # T1_repro_errors = [ np.linalg.norm(true_projection[i] - T1_projection[i])
    #     for i in range(num_pts)
    # ]
    # print("T1_repro_errors", T1_repro_errors)

    T2 =homography2transformation(H_mat, K_mat)
    T2_projection = [np.concatenate((K_mat, np.zeros((3,1))), axis=1) @ T2 @ pt for pt in test_3d_points]
    T2_projection = [pixel[:2]/pixel[2] for pixel in T2_projection]
    T2_repro_errors = [ np.linalg.norm(true_projection[i] - T2_projection[i])
        for i in range(num_pts)
    ]
    print("T2_repro_errors", T2_repro_errors)

    visualize = True

    if visualize:
        img_ground_truth = cv2.imread("/home/chenjunting/Pictures/checkerboard.jpg")
        img_ground_truth = cv2.resize(img_ground_truth, (640, 480))
        # img_T1 = cv2.imread("/home/chenjunting/Pictures/checkerboard.jpg")
        # img_T1 = cv2.resize(img_T1, (640, 480))
        img_T2 = cv2.imread("/home/chenjunting/Pictures/checkerboard.jpg")
        img_T2 = cv2.resize(img_T2, (640, 480))
        num = len(test_planar_points)
        for idx in range(num):
            cv2.line(
                img_ground_truth, tuple(true_projection[idx%num].astype(int)), tuple(true_projection[(idx+1)%num].astype(int)), (0, 255, 0))

            cv2.line(
                img_T2, tuple(T2_projection[idx%num].astype(int)), tuple(T2_projection[(idx+1)%num].astype(int)), (0, 255, 0))

        cv2.imshow('img_ground_truth', img_ground_truth)
        cv2.imshow('img_T2', img_T2)
        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
    pass