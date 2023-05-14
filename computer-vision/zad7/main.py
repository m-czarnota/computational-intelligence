import os
import numpy as np
import cv2

IMAGES_DIR = './images'
RESULTS_DIR = './results'


if __name__ == '__main__':
    images_dirs = [folder.path.replace("\\", "/") for folder in os.scandir(IMAGES_DIR)]

    for image_dir in images_dirs:
        folder_name = image_dir.split('/')[-1]
        result_folder = f'{RESULTS_DIR}/{folder_name}'

        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # read images
        img_left = cv2.imread(f'{image_dir}/img-left.png', 0)
        img_right = cv2.imread(f'{image_dir}/img-right.png', 0)
        img_disp = cv2.imread(f'{image_dir}/img-disp.png')

        # params of algorithm StereoSGBM
        window_size = 5
        min_disp = 0
        num_disp = 16 * 5
        P1 = 8 * 3 * window_size ** 2
        P2 = 32 * 3 * window_size ** 2

        # depth map calculation
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=P1,
            P2=P2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        disparity_map = stereo.compute(img_left, img_right)

        # normalization of the depth map to the range of 0-255
        disparity_map_norm = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                           dtype=cv2.CV_8U)
        # cv2.imshow("Depth detection", disparity_map_norm)
        # cv2.waitKey()
        cv2.imwrite(f"{result_folder}/depth-detection.png", disparity_map_norm)

        # Gaussian filtration
        filtered_map_gauss = cv2.GaussianBlur(disparity_map_norm, (5, 5), 0)
        # cv2.imshow("Filtered with Gaussian method", filtered_map_gauss)
        # cv2.waitKey()
        cv2.imwrite(f"{result_folder}/filtered-gaussian.png", filtered_map_gauss)

        # median filtration
        filtered_map_median = cv2.medianBlur(disparity_map_norm, 5)
        # cv2.imshow("Filtered with Median method", filtered_map_median)
        # cv2.waitKey()
        cv2.imwrite(f"{result_folder}/filtered-median.png", filtered_map_median)

        # Weighted Least Squares filtration
        sigma = 2
        lamda = 8000.0
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
        wls_filter.setLambda(lamda)
        wls_filter.setSigmaColor(sigma)
        filtered_map_wls = wls_filter.filter(disparity_map_norm, img_right, None, img_left)
        # cv2.imshow("Filtered with WLS method", filtered_map_wls)
        # cv2.waitKey()
        cv2.imwrite(f"{result_folder}/filtered-wls.png", filtered_map_wls)

        disparity_map_norm_3d_gauss = cv2.merge((filtered_map_gauss, filtered_map_gauss, filtered_map_gauss))
        mse_gauss = np.mean((img_disp - disparity_map_norm_3d_gauss) ** 2)

        disparity_map_norm_3d_median = cv2.merge((filtered_map_median, filtered_map_median, filtered_map_median))
        mse_median = np.mean((img_disp - disparity_map_norm_3d_median) ** 2)

        disparity_map_norm_3d_wls = cv2.merge((filtered_map_wls, filtered_map_wls, filtered_map_wls))
        mse_wls = np.mean((img_disp - disparity_map_norm_3d_wls) ** 2)

        with open(f'{result_folder}/mse.txt', 'w') as f:
            f.write(f'gauss: {mse_gauss}\n')
            f.write(f'median: {mse_median}\n')
            f.write(f'wls: {mse_wls}\n')
