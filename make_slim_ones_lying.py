import cv2
import numpy as np
import os
import shutil
import elim_pure_borders
# import corr_ang_by_radon
# import corr_ang_by_minAreaRect
import get_img_rot_broa


def extract_ocr_region(
    datasets_root_path,
    img_extracted_path,
    labels_out_path
):
    """
    Description:
            To extract ocr regions from the raw imgs,
        and record corresponding label.

    params:
        path_with_raw_imgs_and_labels,
        path to save imgs extracted,
        path to save labels by lines
    return: None
    """
    if os.path.exists(labels_out_path):
        os.remove(labels_out_path)
    if os.path.exists(img_extracted_path):
        shutil.rmtree(img_extracted_path)
        os.mkdir(img_extracted_path)

    # get paths
    label_axis_paths = sorted(
        os.listdir(os.path.join(datasets_root_path, 'txt_9000'))
    )
    for i in range(len(label_axis_paths)):
        label_axis_paths[i] = os.path.join(
            datasets_root_path, 'txt_9000', label_axis_paths[i]
        )

    image_paths = sorted(
        os.listdir(os.path.join(datasets_root_path, 'image_9000'))
    )
    for i in range(len(image_paths)):
        image_paths[i] = os.path.join(
            datasets_root_path, 'image_9000', image_paths[i]
        )

    img_idx = 0
    for image_path_idx in range(len(image_paths))[:100]:
        img = cv2.imread(image_paths[image_path_idx])
        if img is None:
            print("Error img " + image_paths[image_path_idx])
            continue
        print('image_path:', image_paths[image_path_idx])
        lab_ax_path = label_axis_paths[image_path_idx]
        with open(lab_ax_path, 'r') as fin:
            for line in fin.readlines():
                (axis, label) = (line.strip().split(',')[:-1],
                                 line.strip().split(',')[-1])
                if label == '###':
                    continue
                point_axis_1 = [round(float(axis[0])), round(float(axis[1]))]
                point_axis_2 = [round(float(axis[2])), round(float(axis[3]))]
                point_axis_3 = [round(float(axis[4])), round(float(axis[5]))]
                point_axis_4 = [round(float(axis[6])), round(float(axis[7]))]
                canvas = np.zeros_like(img)
                point_axis = np.array([[point_axis_1, point_axis_2,
                                        point_axis_3, point_axis_4]])
                mask = cv2.fillPoly(
                    canvas,
                    point_axis,
                    (255, 255, 255)
                )
                text = cv2.bitwise_and(img, mask)
                # 怎样才能保证 全为字白背景黑 或 全为字黑背景白呢?

                # Get diagonal point
                idx_dis_max = np.argmax(
                    np.sum((point_axis[0] - point_axis_1)**2, axis=1)
                )

                point_axis_diag_pair = [point_axis[0][0],
                                        point_axis[0][idx_dis_max]]
                diagonal_line_square_len = np.sum(np.square((
                    point_axis_diag_pair[0] - point_axis_diag_pair[1])))
                # get left side 2 points
                left_two_points = sorted(point_axis[0], key=lambda x: x[1])[:2]
                if diagonal_line_square_len < 666:
                    correcting_angle = 0
                else:
                    # 矫正ocr区域的角度, depending on the the slope
                    # between 4 vertices and the centroid.
                    centroid_x, centroid_y = (np.mean(point_axis[0][:, 0]),
                                              np.mean(point_axis[0][:, 1]))
                    vertical_existence = False
                    for idx_prev_zero in range(4):
                        if not point_axis[0][idx_prev_zero][0] - centroid_x:
                            vertical_existence = True
                            break
                    if not vertical_existence:
                        correcting_angle = (
                            np.arctan((point_axis_1[1] - centroid_y) /
                                      (point_axis_1[0] - centroid_x)) +
                            np.arctan((point_axis_2[1] - centroid_y) /
                                      (point_axis_2[0] - centroid_x)) +
                            np.arctan((point_axis_3[1] - centroid_y) /
                                      (point_axis_3[0] - centroid_x)) +
                            np.arctan((point_axis_4[1] - centroid_y) /
                                      (point_axis_4[0] - centroid_x))
                        ) / 4
                        correcting_angle = np.rad2deg(correcting_angle)
                        if not (left_two_points[0][0] -
                                left_two_points[1][0]):
                            correcting_angle = 90
                        elif (abs(left_two_points[0][1] -
                                  left_two_points[1][1]) /
                              abs(left_two_points[0][0] -
                                  left_two_points[1][0])) > 1.23 * np.pi:
                            correcting_angle = 90 - correcting_angle
                    elif (np.abs(point_axis_diag_pair[0][1] -
                                 point_axis_diag_pair[1][1]) >
                          np.abs(point_axis_diag_pair[0][0] -
                                 point_axis_diag_pair[1][0])):
                        # the region should lie but for standing.
                        correcting_angle = 90
                    else:
                        correcting_angle = 0
                # print("correcting_angle:", correcting_angle)
                text_rot = get_img_rot_broa.get_img_rot_broa(text,
                                                             correcting_angle)
                # text_rot = corr_ang_by_radon.corr_ang_by_radon(text)[0]
                text_cropped = elim_pure_borders.elim_pure_borders(text_rot)
                if text_cropped is None:
                    print("text_cropped:", text_cropped)
                    print("img_idx:", img_idx)
                    continue
                cv2.imwrite(os.path.join(img_extracted_path,
                                         str(img_idx)+'.jpg'),
                            text_cropped)

                # show figures
                # import matplotlib.pyplot as plt
                # fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 4))
                # ax0.imshow(cv2.cvtColor(text, cv2.COLOR_BGR2RGB))
                # ax0.set_title('text')
                # ax1.imshow(cv2.cvtColor(text_rot, cv2.COLOR_BGR2RGB))
                # ax1.set_title('text_rot')
                # ax2.imshow(cv2.cvtColor(text_cropped, cv2.COLOR_BGR2RGB))
                # ax2.set_title('text_cropped')
                # plt.show()
                if cv2.imread(
                    os.path.join(img_extracted_path, str(img_idx)+'.jpg')
                ) is None:
                    os.remove(
                        os.path.join(img_extracted_path, str(img_idx)+'.jpg')
                    )
                    continue

                img_idx += 1

                # single labels
                with open('./labels_out.txt', 'a+') as fout:
                    fout.write(label + '\n')
