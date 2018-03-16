import cv2
import numpy as np
import os
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

                # 矫正ocr区域的角度
                # centroid_x, centroid_y = (np.mean(point_axis[0][:, 0]),
                #                           np.mean(point_axis[0][:, 1]))

                # Get diagonal point pairs
                idx_dis_max = np.argmax(
                    np.sum((point_axis[0] - point_axis_1)**2, axis=1)
                )

                point_axis_pair_1 = [point_axis[0][0],
                                     point_axis[0][idx_dis_max]]
                pair_2_idx = list(set(range(1, 4)) - {idx_dis_max})
                point_axis_pair_2 = [point_axis[0][pair_2_idx[0]],
                                     point_axis[0][pair_2_idx[1]]]
                # correcting_angle = (
                #     np.arctan((point_axis_1[1] - centroid_y) /
                #               (point_axis_1[0] - centroid_x)) +
                #     np.arctan((point_axis_2[1] - centroid_y) /
                #               (point_axis_2[0] - centroid_x)) +
                #     np.arctan((point_axis_3[1] - centroid_y) /
                #               (point_axis_3[0] - centroid_x)) +
                #     np.arctan((point_axis_4[1] - centroid_y) /
                #               (point_axis_4[0] - centroid_x))
                # ) / 4
                if not ((point_axis_pair_1[1][0] - point_axis_pair_1[0][0]) and
                        (point_axis_pair_2[1][0] - point_axis_pair_2[0][0])):
                        correcting_angle = 0
                else:
                    correcting_angle = ((point_axis_pair_1[1][1] -
                                        point_axis_pair_1[0][1]) /
                                        (point_axis_pair_1[1][0] -
                                        point_axis_pair_1[0][0]) +
                                        (point_axis_pair_2[1][1] -
                                        point_axis_pair_2[0][1]) /
                                        (point_axis_pair_2[1][0] -
                                        point_axis_pair_2[0][0])) / 2
                correcting_angle = np.rad2deg(correcting_angle)
                print("correcting_angle:", correcting_angle)
                text_rot = get_img_rot_broa.get_img_rot_broa(text,
                                                             correcting_angle)
                # text_rot = corr_ang_by_radon.corr_ang_by_radon(text)[0]
                text_cropped = elim_pure_borders.elim_pure_borders(text_rot)
                cv2.imwrite(os.path.join(img_extracted_path,
                                         str(img_idx)+'.jpg'),
                            text_cropped)
                # import matplotlib.pyplot as plt
                # fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 4))
                # ax0.imshow(cv2.cvtColor(text, cv2.COLOR_BGR2RGB))
                # ax0.set_title('text')
                # ax1.imshow(cv2.cvtColor(text_rot, cv2.COLOR_BGR2RGB))
                # ax1.set_title('text_rot')
                # ax2.imshow(cv2.cvtColor(text_cropped, cv2.COLOR_BGR2RGB))
                # ax2.set_title('text_cropped')
                # plt.show()

                # thresholding
                # blur = cv2.cvtColor(cv2.GaussianBlur(text, (5, 5), 0),
                #                     cv2.COLOR_BGR2GRAY)
                # thr = cv2.threshold(blur, 127, 255,
                #                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                # plt.figure(1)
                # plt.imshow(blur, cmap='gray')
                # plt.figure(2)
                # plt.imshow(thr, cmap='gray')
                # plt.title('thr')
                # plt.show()
                img_idx += 1

                # Test by tesseract
                # import pytesseract
                # from PIL import Image
                # cv2.imwrite('./t.png', thr)
                # res = pytesseract.image_to_string(
                #     Image.open('./t.png'),
                #     lang='chi_sim+eng'
                # )
                # print('Prediction by tesseract is:', res)
                # Use NN to replace tesseract

                # single labels
                with open('./labels_out.txt', 'a+') as fout:
                    fout.write(label + '\n')
