import extract_ocr_region


datasets_root_path = './datasets/ICPR_text_train_part2_20180313/'
labels_out_path = './labels_out.txt'
img_extracted_path = 'images_extracted'

extract_ocr_region.extract_ocr_region(
    datasets_root_path,
    img_extracted_path,
    labels_out_path
)
