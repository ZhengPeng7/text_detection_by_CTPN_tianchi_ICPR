OCR_TIANCHI_ICPR

[The Alibaba TIANCHI OCR competition](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.34b6d780oYcxs2&raceId=231650)

[CTPN reference](https://github.com/YCG09/chinese_ocr)

### Usage: 

- Download well-trained model from [here](https://www.floydhub.com/api/v1/resources/D9J7q8XFJNWdG4SjoZT6ag?content=true&download=true).

- Run the text_detector.ipynb in the single_ctpn.

### Samples:

![sample_1](./samples/sample_1.png)![](./samples/sample_2.png)

### Results:

$ Anchor: (x1, y1, x2, y2, x3, y3, x4, y4) $ for each text region will be stored in the results folder, where the names of txt files are the corresponding image files.