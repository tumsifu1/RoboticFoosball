import cv2

vid = cv2.VideoCapture('../data/test.mp4')

img_counter = 0
read, img = vid.read()
while read:
    cv2.imwrite(f"./test/test_img_{img_counter}.jpg", img)

    h, w, _ = img.shape
    segment_counter = 0
    for i in range(1, 9):
        for j in range(1, 4):
            cv2.imwrite(f"../data/images_part{img_counter//4}/img_{img_counter}_{segment_counter}.jpg", img[(j - 1) * 432:(j * 432), (i - 1) * 288:(i * 288)])
            segment_counter += 1
    read, img = vid.read()
    img_counter += 1
