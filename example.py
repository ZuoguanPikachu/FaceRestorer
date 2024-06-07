from face_restorer import FaceRestorer
import cv2


if __name__ == '__main__':
    face_restorer = FaceRestorer("GFPGAN", bg_up_sample=True)

    input_img = cv2.imread('test.jpg')
    output_img = face_restorer.restore(input_img, out_scale=2)

    cv2.imwrite('out.jpg', output_img)
