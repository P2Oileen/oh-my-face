import dlib
import argparse
#from bicubic import BicubicDownSample
import torchvision
import cv2
from shape_predictor import align_face


def align_func(img, data_type, weight_dir):
    if data_type == 'face':
        predictor_face = dlib.shape_predictor(weight_dir + '/shape_predictor_68_face_landmarks.dat')
        faces = align_face(img, predictor_face, data_type)
    else:
        predictor_cat = dlib.shape_predictor(weight_dir + '/shape_predictor_cat.dat')
        faces = align_face(img, predictor_cat, data_type)
    for i,face in enumerate(faces):
        if i > 0:
            break
        face_tensor = torchvision.transforms.ToTensor()(face)
        face = face_tensor.numpy()
        # print(face.shape) (3, 1024, 1024) 0~1 bgr
        return face

if __name__ == "__main__":
    img = cv2.imread("cat.jpg")
    img = align_func(img, 'cat')
    cv2.imwrite("cat_aligned.jpg", img.transpose(1, 2, 0) * 255)

