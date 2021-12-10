import numpy as np
import PIL
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from numpy.lib.twodim_base import eye
import scipy
import scipy.ndimage
import dlib
from pathlib import Path

def draw_points(image, lm, r):
    draw = ImageDraw.Draw(image)
    leftUpPoint = (lm[0] - r, lm[1] - r)
    rightDownPoint = (lm[0] + r, lm[1] + r)
    twoPointList = [leftUpPoint, rightDownPoint]
    draw.ellipse(twoPointList, fill=(255, 0, 0, 255))
    del draw
    return image


def get_landmark(img, predictor, data_type):
    if data_type == 'face':
        detector = dlib.get_frontal_face_detector()
    else:
        detector = dlib.fhog_object_detector('weights/detector_catface.svm')
    dets = detector(img, 1)
    shapes = [predictor(img, d) for k, d in enumerate(dets)]
    lms = [np.array([[tt.x, tt.y] for tt in shape.parts()]) for shape in shapes]
    return lms


def align_face(img, predictor, data_type):
    lms = get_landmark(img, predictor, data_type)
    imgs = []
    if(data_type == 'face'):
        for lm in lms:
            lm_chin = lm[0: 17]  # left-right
            lm_eyebrow_left = lm[17: 22]  # left-right
            lm_eyebrow_right = lm[22: 27]  # left-right
            lm_nose = lm[27: 31]  # top-down
            lm_nostrils = lm[31: 36]  # top-down
            lm_eye_left = lm[36: 42]  # left-clockwise
            lm_eye_right = lm[42: 48]  # left-clockwise
            lm_mouth_outer = lm[48: 60]  # left-clockwise
            lm_mouth_inner = lm[60: 68]  # left-clockwise

            # Calculate auxiliary vectors.
            eye_left = np.mean(lm_eye_left, axis=0)
            eye_right = np.mean(lm_eye_right, axis=0)
            eye_avg = (eye_left + eye_right) * 0.5
            eye_to_eye = eye_right - eye_left
            mouth_left = lm_mouth_outer[0]
            mouth_right = lm_mouth_outer[6]
            mouth_avg = (mouth_left + mouth_right) * 0.5
            eye_to_mouth = mouth_avg - eye_avg
    else:
        for lm in lms:
            lm_chin = lm[0]
            eye_left = lm[1]
            lm_left_of_left_ear = lm[2]
            lm_left_of_right_ear = lm[3]
            lm_nose = lm[4]
            eye_right = lm[5]
            lm_right_of_left_ear = lm[6]
            lm_right_of_right_ear = lm[7]
            eye_to_eye = eye_right - eye_left
            eye_avg = (eye_left + eye_right) * 0.5
            mouth_avg = (lm_nose + lm_chin) * 0.5
            eye_to_mouth = mouth_avg - eye_avg


        # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = Image.fromarray(img)
    # img = draw_points(img, eye_left, 15)
    # img = draw_points(img, eye_right, 15)
    # img = draw_points(img, mouth_avg, 15)
    # img = draw_points(img, lm_chin, 15)
    # img = draw_points(img, lm_nose, 15)

    output_size = 1024 #32*32
    transform_size = 4096 #64*64
    enable_padding = False #True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
            max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                        PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    imgs.append(img)
    return imgs