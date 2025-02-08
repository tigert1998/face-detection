import sys

sys.path.append("third_party/AdaFace")

import os
import pickle
from typing import List, Tuple

from tqdm import tqdm
import PIL.Image
import numpy as np
from inference import load_pretrained_model, to_input
from sklearn.neighbors import KNeighborsClassifier


class FaceFindResult:
    paths: List[str]
    dists: List[float]
    bboxes: List[List[float]]

    def __init__(self):
        self.paths = []
        self.dists = []
        self.bboxes = []


class AdaFaceFinder:
    def get_aligned_faces(self, image_path, rgb_pil_image=None):
        from face_alignment import mtcnn

        mtcnn_model = mtcnn.MTCNN(device=self.device, crop_size=(112, 112))
        if rgb_pil_image is None:
            img = PIL.Image.open(image_path).convert("RGB")
        else:
            assert isinstance(
                rgb_pil_image, PIL.Image.Image
            ), "Face alignment module requires PIL image or path to the image"
            img = rgb_pil_image
        # find face
        try:
            bboxes, faces = mtcnn_model.align_multi(img, limit=4)
        except Exception as e:
            self.logger.error(f"Face detection Failed due to error: {e}")
            faces = None

        return faces, bboxes

    def __init__(self, db_path, logger, device="cuda:0"):
        img_paths = [
            os.path.join(db_path, filename)
            for filename in os.listdir(db_path)
            if os.path.splitext(filename)[-1].lower() in [".jpg", ".png"]
        ]

        self.device = device
        self.logger = logger
        self.model = load_pretrained_model("ir_50")
        self.model.to(self.device)
        pkl_path = os.path.join(db_path, "adaface.pkl")
        is_pkl_changed = True
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                self.dic = pickle.load(f)
            is_pkl_changed = set(self.dic.keys()) != set(img_paths)

        if is_pkl_changed:
            self.dic = {}
            for img_path in tqdm(img_paths):
                features, _ = self._extract_features(img_path)
                self.dic[img_path] = features[0]
            with open(pkl_path, "wb") as f:
                pickle.dump(self.dic, f)

        self.knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        x = np.concatenate(list(self.dic.values()), axis=0)
        self.img_paths = list(self.dic.keys())
        self.knn.fit(x, list(range(x.shape[0])))

    def _extract_features(self, img):
        if isinstance(img, PIL.Image.Image):
            aligned_rgb_imgs, bboxes = self.get_aligned_faces(
                image_path=None, rgb_pil_image=img
            )
        else:
            aligned_rgb_imgs, bboxes = self.get_aligned_faces(img)
        features = []
        for aligned_rgb_img in aligned_rgb_imgs:
            bgr_input = to_input(aligned_rgb_img)
            feature, _ = self.model(bgr_input.to(self.device))
            feature = feature.detach().cpu().numpy()
            features.append(feature)
        return features, bboxes

    def face_find(self, rgb_frame, distance_threshold=None):
        res = FaceFindResult()
        failed_res = FaceFindResult()
        img = PIL.Image.fromarray(rgb_frame.astype("uint8"), "RGB")
        features, bboxes = self._extract_features(img)
        if len(features) == 0:
            return res, failed_res

        dists, idxs = self.knn.kneighbors(np.concatenate(features, axis=0))

        for idx, dist, bbox in zip(idxs, dists, bboxes):
            if distance_threshold is not None and dist > distance_threshold:
                failed_res.paths.append(self.img_paths[idx[0]])
                failed_res.dists.append(dist[0])
                failed_res.bboxes.append(bbox)
            else:
                res.paths.append(self.img_paths[idx[0]])
                res.dists.append(dist[0])
                res.bboxes.append(bbox)
        return res, failed_res
