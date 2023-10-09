# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import cv2
import copy
import json
import base64
import numpy as np
from cog import BasePredictor, Input, Path, File
from segment_anything import SamPredictor, sam_model_registry


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.predictor = SamPredictor(self.model)

    # returns an array of paths of segmented images
    def predict(
        self,
        image: Path = Input(description="URL of the image to be segmented", default=None),
        image_file_b64: str = Input(description="Image encoded as a base64 string", default=None),
        positive_prompts: str = Input(
            description="Stringified array of [x, y] values representing points",
            default="[]"
        ),
        negative_prompts: str = Input(
            description="Stringified array of [x, y] values representing points",
            default="[]"
        )
    ) -> list[Path]:
        
        if image_file_b64:
            im_bytes = base64.b64decode(image_file_b64)
            im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
            self.image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        else:
            self.image = cv2.imread(str(image))


        self.predictor.set_image(self.image)

        input_points = []
        input_labels = []

        positive_prompts = json.loads(positive_prompts)
        negative_prompts = json.loads(negative_prompts)

        # add positive promtps and 1 in the same index in labels
        for point in positive_prompts:
            input_points.append(point)
            input_labels.append(1)

        # add negative prompts ad 0 in the same index in labels
        for point in negative_prompts:
            input_points.append(point)
            input_labels.append(0)
        
        input_points = np.array(input_points)

        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )

        segmented_images = []

        # reference for "cutting" out segmented images using mask -
        # https://github.com/facebookresearch/segment-anything/issues/221#issuecomment-1614280903

        for idx, mask in enumerate(masks):
            segmented_image_path = f"/tmp/{idx}.png"

            # converting false parts to white color
            image_copy = copy.deepcopy(self.image)
            image_copy[mask==False] = [255, 255, 255]

            # writing segmented image to disk
            cv2.imwrite(segmented_image_path, image_copy)

            segmented_images.append(Path(segmented_image_path))

        return segmented_images
