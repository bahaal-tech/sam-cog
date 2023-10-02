# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import cv2
import json
import copy
import base64
import numpy as np
from io import BytesIO
from cog import BasePredictor, Input, Path
from segment_anything import SamPredictor, sam_model_registry


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.predictor = SamPredictor(self.model)

    # returns a base64 encoded image
    def predict(
        self,
        image_b64: str = Input(
            description="Image as b64 encoded string"
        ),
        positive_prompts: str = Input(
            description="Stringified array of [x, y] values representing points",
            default="[]"
        ),
        negative_prompts: str = Input(
            description="Stringified array of [x, y] values representing points",
            default="[]"
        )
    ) -> str:
        
        # decode the base 64 encoded image and load it using cv2
        image_bytes = base64.b64decode(image_b64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)

        self.predictor.set_image(image)

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

        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )

        segmented_images = []

        # reference for "cutting" out segmented images -
        # https://github.com/facebookresearch/segment-anything/issues/221#issuecomment-1614280903
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for mask in masks:
            # creating a deepcopy of the image
            image_copy = copy.deepcopy(image)
            # converting false parts to white color
            image_copy[mask==False] = [255, 255, 255]

            # add image to segmented images by converting them to b64
            segmented_images.append(
                base64.b64encode(
                    image_copy.tobytes()
                )
            )

        return json.dumps(segmented_images)
