"""SAM2 point prompt segmentation module.

Dramatically accelerate multi-point inference with feature caching using Ultralytics integrated SAM2Predictor.
SAM2 uses official weights without fine-tuning.

Working principle:
  1. Call set_image() to encode each image once (about 2-3 seconds)
  2. Call predictor() for each point prompt and only do mask decoding (about 0.05-0.1 seconds)
  3. Generate masks for multiple points respectively and then take the union
"""
import numpy as np
import torch


class SAM2Segmenter:
    """SAM2 segmenter: receives point coordinate prompts and outputs binary segmentation masks.

    Internally uses SAM2Predictor and caches image features to avoid repeated encoding.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """Load the SAM2 model.

        Args:
            model_path: SAM2 weight file path (such as weights/sam2_t.pt).
            device: Inference device, \"cpu\" or \"cuda\".
        """
        from ultralytics.models.sam import SAM2Predictor

        self.device = device
        self.predictor = SAM2Predictor(overrides={
            "model": model_path,
            "task": "segment",
            "mode": "predict",
            "save": False,
            "verbose": False,
            "imgsz": 1024,
        })
        # Preloading the model (first call to setup_model)
        self.predictor.set_image(np.zeros((64, 64, 3), dtype=np.uint8))
        print(f"[SAM2] model loaded: {model_path}")

    def _extract_mask(self, results, h: int, w: int) -> np.ndarray:
        'Extract binary masks from prediction results.'
        if not results or results[0].masks is None:
            return np.zeros((h, w), dtype=np.uint8)
        mask_data = results[0].masks.data
        if isinstance(mask_data, torch.Tensor):
            if mask_data.shape[0] == 0:
                return np.zeros((h, w), dtype=np.uint8)
            mask = mask_data[0].cpu().numpy()
        else:
            if len(mask_data) == 0:
                return np.zeros((h, w), dtype=np.uint8)
            mask = np.asarray(mask_data[0])
        return (mask > 0).astype(np.uint8)

    def segment_single_point(self, image: np.ndarray, point: list[float]) -> np.ndarray:
        """Segment a single point tip of a single image (set_image needs to be called first).

        Args:
            image: HxWx3 BGR image (OpenCV format).
            point: [x, y] pixel coordinates.

        Returns:
            HxW uint8 binary mask (0=background, 1=foreground).
        """
        h, w = image.shape[:2]
        results = list(self.predictor(
            image,
            points=[[int(point[0]), int(point[1])]],
            labels=[1],
            verbose=False,
        ))
        return self._extract_mask(results, h, w)

    def segment_with_points(self, image: np.ndarray,
                            points: list[list[float]]) -> np.ndarray:
        """Divide multiple point prompts separately and then take the union.

        Encode the image once (set_image) first, and then quickly reason for each point.
        If the point list is empty, an all-zero mask is returned.

        Args:
            image: HxWx3 BGR image.
            points: [[x1,y1], [x2,y2], ...] Pixel coordinate list.

        Returns:
            HxW uint8 Merged binary mask.
        """
        h, w = image.shape[:2]
        if not points:
            return np.zeros((h, w), dtype=np.uint8)

        # Encode image features (do this only once per image)
        self.predictor.set_image(image)

        combined = np.zeros((h, w), dtype=np.uint8)
        for pt in points:
            mask = self.segment_single_point(image, pt)
            combined = np.maximum(combined, mask)
        return combined

    def segment_with_boxes(self, image: np.ndarray,
                           boxes: np.ndarray) -> np.ndarray:
        """According to the YOLO detection frame, the center point of the frame is used as the prompt for segmentation.

        Args:
            image: HxWx3 BGR image.
            boxes: Nx4 array, each row contains [x1, y1, x2, y2] pixel coordinates.

        Returns:
            HxW uint8 Merged binary mask.
        """
        if len(boxes) == 0:
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        # Calculate the center point of each detection frame
        points = []
        for box in boxes:
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            points.append([cx, cy])

        return self.segment_with_points(image, points)

    def segment_with_boxes_labeled(self, image: np.ndarray,
                                   boxes: np.ndarray,
                                   box_classes: np.ndarray,
                                   confs: np.ndarray | None = None,
                                   num_classes: int = 5) -> np.ndarray:
        """Generate class-indexed segmentation masks based on classed YOLO detection boxes.

        SAM2 is only responsible for cutting out the shape of each target, and the category uses the category number given by YOLO;
        When the masks of multiple boxes overlap, the box with higher confidence will occupy the pixels first (in the order of incoming when there is no confidence).

        Args:
            image: HxWx3 BGR image.
            boxes: Nx4 array, each row contains [x1, y1, x2, y2] pixel coordinates.
            box_classes: N array, YOLO class number (0..num_classes-2, excluding background).
            confs: N array, YOLO confidence, used for overlapping region resolution (optional).
            num_classes: The total number of classes (including background), the output mask value is 0..num_classes-1.

        Returns:
            HxW uint8 class index mask (0=background, 1..num_classes-1=foreground classes).
        """
        h, w = image.shape[:2]
        if len(boxes) == 0:
            return np.zeros((h, w), dtype=np.uint8)

        # Confidence is processed from high to low, ensuring that overlapping pixels are determined by high-confidence boxes
        order = np.arange(len(boxes))
        if confs is not None:
            order = order[np.argsort(-np.asarray(confs))]

        # Encode image features (do this only once per image)
        self.predictor.set_image(image)

        pred = np.zeros((h, w), dtype=np.uint8)
        for idx in order:
            box = boxes[idx]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            mask = self.segment_single_point(image, [cx, cy])
            cls_idx = int(box_classes[idx]) + 1  # 0=background, 1.. = foreground category
            # Only fill in pixels not occupied by higher confidence boxes
            pred = np.where((pred == 0) & (mask > 0), cls_idx, pred)
        return pred
