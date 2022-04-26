import numpy as np


class OdsSequence:
    """Single data for network"""
    def __init__(self, scene_id, image_id, baseline, tgt_pose):
        """
        Args:
            scene_id: String for identifying scene and position
            image_id: [3, 1] image id for src, ref, tgt images
            baseline: Camera intrinsic
            tgt_pose: [3, 1] translation vector for target image
        """
        self.scene_id = scene_id
        self.image_id = image_id
        self.baseline = baseline
        self.src_pose = np.identity(4)
        self.ref_pose = np.identity(4)
        self.tgt_pose = np.array(tgt_pose, dtype=np.float)
        self.intrinsic = np.array([[baseline, 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float)
