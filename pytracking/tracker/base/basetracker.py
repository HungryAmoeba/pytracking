from collections import OrderedDict

class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params
        self.visdom = None


    def predicts_segmentation_mask(self):
        return False


    def initialize(self, image, info):
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError


    def track(self, image, info = None):
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError


    def visdom_draw_tracking(self, image, box, segmentation=None):
        if isinstance(box, OrderedDict):
            box = [v for k, v in box.items()]
        else:
            box = box
        if segmentation is None:
            self.visdom.register(tuple(image + box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register(tuple(image + box + segmentation), 'Tracking', 1, 'Tracking')
