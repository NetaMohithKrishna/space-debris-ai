from models.heatmap_model_v5_hr import HeatmapDetectorV5HR


class HeatmapDetectorV5Temporal(HeatmapDetectorV5HR):
    """
    V5-HR with 3-channel temporal input:

        channel 0 = current frame
        channel 1 = |current - previous|
        channel 2 = |next - current|
    """

    def __init__(self, output_stride=4):
        super().__init__(
            input_channels=3,
            output_stride=output_stride
        )
