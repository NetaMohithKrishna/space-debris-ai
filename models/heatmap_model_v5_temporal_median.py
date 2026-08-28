from models.heatmap_model_v5_hr import HeatmapDetectorV5HR


class HeatmapDetectorV5TemporalMedian(HeatmapDetectorV5HR):
    """
    4-channel V5-HR:

      0 = current frame
      1 = |current - previous|
      2 = |next - current|
      3 = |current - temporal median|
    """

    def __init__(self, output_stride=4):
        super().__init__(
            input_channels=4,
            output_stride=output_stride
        )
