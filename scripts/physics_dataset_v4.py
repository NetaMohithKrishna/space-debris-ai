import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path.home() / "space-debris-ai"

DATA_DIR = BASE_DIR / "data" / "v4"

IMAGE_SIZE = 1024


# ============================================================
# DATASET
# ============================================================

TRAIN_SEQUENCES = 500
VAL_SEQUENCES = 50
TEST_SEQUENCES = 50

SEQUENCE_LENGTH = 16

FRAME_INTERVAL_S = 0.00025


# ============================================================
# CAMERA
# ============================================================

CAMERA = {
    "focal_length_m": 1.0,
    "aperture_m": 0.15,
    "pixel_pitch_m": 5e-6,
    "exposure_s": 0.0001,
    "wavelength_m": 550e-9,
}


# ============================================================
# DEBRIS
# ============================================================

DEBRIS_SIZES_M = [
    0.01,   # 1 cm
    0.02,   # 2 cm
    0.05,   # 5 cm
    0.10,   # 10 cm
]

DISTANCES_M = [
    500,
    1000,
    2000,
    5000,
    10000,
]


# ============================================================
# PHYSICS
# ============================================================

def projected_pixels(size_m, distance_m):

    return (
        size_m
        * CAMERA["focal_length_m"]
        / (
            distance_m
            * CAMERA["pixel_pitch_m"]
        )
    )


def airy_diameter_pixels():

    airy_diameter = (
        2.44
        * CAMERA["wavelength_m"]
        * CAMERA["focal_length_m"]
        / CAMERA["aperture_m"]
    )

    return (
        airy_diameter
        / CAMERA["pixel_pitch_m"]
    )


# ============================================================
# BACKGROUND
# ============================================================

def create_background():

    image = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float32
    )

    # Very low sky background
    image += random.uniform(
        0.5,
        3.0
    )

    return image


# ============================================================
# STARS
# ============================================================

def add_stars(image):

    number = random.randint(
        100,
        300
    )

    for _ in range(number):

        x = random.randrange(
            IMAGE_SIZE
        )

        y = random.randrange(
            IMAGE_SIZE
        )

        brightness = random.uniform(
            20,
            150
        )

        image[y, x] += brightness

    return image


# ============================================================
# STAR PSF
# ============================================================

def add_star_psf(image):

    airy = airy_diameter_pixels()

    sigma = max(
        0.35,
        airy / 2.355
    )

    image = cv2.GaussianBlur(
        image,
        (
            max(3, int(sigma * 8) | 1),
            max(3, int(sigma * 8) | 1)
        ),
        sigma
    )

    return image


# ============================================================
# DEBRIS OBJECT
# ============================================================

def create_debris():

    size_m = random.choice(
        DEBRIS_SIZES_M
    )

    distance_m = random.choice(
        DISTANCES_M
    )

    geometric_pixels = projected_pixels(
        size_m,
        distance_m
    )

    # Random initial position
    x = random.uniform(
        50,
        IMAGE_SIZE - 50
    )

    y = random.uniform(
        50,
        IMAGE_SIZE - 50
    )

    # Relative transverse velocity
    velocity = random.uniform(
        10,
        200
    )

    angle = random.uniform(
        0,
        2 * math.pi
    )

    vx_mps = (
        velocity
        * math.cos(angle)
    )

    vy_mps = (
        velocity
        * math.sin(angle)
    )

    reflectivity = random.uniform(
        0.2,
        1.0
    )

    illumination = random.uniform(
        0.2,
        1.0
    )

    # --------------------------------------------------------
    # Simplified radiometric model
    #
    # Relative received signal is proportional to:
    #
    # projected area
    # × illumination
    # × reflectivity
    # × aperture area
    # ─────────────────────────
    #       distance^2
    #
    # This is a normalized simulation model rather than
    # an absolute spacecraft radiometry model.
    # --------------------------------------------------------

    projected_area_m2 = (
        size_m ** 2
    )

    aperture_area_m2 = (
        math.pi
        * (
            CAMERA["aperture_m"] / 2.0
        ) ** 2
    )

    range_factor = (
        1.0
        / max(
            distance_m ** 2,
            1.0
        )
    )

    radiometric_signal = (
        projected_area_m2
        * illumination
        * reflectivity
        * aperture_area_m2
        * range_factor
    )

    # Normalize the physically-scaled quantity into the
    # simulator's image intensity range.
    #
    # The scale is deliberately configurable rather than
    # treating 255 as a physical photon count.
    signal_scale = 1.0e10

    signal = (
        radiometric_signal
        * signal_scale
    )

    signal = max(
        signal,
        0.0
    )

    return {
        "size_m": size_m,
        "distance_m": distance_m,
        "geometric_pixels": geometric_pixels,

        "x": x,
        "y": y,

        "velocity_mps": velocity,
        "vx_mps": vx_mps,
        "vy_mps": vy_mps,

        "reflectivity": reflectivity,
        "illumination": illumination,
        "signal": signal,
    }


# ============================================================
# DRAW DEBRIS
# ============================================================

def draw_debris(
    image,
    debris,
    frame_index
):
    """
    Render debris with:

    - physical motion between frames
    - motion during exposure
    - sub-pixel positioning
    - optical PSF
    """

    x0 = debris["x"]
    y0 = debris["y"]
    distance = debris["distance_m"]

    geometric_pixels = (
        debris["geometric_pixels"]
    )

    signal = max(
        debris["signal"],
        0.0
    )

    # --------------------------------------------------------
    # Frame time
    # --------------------------------------------------------

    frame_time = (
        frame_index
        * FRAME_INTERVAL_S
    )

    # --------------------------------------------------------
    # Exposure time
    # --------------------------------------------------------

    exposure = CAMERA[
        "exposure_s"
    ]

    # Integrate the object trajectory during exposure.
    exposure_samples = 7

    source = np.zeros_like(
        image,
        dtype=np.float32
    )

    # Physical displacement -> image pixels
    scale = (
        CAMERA["focal_length_m"]
        / (
            distance
            * CAMERA["pixel_pitch_m"]
        )
    )

    valid_positions = []

    # --------------------------------------------------------
    # Exposure integration
    # --------------------------------------------------------

    for sample_index in range(
        exposure_samples
    ):

        # Sample uniformly across exposure.
        sample_offset = (
            (
                sample_index + 0.5
            )
            / exposure_samples
            - 0.5
        )

        t = (
            frame_time
            + sample_offset * exposure
        )

        # Physical motion
        dx_m = (
            debris["vx_mps"]
            * t
        )

        dy_m = (
            debris["vy_mps"]
            * t
        )

        # Convert to pixels
        x = x0 + dx_m * scale
        y = y0 + dy_m * scale

        if not (
            5 <= x < IMAGE_SIZE - 5
            and
            5 <= y < IMAGE_SIZE - 5
        ):
            continue

        valid_positions.append(
            (x, y)
        )

        # ----------------------------------------------------
        # Sub-pixel position
        # ----------------------------------------------------

        x_floor = int(
            math.floor(x)
        )

        y_floor = int(
            math.floor(y)
        )

        fx = x - x_floor
        fy = y - y_floor

        w00 = (1 - fx) * (1 - fy)
        w10 = fx * (1 - fy)
        w01 = (1 - fx) * fy
        w11 = fx * fy

        # Each sample contributes part of the integrated
        # exposure signal.
        sample_signal = (
            signal
            / exposure_samples
        )

        # ----------------------------------------------------
        # Sub-pixel debris
        # ----------------------------------------------------

        if geometric_pixels < 2.0:

            if (
                0 <= x_floor < IMAGE_SIZE
                and
                0 <= y_floor < IMAGE_SIZE
            ):
                source[
                    y_floor,
                    x_floor
                ] += (
                    sample_signal * w00
                )

            if (
                0 <= x_floor + 1 < IMAGE_SIZE
                and
                0 <= y_floor < IMAGE_SIZE
            ):
                source[
                    y_floor,
                    x_floor + 1
                ] += (
                    sample_signal * w10
                )

            if (
                0 <= x_floor < IMAGE_SIZE
                and
                0 <= y_floor + 1 < IMAGE_SIZE
            ):
                source[
                    y_floor + 1,
                    x_floor
                ] += (
                    sample_signal * w01
                )

            if (
                0 <= x_floor + 1 < IMAGE_SIZE
                and
                0 <= y_floor + 1 < IMAGE_SIZE
            ):
                source[
                    y_floor + 1,
                    x_floor + 1
                ] += (
                    sample_signal * w11
                )

        # ----------------------------------------------------
        # Larger debris
        # ----------------------------------------------------

        else:

            radius = (
                geometric_pixels / 2.0
            )

            x_min = max(
                0,
                int(
                    math.floor(
                        x - radius - 2
                    )
                )
            )

            x_max = min(
                IMAGE_SIZE - 1,
                int(
                    math.ceil(
                        x + radius + 2
                    )
                )
            )

            y_min = max(
                0,
                int(
                    math.floor(
                        y - radius - 2
                    )
                )
            )

            y_max = min(
                IMAGE_SIZE - 1,
                int(
                    math.ceil(
                        y + radius + 2
                    )
                )
            )

            yy, xx = np.mgrid[
                y_min:y_max + 1,
                x_min:x_max + 1
            ]

            distance_from_center = np.sqrt(
                (xx - x) ** 2
                +
                (yy - y) ** 2
            )

            mask = (
                distance_from_center <= radius
            )

            region = source[
                y_min:y_max + 1,
                x_min:x_max + 1
            ]

            region[mask] += sample_signal

    # --------------------------------------------------------
    # Object did not intersect image
    # --------------------------------------------------------

    if not valid_positions:
        return None

    # --------------------------------------------------------
    # Optical PSF
    # --------------------------------------------------------

    airy = airy_diameter_pixels()

    sigma = max(
        0.4,
        airy / 2.355
    )

    kernel_size = max(
        3,
        int(sigma * 8) | 1
    )

    source = cv2.GaussianBlur(
        source,
        (
            kernel_size,
            kernel_size
        ),
        sigma
    )

    image += source

    # --------------------------------------------------------
    # Ground-truth center
    # --------------------------------------------------------

    center_x = (
        sum(
            position[0]
            for position in valid_positions
        )
        / len(valid_positions)
    )

    center_y = (
        sum(
            position[1]
            for position in valid_positions
        )
        / len(valid_positions)
    )

    # --------------------------------------------------------
    # Motion extent
    # --------------------------------------------------------

    motion_x = (
        max(
            position[0]
            for position in valid_positions
        )
        -
        min(
            position[0]
            for position in valid_positions
        )
    )

    motion_y = (
        max(
            position[1]
            for position in valid_positions
        )
        -
        min(
            position[1]
            for position in valid_positions
        )
    )

    # Apparent dimensions include:
    #
    # geometric object size
    # optical PSF
    # exposure motion
    apparent_width = max(
        geometric_pixels,
        airy,
        motion_x
    )

    apparent_height = max(
        geometric_pixels,
        airy,
        motion_y
    )

    return {
        "x_pixels": center_x,
        "y_pixels": center_y,

        "geometric_pixels":
            geometric_pixels,

        "apparent_width_pixels":
            apparent_width,

        "apparent_height_pixels":
            apparent_height,

        "motion_x_pixels":
            motion_x,

        "motion_y_pixels":
            motion_y
    }


def sensor_model(image):

    image = np.clip(
        image,
        0,
        None
    )

    # Shot noise
    shot_noise = np.random.normal(
        0,
        np.sqrt(image + 1) * 0.5,
        image.shape
    )

    # Read noise
    read_noise = np.random.normal(
        0,
        1.5,
        image.shape
    )

    image += (
        shot_noise
        + read_noise
    )

    image = np.clip(
        image,
        0,
        255
    )

    return image.astype(
        np.uint8
    )


# ============================================================
# YOLO-LIKE LABEL
#
# This is only a convenient detection target format.
# We are NOT using YOLO.
# ============================================================

def create_label(
    detection
):
    """
    Convert renderer output into normalized
    center and extent coordinates.

    This is only a storage representation.
    The neural network will use its own target format.
    """

    x = detection["x_pixels"]
    y = detection["y_pixels"]

    bbox_w = (
        detection["apparent_width_pixels"]
    )

    bbox_h = (
        detection["apparent_height_pixels"]
    )

    half_w = bbox_w / 2.0
    half_h = bbox_h / 2.0

    x1 = max(
        0.0,
        x - half_w
    )

    y1 = max(
        0.0,
        y - half_h
    )

    x2 = min(
        IMAGE_SIZE - 1.0,
        x + half_w
    )

    y2 = min(
        IMAGE_SIZE - 1.0,
        y + half_h
    )

    cx = (
        (x1 + x2) / 2.0
    ) / IMAGE_SIZE

    cy = (
        (y1 + y2) / 2.0
    ) / IMAGE_SIZE

    w = (
        x2 - x1
    ) / IMAGE_SIZE

    h = (
        y2 - y1
    ) / IMAGE_SIZE

    return [
        cx,
        cy,
        w,
        h
    ]


def generate_sequence():

    # Background
    base = create_background()

    base = add_stars(
        base
    )

    base = add_star_psf(
        base
    )

    base = add_star_psf(
        base
    )

    # Number of debris objects
    number_of_debris = random.randint(
        1,
        4
    )

    debris_objects = [
        create_debris()
        for _ in range(number_of_debris)
    ]

    frames = []

    labels = []

    metadata_frames = []

    for frame_index in range(
        SEQUENCE_LENGTH
    ):

        image = base.copy()

        frame_objects = []

        frame_labels = []

        for debris in debris_objects:

            detection = draw_debris(
                image,
                debris,
                frame_index
            )

            if detection is None:
                continue

            frame_objects.append(
                detection
            )

            frame_labels.append(
                create_label(
                    detection
                )
            )

        image = sensor_model(
            image
        )

        frames.append(
            image
        )

        labels.append(
            frame_labels
        )

        metadata_frames.append(
            frame_objects
        )

    return (
        frames,
        labels,
        metadata_frames,
        debris_objects
    )


# ============================================================
# SAVE SEQUENCE
# ============================================================

def save_sequence(
    split,
    index
):

    image_dir = (
        DATA_DIR
        / split
        / "images"
    )

    label_dir = (
        DATA_DIR
        / split
        / "labels"
    )

    metadata_dir = (
        DATA_DIR
        / split
        / "metadata"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        frames,
        labels,
        metadata_frames,
        debris_objects
    ) = generate_sequence()

    sequence_name = (
        f"{split}_{index:06d}"
    )

    # Save frames
    for frame_index, image in enumerate(
        frames
    ):

        image_path = (
            image_dir
            / f"{sequence_name}_frame_{frame_index:03d}.png"
        )

        cv2.imwrite(
            str(image_path),
            image
        )

        label_path = (
            label_dir
            / f"{sequence_name}_frame_{frame_index:03d}.txt"
        )

        with open(
            label_path,
            "w"
        ) as f:

            for label in labels[frame_index]:

                f.write(
                    "0 "
                    + " ".join(
                        f"{v:.8f}"
                        for v in label
                    )
                    + "\n"
                )

    # Metadata
    metadata = {
        "sequence_length":
            SEQUENCE_LENGTH,

        "frame_interval_s":
            FRAME_INTERVAL_S,

        "camera":
            CAMERA,

        "num_debris":
            len(debris_objects),

        "objects": []
    }

    for debris in debris_objects:

        metadata["objects"].append({
            "physical_size_m":
                debris["size_m"],

            "physical_size_cm":
                debris["size_m"] * 100,

            "distance_m":
                debris["distance_m"],

            "distance_km":
                debris["distance_m"] / 1000,

            "geometric_pixels":
                debris["geometric_pixels"],

            "velocity_mps":
                debris["velocity_mps"],

            "vx_mps":
                debris["vx_mps"],

            "vy_mps":
                debris["vy_mps"],

            "reflectivity":
                debris["reflectivity"],

            "illumination":
                debris["illumination"],
        })

    metadata["frames"] = []

    for frame_index in range(
        SEQUENCE_LENGTH
    ):

        metadata["frames"].append({
            "frame_index":
                frame_index,

            "objects":
                metadata_frames[frame_index]
        })

    metadata_path = (
        metadata_dir
        / f"{sequence_name}.json"
    )

    with open(
        metadata_path,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )


# ============================================================
# GENERATE SPLIT
# ============================================================

def generate_split(
    split,
    count
):

    print(
        f"\nGenerating {split}: "
        f"{count} sequences"
    )

    for i in tqdm(
        range(count),
        desc=split
    ):

        save_sequence(
            split,
            i
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "SPACE DEBRIS PHYSICS DATASET V4"
    )
    print("=" * 70)

    print()
    print("Camera:")
    print(
        f"  focal length: "
        f"{CAMERA['focal_length_m']} m"
    )

    print(
        f"  aperture: "
        f"{CAMERA['aperture_m']} m"
    )

    print(
        f"  pixel pitch: "
        f"{CAMERA['pixel_pitch_m'] * 1e6} um"
    )

    print(
        f"  exposure: "
        f"{CAMERA['exposure_s']} s"
    )

    print()
    print(
        "Airy diameter:",
        airy_diameter_pixels(),
        "pixels"
    )

    print()
    print(
        "Generating multiple-debris "
        "temporal sequences..."
    )

    generate_split(
        "train",
        TRAIN_SEQUENCES
    )

    generate_split(
        "val",
        VAL_SEQUENCES
    )

    generate_split(
        "test",
        TEST_SEQUENCES
    )

    print()
    print("=" * 70)
    print("DATASET COMPLETE")
    print("=" * 70)

    print(
        "Dataset:",
        DATA_DIR
    )


if __name__ == "__main__":
    main()
