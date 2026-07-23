import json
import os
import time

import cv2
import numpy as np
from maix import app, camera, display, image


OUTPUT_PATH = "/root/camera_calib.json"

# OpenCV uses (horizontal inner corners, vertical inner corners).
PATTERN_SIZE = (9, 6)
SQUARE_SIZE_MM = 24.0
TARGET_SAMPLES = 25
MIN_CAPTURE_INTERVAL_S = 1.5
MIN_SHARPNESS = 30.0

# These must match the image size used by main.py. The current model and PnP
# configuration use a 448 x 448 image with principal point near (224, 224).
CALIB_WIDTH = 448
CALIB_HEIGHT = 448


def make_object_points():
    cols, rows = PATTERN_SIZE
    points = np.zeros((rows * cols, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= SQUARE_SIZE_MM
    return points


def corner_signature(corners, width, height):
    """Normalize all corners so different board poses can be compared."""
    signature = corners.reshape(-1, 2).astype(np.float32).copy()
    signature[:, 0] /= float(width)
    signature[:, 1] /= float(height)
    return signature


def is_new_pose(signature, saved_signatures):
    if not saved_signatures:
        return True

    # Mean corner displacement relative to the image size. This rejects
    # repeated captures while allowing changes in position, scale and angle.
    for saved in saved_signatures:
        mean_displacement = float(
            np.mean(np.linalg.norm(signature - saved, axis=1))
        )
        if mean_displacement < 0.035:
            return False
    return True


def reprojection_errors(object_points, image_points, rvecs, tvecs, matrix, dist):
    errors = []
    for index, object_point in enumerate(object_points):
        projected, _ = cv2.projectPoints(
            object_point,
            rvecs[index],
            tvecs[index],
            matrix,
            dist,
        )
        # Root-mean-square pixel error for this view. Dividing the L2 norm
        # directly by N would make the reported error artificially too small.
        error = cv2.norm(
            image_points[index],
            projected,
            cv2.NORM_L2,
        ) / np.sqrt(float(len(projected)))
        errors.append(float(error))
    return errors


def calibrate(object_points, image_points, image_size):
    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    errors = reprojection_errors(
        object_points,
        image_points,
        rvecs,
        tvecs,
        matrix,
        dist,
    )

    # Remove only obvious bad samples, then calibrate once more.
    median_error = float(np.median(np.array(errors, dtype=np.float64)))
    error_limit = max(0.8, median_error * 2.5)
    keep_indices = [
        index for index, error in enumerate(errors) if error <= error_limit
    ]

    if 10 <= len(keep_indices) < len(object_points):
        object_points = [object_points[index] for index in keep_indices]
        image_points = [image_points[index] for index in keep_indices]
        rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
        errors = reprojection_errors(
            object_points,
            image_points,
            rvecs,
            tvecs,
            matrix,
            dist,
        )

    return {
        "rms": float(rms),
        "camera_matrix": matrix,
        "dist_coeffs": dist,
        "view_errors": errors,
        "used_samples": len(object_points),
    }


def save_result(result, width, height):
    view_errors = result["view_errors"]
    data = {
        "image_width": int(width),
        "image_height": int(height),
        "pattern_cols": int(PATTERN_SIZE[0]),
        "pattern_rows": int(PATTERN_SIZE[1]),
        "square_size_mm": float(SQUARE_SIZE_MM),
        "rms": result["rms"],
        "mean_reprojection_error": (
            float(np.mean(np.array(view_errors, dtype=np.float64)))
            if view_errors
            else 0.0
        ),
        "used_samples": int(result["used_samples"]),
        "camera_matrix": result["camera_matrix"].tolist(),
        "dist_coeffs": result["dist_coeffs"].reshape(-1).tolist(),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return data


def draw_status(frame, text, color=(0, 255, 0), y=30):
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
    )


def main():
    width = CALIB_WIDTH
    height = CALIB_HEIGHT

    print("Calibration resolution:", width, "x", height)
    print("Chessboard inner corners: 9 x 6")
    print("Square size:", SQUARE_SIZE_MM, "mm")
    print("Move the board around the whole image and change its angle.")

    cam = camera.Camera(width, height,fps=60)
    disp = display.Display()

    template_object_points = make_object_points()
    object_points = []
    image_points = []
    saved_signatures = []
    last_capture_time = 0.0
    result_data = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    find_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    while not app.need_exit():
        maix_image = cam.read()
        frame = image.image2cv(
            maix_image,
            ensure_bgr=True,
            copy=True,
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            PATTERN_SIZE,
            flags=find_flags,
        )

        status = "Show the 9x6 inner-corner board"
        status_color = (0, 0, 255)

        if found:
            refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria,
            )
            cv2.drawChessboardCorners(
                frame,
                PATTERN_SIZE,
                refined,
                True,
            )

            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            signature = corner_signature(refined, width, height)
            now = time.monotonic()

            if sharpness < MIN_SHARPNESS:
                status = "Too blurry - hold the board still"
                status_color = (0, 165, 255)
            elif not is_new_pose(signature, saved_signatures):
                status = "Move/tilt the board to a new pose"
                status_color = (0, 165, 255)
            elif now - last_capture_time < MIN_CAPTURE_INTERVAL_S:
                status = "Hold still..."
                status_color = (0, 255, 255)
            else:
                object_points.append(template_object_points.copy())
                image_points.append(refined.copy())
                saved_signatures.append(signature)
                last_capture_time = now
                status = "Captured"
                status_color = (0, 255, 0)
                print(
                    "Captured sample",
                    len(image_points),
                    "/",
                    TARGET_SAMPLES,
                    "sharpness:",
                    round(sharpness, 1),
                )

        draw_status(
            frame,
            "Samples: {}/{}".format(len(image_points), TARGET_SAMPLES),
            (0, 255, 0),
            28,
        )
        draw_status(frame, status, status_color, 58)

        if len(image_points) >= TARGET_SAMPLES:
            draw_status(frame, "Calibrating...", (0, 255, 255), 88)
            disp.show(image.cv2image(frame, bgr=True, copy=True))

            result = calibrate(
                object_points,
                image_points,
                (width, height),
            )
            result_data = save_result(result, width, height)

            print("Calibration finished")
            print("Used samples:", result_data["used_samples"])
            print("RMS:", result_data["rms"])
            print(
                "Mean reprojection error:",
                result_data["mean_reprojection_error"],
                "pixel",
            )
            print("Camera matrix:")
            print(np.array(result_data["camera_matrix"]))
            print("Distortion coefficients:")
            print(np.array(result_data["dist_coeffs"]))
            print("Saved to:", os.path.abspath(OUTPUT_PATH))
            break

        disp.show(image.cv2image(frame, bgr=True, copy=True))

    if result_data is not None:
        # Keep the result visible until the user exits the app.
        while not app.need_exit():
            maix_image = cam.read()
            frame = image.image2cv(
                maix_image,
                ensure_bgr=True,
                copy=True,
            )
            draw_status(frame, "Calibration saved", (0, 255, 0), 30)
            draw_status(
                frame,
                "Mean error: {:.3f}px".format(
                    result_data["mean_reprojection_error"]
                ),
                (0, 255, 0),
                60,
            )
            disp.show(image.cv2image(frame, bgr=True, copy=True))


if __name__ == "__main__":
    main()
