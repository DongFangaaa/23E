import os
import time

from maix import app, camera, display, image, nn


MODEL_CANDIDATES = [
    "model/model_293643.mud",
    "/root/model/model_293643.mud",
    "/root/models/model_293643.mud",
]

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# False: 检测结果和当前画面对应，适合观察识别效果。
# True: 启用 NPU/CPU 双缓冲，帧率更高，但结果会延迟一帧。
USE_DUAL_BUFF = False


def find_model():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def draw_text(img, text, x, y, color=image.COLOR_GREEN):
    img.draw_string(
        int(x),
        int(y),
        str(text),
        color=color,
        scale=1.0,
        thickness=2,
    )


def main():
    model_path = find_model()
    if model_path is None:
        print("YOLO model not found")
        print("Tried:")
        for path in MODEL_CANDIDATES:
            print("  ", path)
        print("请使用 MaixVision 的 Run Project，或先把 model 目录上传到设备。")
        return

    print("Loading model:", model_path)
    detector = nn.YOLOv5(
        model=model_path,
        dual_buff=USE_DUAL_BUFF,
    )

    width = int(detector.input_width())
    height = int(detector.input_height())
    cam = camera.Camera(
        width,
        height,
        detector.input_format(),
        fps=60,
    )
    disp = display.Display()

    print("Input size:", width, "x", height)
    print("Dual buffer:", USE_DUAL_BUFF)
    print("Press the device function button or stop from MaixVision to exit.")

    fps = 0.0
    last_time = time.monotonic()

    while not app.need_exit():
        img = cam.read()
        objects = detector.detect(
            img,
            conf_th=CONF_THRESHOLD,
            iou_th=IOU_THRESHOLD,
        )

        # 检测后再绘制，不会影响模型输入图像。
        for obj in objects:
            img.draw_rect(
                obj.x,
                obj.y,
                obj.w,
                obj.h,
                color=image.COLOR_RED,
                thickness=2,
            )
            label = "{} {:.2f}".format(
                detector.labels[obj.class_id],
                obj.score,
            )
            draw_text(
                img,
                label,
                obj.x,
                max(0, obj.y - 20),
                image.COLOR_RED,
            )

        now = time.monotonic()
        elapsed = now - last_time
        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            fps = (
                instant_fps
                if fps == 0.0
                else 0.9 * fps + 0.1 * instant_fps
            )
        last_time = now

        draw_text(
            img,
            "FPS: {:.1f}".format(fps),
            10,
            10,
            image.COLOR_GREEN,
        )
        draw_text(
            img,
            "Objects: {}  conf: {:.2f}".format(
                len(objects),
                CONF_THRESHOLD,
            ),
            10,
            35,
            image.COLOR_GREEN,
        )

        disp.show(img)


if __name__ == "__main__":
    main()
