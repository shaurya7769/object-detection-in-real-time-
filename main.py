import cv2
import numpy as np
from flask import Flask, render_template_string, Response
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timezone
import hashlib

app = Flask(__name__)

model = YOLO("yolov8n.pt")  # your existing model

# ------------------------------------------------------
# MATERIAL CLASSIFICATION RULES (ADDED)
# ------------------------------------------------------
MATERIAL_MAP = {
    "bottle": ["plastic bottle", "metal bottle"],
    "cup": ["plastic cup", "paper cup"],
    "book": "paper",
    "cell phone": "metal",
    "laptop": "metal",
    "keyboard": "plastic",
    "mouse": "plastic",
    "remote": "plastic",
    "backpack": "cloth",
    "handbag": "cloth",
    "towel": "cloth",
    "chair": "wood/metal",
    "bowl": "plastic",
    "spoon": "metal",
    "fork": "metal",
    "knife": "metal",
    "orange": "organic",
    "banana": "organic",
}

def classify_material(cls_name, frame, box):
    """
    Smart rule-based material classifier
    """
    cls_name = cls_name.lower()

    # Direct mapping if simple
    if cls_name in MATERIAL_MAP and isinstance(MATERIAL_MAP[cls_name], str):
        return MATERIAL_MAP[cls_name]

    # For items that can be multi-material (e.g., bottle, cup)
    if cls_name == "bottle":
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]

        # heuristic: metallic is more reflective → high pixel variance
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        var = np.var(gray)

        if var > 2500:
            return "metal bottle"
        else:
            return "plastic bottle"

    if cls_name == "cup":
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        h, w = crop.shape[:2]

        # paper cups are usually matte + brighter
        brightness = np.mean(crop)

        if brightness > 140:
            return "paper cup"
        else:
            return "plastic cup"

    # fallback if unknown
    return cls_name
# ------------------------------------------------------


def add_visible_watermark_cv2(frame, text):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = image_pil.width - text_w - 20
    y = image_pil.height - text_h - 20

    draw.rectangle((x - 10, y - 5, x + text_w + 10, y + text_h + 5), fill=(0, 0, 0, 128))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def compute_sha256(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return hashlib.sha256(buffer).hexdigest()


def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, conf=0.5)
        result = results[0]

        annotated_frame = frame.copy()

        # ------------------------------------------------------
        # MATERIAL LABEL DRAWING (ADDED)
        # ------------------------------------------------------
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            cls_name = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box.tolist())

            material_label = classify_material(cls_name, frame, box)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f"{material_label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        # ------------------------------------------------------

        timestamp = datetime.now(timezone.utc).isoformat()
        annotated_frame = add_visible_watermark_cv2(
            annotated_frame,
            f"Secure Ocean Detection | {timestamp}"
        )

        hash_val = compute_sha256(annotated_frame)
        cv2.putText(
            annotated_frame,
            f"SHA256: {hash_val[:12]}...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template_string('''
    <html>
    <head>
        <title>Secure Plastic Detection</title>
        <style>
            body { background-color: #111; color: white; text-align: center; font-family: Arial; }
            h1 { color: #4dd0e1; margin-top: 20px; }
            img { border: 3px solid #4dd0e1; border-radius: 12px; }
        </style>
    </head>
    <body>
        <h1>🌊 Secure AI-based Waste Material Detection</h1>
        <img src="{{ url_for('video_feed') }}" width="1080" />
        <p>Detecting objects & their materials in real-time.</p>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
