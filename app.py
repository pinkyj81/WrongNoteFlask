import os
import re
import uuid
import json
from datetime import date, datetime
from urllib.parse import urlparse

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, url_for
from sqlalchemy import inspect, text
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

from config import AppConfig


db = SQLAlchemy()
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


class WrongNote(db.Model):
    __tablename__ = "wrong_notes"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    subject = db.Column(db.String(30), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    error_type = db.Column(db.String(50), nullable=False)
    image_url = db.Column(db.String(500), nullable=False)
    question_text = db.Column(db.Text, nullable=True)
    choices_json = db.Column(db.Text, nullable=True)
    similar_json = db.Column(db.Text, nullable=True)
    solved = db.Column(db.Boolean, default=False, nullable=False)
    review_date = db.Column(db.Date, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def as_dict(self):
        parsed_choices = []
        if self.choices_json:
            try:
                parsed = json.loads(self.choices_json)
                if isinstance(parsed, list):
                    parsed_choices = [str(item) for item in parsed if str(item).strip()]
            except Exception:
                parsed_choices = []

        parsed_similar = []
        if self.similar_json:
            try:
                raw_items = json.loads(self.similar_json)
                if isinstance(raw_items, list):
                    parsed_similar = normalize_similar_items(raw_items)
            except Exception:
                parsed_similar = []

        return {
            "id": self.id,
            "subject": self.subject,
            "title": self.title,
            "error_type": self.error_type,
            "image_url": normalize_image_url(self.image_url),
            "question_text": self.question_text,
            "choices": parsed_choices,
            "similar_items": parsed_similar,
            "solved": self.solved,
            "review_date": self.review_date.isoformat(),
            "created_at": self.created_at.isoformat(),
        }


def normalize_image_url(raw_url: str) -> str:
    value = str(raw_url or "").strip()
    if not value:
        return ""

    normalized = value.replace("\\", "/")

    static_marker = "/static/uploads/"
    marker_index = normalized.find(static_marker)
    if marker_index >= 0:
        return normalized[marker_index:]

    if normalized.startswith("static/uploads/"):
        return f"/{normalized}"

    if normalized.startswith("/uploads/"):
        return f"/static{normalized}"

    if normalized.startswith("uploads/"):
        return f"/static/{normalized}"

    if normalized.startswith(("http://", "https://")):
        parsed = urlparse(normalized)
        host = (parsed.hostname or "").lower()
        if host in {"127.0.0.1", "localhost"} and parsed.path:
            if parsed.path.startswith("/static/"):
                return parsed.path
            if parsed.path.startswith("/uploads/"):
                return f"/static{parsed.path}"

    return normalized


def build_subject_counts(notes):
    counts = {
        "전체": len(notes),
        "수학": 0,
        "영어": 0,
        "과학": 0,
        "국어": 0,
        "기타": 0,
    }

    for note in notes:
        if note.subject in counts:
            counts[note.subject] += 1
        else:
            counts["기타"] += 1

    return counts


def is_allowed_image(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS


def decode_image(file_bytes: bytes):
    np_data = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_COLOR)


def encode_jpg(image, quality: int = 96) -> bytes:
    encoded_ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not encoded_ok:
        raise ValueError("failed to encode image")
    return encoded.tobytes()


def detect_question_roi(gray):
    height, width = gray.shape
    img_area = height * width

    inv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        12,
    )
    inv = cv2.medianBlur(inv, 3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    filtered = np.zeros_like(inv)
    min_area = max(12, int(img_area * 0.00003))
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    kx = max(15, width // 35)
    ky = max(3, height // 150)
    merged = cv2.morphologyEx(
        filtered,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)),
        iterations=2,
    )
    merged = cv2.dilate(merged, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, width, height, 1.0

    best = None
    best_score = -1.0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < img_area * 0.006:
            continue
        if w < width * 0.18 or h < height * 0.08:
            continue

        roi = filtered[y:y + h, x:x + w]
        text_pixels = int(np.count_nonzero(roi))
        density = text_pixels / max(1, area)
        if density < 0.004:
            continue

        score = (area / img_area) * (1.0 + min(density * 10.0, 1.5))
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        return 0, 0, width, height, 1.0

    x, y, w, h = best
    x1, y1, x2, y2 = x, y, x + w, y + h

    pad = int(min(height, width) * 0.03)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    roi_ratio = ((x2 - x1) * (y2 - y1)) / img_area
    return x1, y1, x2, y2, roi_ratio


def keep_only_question_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    x1, y1, x2, y2, roi_ratio = detect_question_roi(gray)

    # Safety fallback: if detected region is too small/too large, skip region wipe.
    if roi_ratio < 0.18 or roi_ratio > 0.98:
        return image

    if len(image.shape) == 2:
        canvas = np.full_like(image, 255)
    else:
        canvas = np.full_like(image, 255)

    canvas[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return canvas


def process_image_bytes(
    payload_bytes: bytes,
    clean_pencil: bool,
    clean_strength: str,
    focus_region: bool,
):
    if clean_pencil:
        payload_bytes = clean_pencil_marks(
            payload_bytes,
            strength=clean_strength,
            focus_region=focus_region,
        )
        return payload_bytes, "jpg"

    if focus_region:
        image = decode_image(payload_bytes)
        if image is None:
            raise ValueError("invalid image data")
        focused = keep_only_question_region(image)
        payload_bytes = encode_jpg(focused, quality=96)
        return payload_bytes, "jpg"

    return payload_bytes, None


def parse_pasted_question_text(raw_text: str):
    text = (raw_text or "").replace("\r\n", "\n").strip()
    if not text:
        return {
            "title": "",
            "question": "",
            "choices": [],
            "normalized_text": "",
        }

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    lines = [line for line in lines if line.lower() not in {"문제"}]

    # Candidate title: the first short non-choice line.
    title = ""
    choice_line_pattern = re.compile(r"^[0-9①-⑩]+[\).\s]+")
    for line in lines:
        if choice_line_pattern.match(line):
            continue
        if len(line) <= 25:
            title = line
            break

    # Capture and split choice lines like "① 3 ② 6 ..." or "1) ...".
    choices = []
    for line in lines:
        if re.search(r"[①②③④⑤⑥⑦⑧⑨⑩]", line) or re.search(r"\b[1-5][\)\.]", line):
            choices.extend(split_choices_line(line))

    question_lines = []
    for line in lines:
        if line == title:
            continue
        if line in choices:
            continue
        question_lines.append(line)

    question = "\n".join(question_lines).strip()
    normalized_parts = []
    if title:
        normalized_parts.append(title)
    if question:
        normalized_parts.append(question)
    if choices:
        normalized_parts.append("\n".join(choices))
    normalized_text = "\n\n".join(normalized_parts)[:10000]

    return {
        "title": title[:200],
        "question": question[:6000],
        "choices": choices[:20],
        "normalized_text": normalized_text,
    }


def normalize_choices(choices_value):
    if choices_value is None:
        return []

    if isinstance(choices_value, list):
        items = choices_value
    else:
        items = str(choices_value).replace("\r\n", "\n").split("\n")

    normalized = []
    for item in items:
        for token in split_choices_line(str(item)):
            text_item = token.strip()
            if text_item:
                normalized.append(text_item)

    return normalized[:20]


def split_similar_blocks(raw_text: str):
    text = (raw_text or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    pattern = re.compile(r"(?m)^\s*(\d{1,2})[\)\.]\s*")
    matches = list(pattern.finditer(text))
    if not matches:
        return [text]

    blocks = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            blocks.append(body)
    return blocks


def parse_similar_questions(raw_text: str):
    text = (raw_text or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    q_text = text
    answer_text = ""
    answer_marker = re.search(r"(?im)^\s*정답\s*$", text)
    if answer_marker:
        q_text = text[:answer_marker.start()].strip()
        answer_text = text[answer_marker.end():].strip()

    answer_map = parse_answer_map(answer_text)
    blocks = split_similar_blocks(q_text)
    result = []
    for idx, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        choices = []
        question_lines = []
        for line in lines:
            if re.search(r"[①②③④⑤⑥⑦⑧⑨⑩]", line) or re.search(r"\b[1-9][\)\.]", line):
                split_choice = split_choices_line(line)
                if split_choice:
                    choices.extend(split_choice)
                else:
                    question_lines.append(line)
            else:
                question_lines.append(line)

        question = "\n".join(question_lines).strip() or block
        result.append({
            "number": idx,
            "question": question[:2000],
            "choices": choices[:20],
            "answer": answer_map.get(idx, ""),
        })

    return result[:30]


def normalize_similar_items(items_value):
    if not items_value:
        return []

    normalized = []
    for idx, item in enumerate(items_value, start=1):
        if isinstance(item, dict):
            number = item.get("number", idx)
            question = str(item.get("question", "")).strip()
            choices = normalize_choices(item.get("choices", []))
            answer = normalize_answer(item.get("answer", ""))
        else:
            number = idx
            question = str(item).strip()
            choices = []
            answer = ""

        if not question:
            continue
        normalized.append({
            "number": int(number) if str(number).isdigit() else idx,
            "question": question[:2000],
            "choices": choices[:20],
            "answer": answer,
        })

    return normalized[:30]


def split_choices_line(line: str):
    text_line = (line or "").strip()
    if not text_line:
        return []

    # Choices may come in one line, so split by choice markers.
    if re.search(r"[①②③④⑤⑥⑦⑧⑨⑩]", text_line):
        parts = [p.strip() for p in re.split(r"(?=[①②③④⑤⑥⑦⑧⑨⑩])", text_line) if p.strip()]
    elif re.search(r"\b[1-9][\)\.]", text_line):
        parts = [p.strip() for p in re.split(r"(?=\b[1-9][\)\.])", text_line) if p.strip()]
    else:
        parts = [text_line]

    cleaned = []
    for part in parts:
        # Remove common separator noise without altering meaningful symbols.
        part = re.sub(r"\s*\?\s*(?=[①②③④⑤⑥⑦⑧⑨⑩]|\b[1-9][\)\.])", " ", part)
        part = part.strip(" |,;:")
        if part.endswith("?") and not re.search(r"[가-힣A-Za-z0-9]\?", part):
            part = part[:-1].rstrip()
        if part:
            cleaned.append(part)

    return cleaned


def normalize_answer(value) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    circled = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤", "6": "⑥", "7": "⑦", "8": "⑧", "9": "⑨"}
    reverse = {v: k for k, v in circled.items()}

    if raw in circled.values():
        return raw
    if raw in circled:
        return circled[raw]

    m = re.search(r"([①②③④⑤⑥⑦⑧⑨]|[1-9])", raw)
    if not m:
        return ""
    token = m.group(1)
    if token in reverse:
        return token
    return circled.get(token, "")


def parse_answer_map(answer_text: str):
    if not answer_text:
        return {}

    text = (answer_text or "").replace("\r\n", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines = [line for line in lines if line not in {"번호", "정답"}]
    compact = "\n".join(lines)

    pairs = re.findall(r"(\d{1,2})\s*(?:번)?\s*[:\-\.]?\s*([①②③④⑤⑥⑦⑧⑨]|[1-9])", compact)
    answer_map = {}
    for num_str, ans_token in pairs:
        num = int(num_str)
        if 1 <= num <= 50:
            norm = normalize_answer(ans_token)
            if norm:
                answer_map[num] = norm

    return answer_map


def ensure_wrong_notes_schema() -> None:
    db.create_all()

    inspector = inspect(db.engine)
    columns = {col["name"].lower() for col in inspector.get_columns(WrongNote.__tablename__)}
    if "question_text" not in columns:
        db.session.execute(text(f"ALTER TABLE {WrongNote.__tablename__} ADD question_text NVARCHAR(MAX) NULL"))
        db.session.commit()
    if "choices_json" not in columns:
        db.session.execute(text(f"ALTER TABLE {WrongNote.__tablename__} ADD choices_json NVARCHAR(MAX) NULL"))
        db.session.commit()
    if "similar_json" not in columns:
        db.session.execute(text(f"ALTER TABLE {WrongNote.__tablename__} ADD similar_json NVARCHAR(MAX) NULL"))
        db.session.commit()


def clean_pencil_marks(file_bytes: bytes, strength: str = "strong", focus_region: bool = False) -> bytes:
    image = decode_image(file_bytes)
    if image is None:
        raise ValueError("invalid image data")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if strength == "light":
        sat_limit, low_gray, high_gray, inpaint_radius = 45, 60, 190, 2
    elif strength == "medium":
        sat_limit, low_gray, high_gray, inpaint_radius = 55, 55, 205, 3
    else:
        sat_limit, low_gray, high_gray, inpaint_radius = 70, 45, 220, 4

    # Pencil strokes are often low-saturation gray; build a mask and inpaint them.
    low_sat_mask = cv2.inRange(hsv[:, :, 1], 0, sat_limit)
    gray_band_mask = cv2.inRange(gray, low_gray, high_gray)
    pencil_mask = cv2.bitwise_and(low_sat_mask, gray_band_mask)
    pencil_mask = cv2.medianBlur(pencil_mask, 3)
    pencil_mask = cv2.morphologyEx(
        pencil_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    pencil_mask = cv2.dilate(pencil_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    inpainted = cv2.inpaint(image, pencil_mask, inpaint_radius, cv2.INPAINT_TELEA)

    gray_clean = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray_clean, None, h=13, templateWindowSize=7, searchWindowSize=21)
    scanned = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )

    # Remove tiny black speckles left after thresholding.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(255 - scanned, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 14:
            scanned[labels == i] = 255

    scanned = cv2.medianBlur(scanned, 3)
    if focus_region:
        scanned = keep_only_question_region(scanned)

    return encode_jpg(scanned, quality=96)


def build_upload_paths(base_upload_dir: str, bucket: str, ext: str, suffix: str = ""):
    now = datetime.utcnow()
    date_path = now.strftime("%Y/%m/%d")
    relative_dir = os.path.join(bucket, *date_path.split("/"))
    target_dir = os.path.join(base_upload_dir, relative_dir)
    os.makedirs(target_dir, exist_ok=True)

    unique_name = f"{now.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}.{ext}"
    save_path = os.path.join(target_dir, unique_name)
    static_rel_path = os.path.join("uploads", relative_dir, unique_name).replace("\\", "/")
    return save_path, static_rel_path


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(AppConfig)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    db.init_app(app)
    app.config["DB_READY"] = False
    app.config["DB_INIT_ERROR"] = None

    @app.route("/")
    def index():
        if not app.config.get("DB_READY"):
            return render_template(
                "index.html",
                notes=[],
                selected_subject="전체",
                subject_counts={"전체": 0, "수학": 0, "영어": 0, "과학": 0, "국어": 0, "기타": 0},
                stats={
                    "review_today": 0,
                    "completed": 0,
                    "scheduled": 0,
                    "total": 0,
                    "accuracy": 0,
                },
                db_error=(
                    app.config.get("DB_INIT_ERROR")
                    or "DB 연결이 아직 준비되지 않았습니다. 환경변수를 먼저 설정해주세요."
                ),
            )

        subject_filter = request.args.get("subject", "전체")
        all_notes = (
            WrongNote.query.order_by(WrongNote.review_date.desc(), WrongNote.id.desc()).all()
        )

        notes = all_notes
        if subject_filter != "전체":
            notes = [item for item in all_notes if item.subject == subject_filter]

        for item in notes:
            item.image_url = normalize_image_url(item.image_url)

        today = date.today()
        review_today = sum(1 for item in all_notes if not item.solved and item.review_date <= today)
        completed = sum(1 for item in all_notes if item.solved)
        scheduled = sum(1 for item in all_notes if not item.solved and item.review_date > today)
        total = len(all_notes)
        accuracy = round((completed / total) * 100) if total else 0

        return render_template(
            "index.html",
            notes=notes,
            selected_subject=subject_filter,
            subject_counts=build_subject_counts(all_notes),
            stats={
                "review_today": review_today,
                "completed": completed,
                "scheduled": scheduled,
                "total": total,
                "accuracy": accuracy,
            },
        )

    @app.get("/api/notes")
    def get_notes():
        if not app.config.get("DB_READY"):
            return jsonify({"ok": False, "message": "database not ready"}), 503

        notes = WrongNote.query.order_by(WrongNote.review_date.desc(), WrongNote.id.desc()).all()
        return jsonify([item.as_dict() for item in notes])

    @app.post("/api/parse-text")
    def parse_text():
        payload = request.get_json(silent=True) or {}
        raw_text = payload.get("raw_text", "")
        if not str(raw_text).strip():
            return jsonify({"ok": False, "message": "raw_text is required"}), 400

        parsed = parse_pasted_question_text(str(raw_text))
        return jsonify({"ok": True, "parsed": parsed})

    @app.post("/api/parse-similar-text")
    def parse_similar_text():
        payload = request.get_json(silent=True) or {}
        raw_text = payload.get("raw_text", "")
        if not str(raw_text).strip():
            return jsonify({"ok": False, "message": "raw_text is required"}), 400

        parsed = parse_similar_questions(str(raw_text))
        return jsonify({"ok": True, "similar_items": parsed})

    @app.get("/notes/<int:note_id>")
    def note_detail(note_id: int):
        if not app.config.get("DB_READY"):
            return render_template(
                "detail.html",
                note=None,
                db_error="DB 연결이 아직 준비되지 않았습니다. 환경변수를 먼저 설정해주세요.",
            )

        note = db.session.get(WrongNote, note_id)
        if note is None:
            return render_template("detail.html", note=None, db_error="해당 오답을 찾을 수 없습니다."), 404

        note.image_url = normalize_image_url(note.image_url)

        parsed_choices = []
        if note.choices_json:
            try:
                parsed = json.loads(note.choices_json)
                if isinstance(parsed, list):
                    parsed_choices = normalize_choices(parsed)
            except Exception:
                parsed_choices = []

        parsed_similar = []
        if note.similar_json:
            try:
                raw_items = json.loads(note.similar_json)
                if isinstance(raw_items, list):
                    parsed_similar = normalize_similar_items(raw_items)
            except Exception:
                parsed_similar = []

        return render_template(
            "detail.html",
            note=note,
            parsed_choices=parsed_choices,
            parsed_similar=parsed_similar,
            db_error=None,
        )

    @app.post("/api/notes")
    def create_note():
        if not app.config.get("DB_READY"):
            return jsonify({"ok": False, "message": "database not ready"}), 503

        payload = request.get_json(silent=True) or {}

        required = ["subject", "title", "error_type", "image_url", "review_date"]
        missing = [key for key in required if not payload.get(key)]
        if missing:
            return jsonify({"ok": False, "message": f"missing fields: {', '.join(missing)}"}), 400

        try:
            review_date = datetime.strptime(payload["review_date"], "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"ok": False, "message": "review_date format must be YYYY-MM-DD"}), 400

        choices_list = normalize_choices(payload.get("choices"))
        similar_items = normalize_similar_items(payload.get("similar_items"))

        new_item = WrongNote(
            subject=payload["subject"],
            title=payload["title"],
            error_type=payload["error_type"],
            image_url=normalize_image_url(payload["image_url"]),
            question_text=(payload.get("question_text") or None),
            choices_json=(
                json.dumps(choices_list, ensure_ascii=False)
                if choices_list
                else None
            ),
            similar_json=(
                json.dumps(similar_items, ensure_ascii=False)
                if similar_items
                else None
            ),
            review_date=review_date,
            solved=bool(payload.get("solved", False)),
        )

        db.session.add(new_item)
        db.session.commit()
        return jsonify({"ok": True, "note": new_item.as_dict()})

    @app.patch("/api/notes/<int:note_id>")
    def update_note(note_id: int):
        if not app.config.get("DB_READY"):
            return jsonify({"ok": False, "message": "database not ready"}), 503

        note = db.session.get(WrongNote, note_id)
        if note is None:
            return jsonify({"ok": False, "message": "note not found"}), 404

        payload = request.get_json(silent=True) or {}

        if "title" in payload:
            title = str(payload.get("title") or "").strip()
            if title:
                note.title = title[:200]

        if "question_text" in payload:
            question_text = str(payload.get("question_text") or "").strip()
            note.question_text = question_text or None

        if "choices" in payload:
            choices_list = normalize_choices(payload.get("choices"))
            note.choices_json = json.dumps(choices_list, ensure_ascii=False) if choices_list else None

        if "similar_items" in payload:
            similar_items = normalize_similar_items(payload.get("similar_items"))
            note.similar_json = json.dumps(similar_items, ensure_ascii=False) if similar_items else None

        db.session.commit()
        return jsonify({"ok": True, "note": note.as_dict()})

    @app.post("/api/notes/delete")
    def delete_notes():
        if not app.config.get("DB_READY"):
            return jsonify({"ok": False, "message": "database not ready"}), 503

        payload = request.get_json(silent=True) or {}
        ids = payload.get("ids") or []
        if not isinstance(ids, list):
            return jsonify({"ok": False, "message": "ids must be a list"}), 400

        normalized_ids = []
        for item in ids:
            try:
                normalized_ids.append(int(item))
            except (TypeError, ValueError):
                continue

        if not normalized_ids:
            return jsonify({"ok": False, "message": "no valid ids provided"}), 400

        notes = WrongNote.query.filter(WrongNote.id.in_(normalized_ids)).all()
        deleted_count = len(notes)
        for note in notes:
            db.session.delete(note)

        db.session.commit()
        return jsonify({"ok": True, "deleted_count": deleted_count})

    @app.post("/api/upload-image")
    def upload_image():
        image_file = request.files.get("image")
        if image_file is None or image_file.filename == "":
            return jsonify({"ok": False, "message": "image file is required"}), 400

        if not is_allowed_image(image_file.filename):
            allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
            return jsonify({"ok": False, "message": f"allowed extensions: {allowed}"}), 400

        original_name = secure_filename(image_file.filename)
        ext = original_name.rsplit(".", 1)[1].lower()
        clean_pencil = (request.form.get("clean_pencil") or "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        clean_strength = (request.form.get("clean_strength") or "strong").strip().lower()
        if clean_strength not in {"light", "medium", "strong"}:
            clean_strength = "strong"
        focus_region = (request.form.get("focus_region") or "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        payload_bytes = image_file.read()
        payload_bytes, force_ext = process_image_bytes(
            payload_bytes,
            clean_pencil=clean_pencil,
            clean_strength=clean_strength,
            focus_region=focus_region,
        )
        if force_ext:
            ext = force_ext

        bucket = "clean" if clean_pencil or focus_region else "raw"
        suffix = "_clean" if clean_pencil or focus_region else ""

        save_path, static_rel_path = build_upload_paths(
            app.config["UPLOAD_FOLDER"],
            bucket=bucket,
            ext=ext,
            suffix=suffix,
        )
        with open(save_path, "wb") as fp:
            fp.write(payload_bytes)

        image_url = url_for("static", filename=static_rel_path)
        return jsonify(
            {
                "ok": True,
                "image_url": image_url,
                "clean_pencil": clean_pencil,
                "clean_strength": clean_strength,
                "focus_region": focus_region,
            }
        )

    @app.post("/api/preview-image")
    def preview_image():
        image_file = request.files.get("image")
        if image_file is None or image_file.filename == "":
            return jsonify({"ok": False, "message": "image file is required"}), 400

        if not is_allowed_image(image_file.filename):
            allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
            return jsonify({"ok": False, "message": f"allowed extensions: {allowed}"}), 400

        original_name = secure_filename(image_file.filename)
        ext = original_name.rsplit(".", 1)[1].lower()
        clean_pencil = (request.form.get("clean_pencil") or "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        clean_strength = (request.form.get("clean_strength") or "strong").strip().lower()
        if clean_strength not in {"light", "medium", "strong"}:
            clean_strength = "strong"
        focus_region = (request.form.get("focus_region") or "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        payload_bytes = image_file.read()
        payload_bytes, force_ext = process_image_bytes(
            payload_bytes,
            clean_pencil=clean_pencil,
            clean_strength=clean_strength,
            focus_region=focus_region,
        )
        if force_ext:
            ext = force_ext

        save_path, static_rel_path = build_upload_paths(
            app.config["UPLOAD_FOLDER"],
            bucket="preview",
            ext=ext,
            suffix="_preview",
        )
        with open(save_path, "wb") as fp:
            fp.write(payload_bytes)

        preview_url = url_for("static", filename=static_rel_path)
        return jsonify(
            {
                "ok": True,
                "preview_url": preview_url,
                "clean_pencil": clean_pencil,
                "clean_strength": clean_strength,
                "focus_region": focus_region,
            }
        )

    @app.post("/api/init-db")
    def init_db_tables():
        try:
            ensure_wrong_notes_schema()
            app.config["DB_READY"] = True
            return jsonify(
                {
                    "ok": True,
                    "message": "tables created or already exist",
                    "database": app.config.get("DB_NAME"),
                    "server": app.config.get("DB_SERVER"),
                    "table": WrongNote.__tablename__,
                }
            )
        except Exception as exc:
            app.config["DB_READY"] = False
            return jsonify({"ok": False, "message": str(exc)}), 500

    @app.post("/seed-sample")
    def seed_sample():
        if not app.config.get("DB_READY"):
            return jsonify({"ok": False, "message": "database not ready"}), 503

        if WrongNote.query.count() > 0:
            return jsonify({"ok": True, "message": "already seeded"})

        sample_notes = [
            WrongNote(
                subject="수학",
                title="이차방정식",
                error_type="계산 실수",
                image_url="https://images.unsplash.com/photo-1635070041078-e363dbe005cb?auto=format&fit=crop&w=420&q=80",
                review_date=date(2024, 5, 25),
                solved=False,
            ),
            WrongNote(
                subject="영어",
                title="관계대명사",
                error_type="개념 부족",
                image_url="https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?auto=format&fit=crop&w=420&q=80",
                review_date=date(2024, 5, 24),
                solved=False,
            ),
            WrongNote(
                subject="과학",
                title="식물세포 구조",
                error_type="암기 부족",
                image_url="https://images.unsplash.com/photo-1530026186672-2cd00ffc50fe?auto=format&fit=crop&w=420&q=80",
                review_date=date(2024, 5, 23),
                solved=False,
            ),
            WrongNote(
                subject="국어",
                title="비유적 표현",
                error_type="문제 해석",
                image_url="https://images.unsplash.com/photo-1455390582262-044cdead277a?auto=format&fit=crop&w=420&q=80",
                review_date=date(2024, 5, 22),
                solved=True,
            ),
        ]

        db.session.add_all(sample_notes)
        db.session.commit()
        return jsonify({"ok": True, "message": "seeded"})

    @app.get("/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "service": "WrongNoteFlask",
                "db_ready": bool(app.config.get("DB_READY")),
                "db_error": app.config.get("DB_INIT_ERROR"),
            }
        )

    with app.app_context():
        try:
            ensure_wrong_notes_schema()
            app.config["DB_READY"] = True
        except Exception as exc:
            app.config["DB_INIT_ERROR"] = f"DB initialization failed: {exc}"
            print(f"[WrongNoteFlask] DB initialization skipped: {exc}")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        debug=app.config.get("FLASK_DEBUG", True),
        host=app.config.get("FLASK_HOST", "0.0.0.0"),
        port=app.config.get("FLASK_PORT", 5003),
    )
