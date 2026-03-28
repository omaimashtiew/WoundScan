from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import base64
import re
import colorsys
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
    logger.info("✅ دعم تنسيق HEIC مفعل")
except ImportError:
    HEIC_SUPPORT = False
    logger.warning("⚠️ تنسيق HEIC غير مدعوم")

app = Flask(__name__)
CORS(app)


def preprocess_image(img):
    try:
        img.thumbnail((600, 600), Image.Resampling.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.1)
        img = ImageEnhance.Sharpness(img).enhance(1.1)
        return img
    except Exception as e:
        logger.error(f"خطأ في معالجة الصورة: {e}")
        return img


def analyze_wound_color(image_bytes):
    try:
        img = None

        try:
            img = Image.open(io.BytesIO(image_bytes))
            logger.info("تم فتح الصورة بنجاح")
        except Exception as e:
            logger.error(f"فشل فتح الصورة: {e}")
            try:
                if isinstance(image_bytes, str):
                    image_bytes = image_bytes.encode('utf-8')
                img_data = re.search(b'base64,(.*)', image_bytes)
                if img_data:
                    image_bytes = base64.b64decode(img_data.group(1))
                    img = Image.open(io.BytesIO(image_bytes))
            except:
                pass

        if img is None:
            raise Exception("لا يمكن قراءة الصورة")

        img = preprocess_image(img)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        width, height = img.size
        cx, cy = width // 2, height // 2
        cs = min(width, height) // 3

        region = img.crop((
            max(0, cx - cs), max(0, cy - cs),
            min(width, cx + cs), min(height, cy + cs)
        ))
        region = region.resize((100, 100))
        arr = np.array(region)

        avg_r = float(np.mean(arr[:, :, 0]))
        avg_g = float(np.mean(arr[:, :, 1]))
        avg_b = float(np.mean(arr[:, :, 2]))

        h, s, v = colorsys.rgb_to_hsv(avg_r / 255, avg_g / 255, avg_b / 255)
        hue = h * 360
        sat = s * 100
        val = v * 100

        rg_ratio = avg_r / avg_g if avg_g > 0 else 1.0

        logger.info(f"RGB=({avg_r:.0f},{avg_g:.0f},{avg_b:.0f}) "
                    f"HSV=({hue:.1f}deg,{sat:.1f}%,{val:.1f}%) "
                    f"R/G={rg_ratio:.3f}")

        # ====================================================================
        # نظام النقاط المُصحَّح
        # ====================================================================

        score_dark_green = 0   # أخضر داكن  → التهاب مزمن حاد (pH ~9)
        score_dark_olive  = 0  # زيتي غامق  → جرح ملتهب (pH ~8)
        score_light_olive = 0  # زيتي فاتح  → متعادل / سليم (pH ~7)
        score_orange      = 0  # برتقالي    → حمضي / لا التهاب (pH ~5)

        # ── 1. أخضر داكن ──────────────────────────────────────────────────
        # hue عالٍ جداً (>90)، تشبع منخفض جداً، R/G أقل من 0.95
        if hue > 90:
            score_dark_green += 70
        if rg_ratio < 0.95:
            score_dark_green += 55
        if sat < 8:
            score_dark_green += 35
        if val < 40:
            score_dark_green += 25

        # ── 2. زيتي غامق (ملتهب) ─────────────────────────────────────────
        # hue متوسط (50-90)، تشبع منخفض جداً (<12%)، سطوع منخفض-متوسط (<55)
        if 50 <= hue <= 90:
            score_dark_olive += 35
        if sat < 12:                      # ← المفتاح الرئيسي للزيتي الغامق
            score_dark_olive += 60
        if 0.92 <= rg_ratio <= 1.08:
            score_dark_olive += 20
        if val < 55:
            score_dark_olive += 25

        # منع الزيتي الغامق من الفوز على الزيتي الفاتح عندما السطوع عالٍ
        if val > 60:
            score_dark_olive -= 40

        # ── 3. زيتي فاتح (سليم / متعادل) ───────────────────────────────
        # hue متوسط (50-90)، تشبع متوسط (12-40%)، سطوع عالٍ (>55)
        if 50 <= hue <= 90:
            score_light_olive += 30
        if 12 <= sat <= 40:               # ← المفتاح الرئيسي للزيتي الفاتح
            score_light_olive += 65
        if 1.00 <= rg_ratio <= 1.25:
            score_light_olive += 20
        if val > 55:                      # ← سطوع عالٍ يميّزه عن الغامق
            score_light_olive += 40
        if sat > 20:                      # تشبع واضح يقوّي التصنيف
            score_light_olive += 15

        # ── 4. برتقالي (حمضي) ────────────────────────────────────────────
        # hue منخفض (<50)، R/G مرتفع (>1.25)، تشبع مرتفع (>30)
        if rg_ratio > 1.30:
            score_orange += 65
        if hue < 45:
            score_orange += 50
        if sat > 30:
            score_orange += 25
        if 25 <= hue <= 45:
            score_orange += 20

        logger.info(f"النقاط: أخضر داكن={score_dark_green}, زيتي غامق={score_dark_olive}, "
                    f"زيتي فاتح={score_light_olive}, برتقالي={score_orange}")

        scores = {
            "dark_green":  score_dark_green,
            "dark_olive":  score_dark_olive,
            "light_olive": score_light_olive,
            "orange":      score_orange,
        }

        winner = max(scores, key=scores.get)
        max_score = max(scores.values())

        # حالة طوارئ: إذا كانت جميع النقاط صفر
        if max_score == 0:
            if rg_ratio > 1.30 or hue < 40:
                winner = "orange"
            elif hue > 90 or rg_ratio < 0.95:
                winner = "dark_green"
            elif sat < 12 and val < 55:
                winner = "dark_olive"
            else:
                winner = "light_olive"

        # ── إرجاع النتيجة ────────────────────────────────────────────────
        if winner == "dark_green":
            result = {
                "status": "critical",
                "level": "التهاب مزمن حاد",
                "ph_range": "~9.0",
                "ph_label": "قاعدي شديد",
                "color_detected": "أخضر داكن",
                "recommendation": (
                    "اللون الأخضر الداكن يدل على بيئة قاعدية شديدة. "
                    "يستلزم هذا تدخلاً طبياً فورياً ومراجعة متخصص."
                ),
                "emoji": "🔴",
                "score": 92,
            }
        elif winner == "dark_olive":
            result = {
                "status": "warning",
                "level": "جرح ملتهب",
                "ph_range": "~8.0",
                "ph_label": "قاعدي",
                "color_detected": "زيتي غامق",
                "recommendation": (
                    "اللون الزيتي الغامق يشير إلى بيئة قاعدية واحتمال وجود التهاب. "
                    "يُنصح باستشارة الطاقم الطبي لتقييم حالة الجرح."
                ),
                "emoji": "🟡",
                "score": 60,
            }
        elif winner == "orange":
            result = {
                "status": "normal",
                "level": "لا يوجد التهاب",
                "ph_range": "~5.0",
                "ph_label": "حمضي",
                "color_detected": "برتقالي",
                "recommendation": (
                    "اللون البرتقالي يدل على بيئة حمضية طبيعية. "
                    "لا يوجد مؤشر على التهاب — يُستخدم كمرجع للمقارنة."
                ),
                "emoji": "🟢",
                "score": 8,
            }
        else:  # light_olive
            result = {
                "status": "normal",
                "level": "الجرح سليم",
                "ph_range": "~7.0",
                "ph_label": "متعادل",
                "color_detected": "زيتي فاتح",
                "recommendation": (
                    "اللون الزيتي الفاتح يدل على بيئة متعادلة. "
                    "لا يوجد مؤشر على التهاب — الجرح في حالة طبيعية."
                ),
                "emoji": "🟢",
                "score": 5,
            }

        result["rgb"] = {"r": round(avg_r), "g": round(avg_g), "b": round(avg_b)}
        result["hsv"] = {"h": round(hue, 1), "s": round(sat, 1), "v": round(val, 1)}
        return result

    except Exception as e:
        logger.error(f"خطأ في تحليل الصورة: {e}")
        return {
            "status": "unknown",
            "level": "خطأ في قراءة الصورة",
            "ph_range": "—",
            "ph_label": "غير معروف",
            "color_detected": "خطأ",
            "recommendation": f"حدث خطأ: {str(e)}. تأكد من الصيغة المدعومة.",
            "emoji": "❌",
            "score": 0,
            "rgb": {"r": 0, "g": 0, "b": 0},
            "hsv": {"h": 0, "s": 0, "v": 0},
        }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "image" not in request.files:
            return jsonify({"error": "لم يتم إرسال صورة"}), 400

        file = request.files["image"]

        if file.filename == '':
            return jsonify({"error": "لم يتم اختيار صورة"}), 400

        image_bytes = file.read()

        if len(image_bytes) == 0:
            return jsonify({"error": "الملف فارغ"}), 400

        if len(image_bytes) > 5 * 1024 * 1024:
            return jsonify({"error": "الصورة كبيرة جداً. الحد الأقصى 5 ميجابايت"}), 400

        result = analyze_wound_color(image_bytes)
        return jsonify(result)

    except Exception as e:
        logger.error(f"خطأ في المعالجة: {e}")
        return jsonify({"error": f"حدث خطأ: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)