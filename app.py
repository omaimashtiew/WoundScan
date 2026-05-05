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
        pixels = list(region.getdata())

        # ====================================================================
        # تحليل كل بكسل على حدة وعدّ النسب
        # ====================================================================

        counts = {
            "orange":      0,   # برتقالي   → حمضي / لا التهاب (pH ~5)
            "light_olive": 0,   # زيتي فاتح → متعادل / سليم   (pH ~7)
            "dark_olive":  0,   # زيتي غامق → جرح ملتهب       (pH ~8)
            "dark_green":  0,   # أخضر داكن → التهاب مزمن حاد (pH ~9)
        }

        for pr, pg, pb in pixels:
            h, s, v = colorsys.rgb_to_hsv(pr / 255, pg / 255, pb / 255)
            hue = h * 360

            # 🟠 برتقالي
            if 15 <= hue <= 35:
                counts["orange"] += 1

            # 🫒 زيتي فاتح / غامق
            elif 40 <= hue <= 70:
                if s < 0.5:
                    if v > 0.6:
                        counts["light_olive"] += 1
                    else:
                        counts["dark_olive"] += 1

            # 🌲 أخضر داكن
            elif 70 <= hue <= 140 and v < 0.5:
                counts["dark_green"] += 1

        total = len(pixels)
        ratios = {k: counts[k] / total for k in counts}

        logger.info(f"نسب الألوان: برتقالي={ratios['orange']:.3f}, "
                    f"زيتي فاتح={ratios['light_olive']:.3f}, "
                    f"زيتي غامق={ratios['dark_olive']:.3f}, "
                    f"أخضر داكن={ratios['dark_green']:.3f}")

        winner = max(ratios, key=ratios.get)
        max_ratio = ratios[winner]

        # حساب RGB تقريبي للمنطقة (للإبلاغ فقط)
        arr = np.array(region)
        avg_r = float(np.mean(arr[:, :, 0]))
        avg_g = float(np.mean(arr[:, :, 1]))
        avg_b = float(np.mean(arr[:, :, 2]))

        # حالة طوارئ: إذا لم يُصنَّف أي بكسل
        if max_ratio == 0:
            h_avg, s_avg, v_avg = colorsys.rgb_to_hsv(avg_r / 255, avg_g / 255, avg_b / 255)
            hue_avg = h_avg * 360
            rg_ratio = avg_r / avg_g if avg_g > 0 else 1.0
            if rg_ratio > 1.30 or hue_avg < 40:
                winner = "orange"
            elif hue_avg > 90 or rg_ratio < 0.95:
                winner = "dark_green"
            elif s_avg * 100 < 12 and v_avg * 100 < 55:
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

        h_final, s_final, v_final = colorsys.rgb_to_hsv(avg_r / 255, avg_g / 255, avg_b / 255)
        result["rgb"] = {"r": round(avg_r), "g": round(avg_g), "b": round(avg_b)}
        result["hsv"] = {"h": round(h_final * 360, 1), "s": round(s_final * 100, 1), "v": round(v_final * 100, 1)}
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