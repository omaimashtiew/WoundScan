from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import base64
import re
import colorsys
import logging
from scipy.spatial.distance import euclidean
from collections import Counter

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

# ====================================================================
# الألوان المرجعية الدقيقة (تم قياسها معملياً)
# ====================================================================
REFERENCE_COLORS = {
    "orange": {
        "name": "برتقالي",
        "hsv_range": {"h_min": 12, "h_max": 38, "s_min": 35, "s_max": 100, "v_min": 40, "v_max": 100},
        "lab_ref": [65, 35, 55],
        "rgb_ref": [255, 165, 50],
        "ph": "~5.0",
        "ph_label": "حمضي",
        "status": "normal",
        "level": "لا يوجد التهاب",
        "score": 8
    },
    "light_olive": {
        "name": "زيتي فاتح",
        "hsv_range": {"h_min": 38, "h_max": 70, "s_min": 15, "s_max": 50, "v_min": 55, "v_max": 90},
        "lab_ref": [75, -5, 20],
        "rgb_ref": [180, 170, 110],
        "ph": "~7.0",
        "ph_label": "متعادل",
        "status": "normal",
        "level": "الجرح سليم",
        "score": 5
    },
    "dark_olive": {
        "name": "زيتي غامق",
        "hsv_range": {"h_min": 38, "h_max": 70, "s_min": 30, "s_max": 70, "v_min": 25, "v_max": 55},
        "lab_ref": [45, -2, 25],
        "rgb_ref": [100, 95, 60],
        "ph": "~8.0",
        "ph_label": "قاعدي",
        "status": "warning",
        "level": "جرح ملتهب",
        "score": 60
    },
    "dark_green": {
        "name": "أخضر داكن",
        "hsv_range": {"h_min": 70, "h_max": 120, "s_min": 40, "s_max": 90, "v_min": 20, "v_max": 50},
        "lab_ref": [35, -15, 20],
        "rgb_ref": [60, 80, 50],
        "ph": "~9.0",
        "ph_label": "قاعدي شديد",
        "status": "critical",
        "level": "التهاب مزمن حاد",
        "score": 92
    }
}

# ====================================================================
# تحويل RGB إلى LAB (أكثر دقة من HSV)
# ====================================================================
def rgb_to_lab(rgb):
    """تحويل RGB إلى LAB color space (أكثر دقة للتمييز البصري)"""
    r, g, b = [x / 255.0 for x in rgb]
    
    # sRGB to XYZ
    r = r ** 2.2 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
    g = g ** 2.2 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
    b = b ** 2.2 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
    
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    # XYZ to LAB (D65 illuminant)
    x, y, z = x / 0.95047, y / 1.0, z / 1.08883
    x = x ** (1/3) if x > 0.008856 else (7.787 * x) + (16/116)
    y = y ** (1/3) if y > 0.008856 else (7.787 * y) + (16/116)
    z = z ** (1/3) if z > 0.008856 else (7.787 * z) + (16/116)
    
    L = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    
    return [L, a, b]

# ====================================================================
# معالجة الصورة متعددة المراحل (دقيقة جداً)
# ====================================================================
def preprocess_image_advanced(img):
    """معالجة متقدمة للصورة للحصول على أفضل دقة"""
    try:
        # تقليص الحجم لسرعة المعالجة مع الحفاظ على الدقة
        img.thumbnail((800, 800), Image.Resampling.LANCZOS)
        
        # تحسين التباين بشكل معتدل (لا نبالغ)
        img = ImageEnhance.Contrast(img).enhance(1.05)
        
        # معالجة توازن البياض التلقائي (Auto White Balance)
        if img.mode == 'RGB':
            pixels = np.array(img)
            r_avg, g_avg, b_avg = pixels[:,:,0].mean(), pixels[:,:,1].mean(), pixels[:,:,2].mean()
            if r_avg > 0 and g_avg > 0 and b_avg > 0:
                r_gain = g_avg / r_avg if r_avg > 0 else 1.0
                b_gain = g_avg / b_avg if b_avg > 0 else 1.0
                pixels[:,:,0] = np.clip(pixels[:,:,0] * r_gain, 0, 255)
                pixels[:,:,2] = np.clip(pixels[:,:,2] * b_gain, 0, 255)
                img = Image.fromarray(pixels.astype('uint8'))
        
        return img
    except Exception as e:
        logger.error(f"خطأ في معالجة الصورة: {e}")
        return img

def analyze_region_intelligently(region):
    """تحليل ذكي متعدد المقاييس لكل نقطة في المنطقة"""
    width, height = region.size
    pixels = list(region.getdata())
    
    # تخزين النتائج لكل بكسل
    classifications = []
    color_scores = {color: [] for color in REFERENCE_COLORS}
    
    for pixel in pixels:
        r, g, b = pixel[:3]
        
        # 1. تحليل HSV
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        hue = h * 360
        
        # 2. تحليل RGB ratio (مهم جداً للتمييز)
        rg_ratio = r / (g + 0.01)
        rb_ratio = r / (b + 0.01)
        gb_ratio = g / (b + 0.01)
        
        # 3. تحليل LAB
        try:
            lab = rgb_to_lab([r, g, b])
        except:
            lab = [0, 0, 0]
        
        # حساب درجة التشابه مع كل لون مرجعي
        best_color = None
        best_score = -1
        
        for color_key, ref in REFERENCE_COLORS.items():
            score = 0
            
            # مقياس 1: هل يقع ضمن نطاق HSV المرجعي؟
            h_range = ref["hsv_range"]
            if h_range["h_min"] <= hue <= h_range["h_max"]:
                if h_range["s_min"] <= (s*100) <= h_range["s_max"]:
                    if h_range["v_min"] <= (v*100) <= h_range["v_max"]:
                        score += 50  # نقاط كبيرة إذا كان ضمن النطاق تماماً
                    else:
                        # قريب من النطاق ولكنه خارج بقليل
                        v_diff = abs((v*100) - ((h_range["v_min"] + h_range["v_max"])/2))
                        if v_diff < 20:
                            score += 25
                else:
                    s_diff = abs((s*100) - ((h_range["s_min"] + h_range["s_max"])/2))
                    if s_diff < 20:
                        score += 15
            else:
                # قريب من النطاق في Hue
                h_center = (h_range["h_min"] + h_range["h_max"]) / 2
                h_diff = min(abs(hue - h_center), 360 - abs(hue - h_center))
                if h_diff < 25:
                    score += 20 - (h_diff * 0.8)
            
            # مقياس 2: نسب RGB (مهم جداً للتمييز بين الزيتي الفاتح والغامق)
            if color_key == "orange":
                if rg_ratio > 1.3 and rb_ratio > 1.2:
                    score += 20
            elif color_key == "light_olive":
                if 0.9 <= rg_ratio <= 1.2 and gb_ratio >= 0.9:
                    score += 20
            elif color_key == "dark_olive":
                if 0.7 <= rg_ratio <= 1.0 and v < 0.5:
                    score += 20
            elif color_key == "dark_green":
                if rg_ratio <= 0.85 and gb_ratio >= 0.9 and v < 0.5:
                    score += 25
            
            # مقياس 3: LAB (تمييز دقيق جداً بين الدرجات المتقاربة)
            lab_diff = euclidean(lab, ref["lab_ref"]) if len(lab) == 3 else 100
            if lab_diff < 30:
                score += (30 - lab_diff)
            
            if score > best_score:
                best_score = score
                best_color = color_key
        
        classifications.append(best_color)
        for color_key in REFERENCE_COLORS:
            if best_color == color_key:
                color_scores[color_key].append(best_score)
    
    return classifications, color_scores

def calculate_confidence(ratios, color_scores):
    """حساب مدى ثقة النتيجة"""
    winner = max(ratios, key=ratios.get)
    winner_ratio = ratios[winner]
    second_ratio = sorted(ratios.values(), reverse=True)[1] if len(ratios) > 1 else 0
    
    # الثقة تعتمد على الفرق بين اللون الفائز والثاني
    margin = winner_ratio - second_ratio
    if margin > 0.3:
        confidence = "عالية"
        confidence_value = 90
    elif margin > 0.15:
        confidence = "متوسطة"
        confidence_value = 70
    else:
        confidence = "منخفضة"
        confidence_value = 50
    
    # تحذير إضافي إذا كانت نسبة اللون الفائز قليلة جداً
    if winner_ratio < 0.3:
        confidence = "منخفضة جداً"
        confidence_value = 30
    
    return confidence, confidence_value, margin

def analyze_wound_color(image_bytes):
    try:
        img = None
        
        # محاولة فتح الصورة بعدة طرق
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
        
        # معالجة متقدمة للصورة
        img = preprocess_image_advanced(img)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # تحليل متعدد المناطق
        width, height = img.size
        cx, cy = width // 2, height // 2
        cs = min(width, height) // 3
        
        # المنطقة المركزية (الأهم)
        center_region = img.crop((cx - cs, cy - cs, cx + cs, cy + cs))
        center_region = center_region.resize((150, 150), Image.Resampling.LANCZOS)
        
        # مناطق إضافية للتأكد (الوسط + 4 حواف)
        regions = [center_region]
        offsets = [(0, -cs//2), (0, cs//2), (-cs//2, 0), (cs//2, 0)]
        for ox, oy in offsets:
            try:
                region = img.crop((cx - cs//2 + ox, cy - cs//2 + oy, cx + cs//2 + ox, cy + cs//2 + oy))
                region = region.resize((100, 100), Image.Resampling.LANCZOS)
                regions.append(region)
            except:
                pass
        
        # تحليل جميع المناطق وجمع النتائج
        all_classifications = []
        all_color_scores = {color: [] for color in REFERENCE_COLORS}
        
        for region in regions:
            classifications, color_scores = analyze_region_intelligently(region)
            all_classifications.extend(classifications)
            for k, v in color_scores.items():
                all_color_scores[k].extend(v)
        
        # حساب النسب النهائية
        total = len(all_classifications)
        counts = {k: all_classifications.count(k) for k in REFERENCE_COLORS}
        ratios = {k: counts[k] / total if total > 0 else 0 for k in counts}
        
        logger.info(f"نسب الألوان بعد التحليل المتقدم:")
        for k, v in ratios.items():
            logger.info(f"  {REFERENCE_COLORS[k]['name']}: {v:.3f}")
        
        # تحديد اللون الفائز
        winner = max(ratios, key=ratios.get)
        winner_ratio = ratios[winner]
        
        # حساب الثقة
        confidence, confidence_value, margin = calculate_confidence(ratios, all_color_scores)
        
        logger.info(f"النتيجة: {REFERENCE_COLORS[winner]['name']} (نسبة {winner_ratio:.1%}, ثقة {confidence})")
        
        # حساب متوسط الألوان للمنطقة المركزية (للRGB/HSV)
        arr = np.array(center_region.resize((50, 50)))
        avg_r = float(np.mean(arr[:, :, 0]))
        avg_g = float(np.mean(arr[:, :, 1]))
        avg_b = float(np.mean(arr[:, :, 2]))
        
        h_final, s_final, v_final = colorsys.rgb_to_hsv(avg_r/255, avg_g/255, avg_b/255)
        
        # بناء النتيجة
        ref = REFERENCE_COLORS[winner]
        
        # إضافة رسائل خاصة حسب الثقة
        recommendation = ref["recommendation"] if "recommendation" in ref else ""
        if confidence == "منخفضة" or confidence == "منخفضة جداً":
            recommendation += " (⚠️ دقة التحليل منخفضة بسبب تعدد الألوان أو سوء الإضاءة — يُنصح بإعادة التصوير باتباع التعليمات)"
        
        result = {
            "status": ref["status"],
            "level": ref["level"],
            "ph_range": ref["ph"],
            "ph_label": ref["ph_label"],
            "color_detected": ref["name"],
            "recommendation": recommendation,
            "emoji": "🟢" if ref["status"] == "normal" else ("🟡" if ref["status"] == "warning" else "🔴"),
            "score": ref["score"],
            "rgb": {"r": round(avg_r), "g": round(avg_g), "b": round(avg_b)},
            "hsv": {"h": round(h_final * 360, 1), "s": round(s_final * 100, 1), "v": round(v_final * 100, 1)},
            "confidence": confidence,
            "confidence_value": confidence_value,
            "color_ratios": {REFERENCE_COLORS[k]["name"]: round(v, 3) for k, v in ratios.items()}
        }
        
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