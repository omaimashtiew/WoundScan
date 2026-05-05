from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import logging
import cv2
from sklearn.cluster import KMeans
import colorsys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

app = Flask(__name__)
CORS(app)

# تعريف الألوان بأوزان دقيقة
COLOR_MAP = {
    "orange": {
        "name": "برتقالي",
        "hsv_center": [20, 80, 70],
        "ph": "~5.0",
        "ph_label": "حمضي",
        "status": "normal",
        "level": "لا يوجد التهاب",
        "score": 8,
        "recommendation": "اللون البرتقالي يدل على بيئة حمضية طبيعية. لا يوجد مؤشر على التهاب."
    },
    "light_olive": {
        "name": "زيتي فاتح",
        "hsv_center": [55, 35, 70],
        "ph": "~7.0",
        "ph_label": "متعادل",
        "status": "normal",
        "level": "الجرح سليم",
        "score": 5,
        "recommendation": "اللون الزيتي الفاتح يدل على بيئة متعادلة. الجرح في حالة طبيعية."
    },
    "dark_olive": {
        "name": "زيتي غامق",
        "hsv_center": [55, 60, 40],
        "ph": "~8.0",
        "ph_label": "قاعدي",
        "status": "warning",
        "level": "جرح ملتهب",
        "score": 60,
        "recommendation": "اللون الزيتي الغامق يشير إلى بيئة قاعدية واحتمال وجود التهاب. يُنصح باستشارة الطبيب."
    },
    "dark_green": {
        "name": "أخضر داكن",
        "hsv_center": [95, 70, 35],
        "ph": "~9.0",
        "ph_label": "قاعدي شديد",
        "status": "critical",
        "level": "التهاب مزمن حاد",
        "score": 92,
        "recommendation": "اللون الأخضر الداكن يدل على بيئة قاعدية شديدة. يستلزم تدخلاً طبياً فورياً."
    },
    "black": {
        "name": "أسود / غير واضح",
        "hsv_center": [0, 0, 10],
        "ph": "—",
        "ph_label": "غير معروف",
        "status": "unknown",
        "level": "الصورة غير واضحة",
        "score": 0,
        "recommendation": "الصورة معتمة جداً أو غير واضحة. يرجى التصوير بإضاءة كافية."
    }
}

def get_dominant_color_hsv(image_bytes):
    """استخراج اللون المهيمن باستخدام K-Means (سريع ودقيق)"""
    # تحويل الصورة إلى OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        # محاولة عن طريق PIL
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = pil_img.convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # تقليص الحجم لتسريع المعالجة
    height, width = img.shape[:2]
    new_size = (400, int(400 * height / width)) if width > 400 else (width, height)
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    
    # أخذ المنطقة المركزية فقط (مكان الجرح)
    h, w = img.shape[:2]
    center = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    
    # تحويل إلى RGB لسهولة التحليل
    center_rgb = cv2.cvtColor(center, cv2.COLOR_BGR2RGB)
    pixels = center_rgb.reshape(-1, 3)
    
    # استخدام K-Means لتجميع الألوان (3 ألوان مهيمنة كحد أقصى)
    if len(pixels) > 500:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        
        # حساب نسب كل لون
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(labels)
    else:
        # لو الصورة صغيرة جداً
        colors = [np.mean(pixels, axis=0).astype(int)]
        percentages = [1.0]
    
    # تحويل الألوان إلى HSV
    dominant_hsv = []
    for i, color in enumerate(colors):
        r, g, b = color
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        dominant_hsv.append({
            "rgb": [r, g, b],
            "hsv": [h*360, s*100, v*100],
            "percentage": percentages[i]
        })
    
    return dominant_hsv

def classify_color(hsv_values):
    """تصنيف اللون بناءً على أقرب مركز لوني"""
    h, s, v = hsv_values
    
    best_match = None
    best_distance = float('inf')
    
    for color_key, color_info in COLOR_MAP.items():
        center_h, center_s, center_v = color_info["hsv_center"]
        
        # حساب المسافة (Hue له وزن أكبر لأنه أهم للون)
        h_diff = min(abs(h - center_h), 360 - abs(h - center_h))
        s_diff = abs(s - center_s)
        v_diff = abs(v - center_v)
        
        # أوزان مختلفة: Hue 50%، Saturation 25%، Value 25%
        distance = (h_diff * 0.5) + (s_diff * 0.25) + (v_diff * 0.25)
        
        if distance < best_distance:
            best_distance = distance
            best_match = color_key
    
    # لو الأسود (قيمة V قليلة جداً)
    if v < 15:
        best_match = "black"
        best_match = "black"
    
    return best_match, best_distance

def analyze_wound_color(image_bytes):
    try:
        # استخراج الألوان المهيمنة
        dominant_colors = get_dominant_color_hsv(image_bytes)
        
        # اختيار اللون صاحب أكبر نسبة
        main_color = max(dominant_colors, key=lambda x: x["percentage"])
        h, s, v = main_color["hsv"]
        
        logger.info(f"اللون المهيمن: H={h:.1f}, S={s:.1f}, V={v:.1f}, نسبة={main_color['percentage']:.1%}")
        
        # تصنيف اللون
        color_key, confidence = classify_color([h, s, v])
        color_info = COLOR_MAP[color_key]
        
        # حساب مصفوفة النسبة المئوية للثقة (على أساس المسافة)
        confidence_percent = max(0, min(100, int(100 - (confidence * 1.5))))
        if confidence_percent < 30:
            confidence_percent = 30
        
        # بناء النتيجة
        result = {
            "status": color_info["status"],
            "level": color_info["level"],
            "ph_range": color_info["ph"],
            "ph_label": color_info["ph_label"],
            "color_detected": color_info["name"],
            "recommendation": color_info["recommendation"],
            "emoji": "🟢" if color_info["status"] == "normal" else ("🟡" if color_info["status"] == "warning" else ("🔴" if color_info["status"] == "critical" else "⚪")),
            "score": color_info["score"],
            "rgb": {"r": int(main_color["rgb"][0]), "g": int(main_color["rgb"][1]), "b": int(main_color["rgb"][2])},
            "hsv": {"h": round(h, 1), "s": round(s, 1), "v": round(v, 1)},
            "confidence": round(confidence_percent)
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
            "recommendation": f"حدث خطأ: {str(e)}. تأكد من الصورة.",
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