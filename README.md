# =====================================================================
# Logistics Priority & Pricing UI  — same UI, Perceptron-backed priority
# - Keeps all features: OSRM route + Folium, vouchers, size preview, QR
# - Uses Perceptron model if available, else falls back to rule-based
# =====================================================================
!pip -q install gradio==4.* geopy folium pandas numpy requests pillow scikit-learn joblib

import gradio as gr
import numpy as np, pandas as pd, re, requests, io, base64, os, joblib, random
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from functools import lru_cache
from PIL import Image
import folium

# ===================== ASSETS (giữ nguyên cách bạn đang dùng) =====================
LOGO_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
QR_B64   = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

def b64_to_pil(b64str: str) -> Image.Image:
    clean_b64 = re.sub(r'[^A-Za-z0-9+/=]', '', b64str)
    return Image.open(io.BytesIO(base64.b64decode(clean_b64))).convert("RGB")

def embedded_logo_html() -> str:
    # Bạn có thể thay HTML này cho hợp Branding LogiPriority
    return f"""
    <div style="display:flex;align-items:center;gap:12px;margin:8px 0 16px">
      <img src="data:image/png;base64,{LOGO_B64}" style="max-height:56px;border-radius:12px;box-shadow:0 6px 18px rgba(0,0,0,.08)"/>
      <div>
        <div style="font-size:28px;font-weight:900;letter-spacing:.3px">LogiPriority</div>
        <div style="color:#64748b">Smart routing • Pricing • Priority</div>
      </div>
    </div>
    """

EMBEDDED_QR = b64_to_pil(QR_B64)

# ===================== Pricing / Scoring (nguyên bản) =====================
TYPE_MULT   = {0:1.00, 1:1.25, 2:1.50}
CAT_MULT    = {"Clothes":1.05, "Food":1.10, "Electronics":1.15,
               "Documents":1.00, "Furniture":1.25, "Other":1.05}
SPEED_MULT  = {0:1.00, 1:1.12, 2:1.30}
SIZE_ADD    = {"S":0, "M":5_000, "L":10_000, "XL":15_000}

TYPE_SCORE  = {0:0.20, 1:0.60, 2:1.00}
SPEED_SCORE = {0:0.30, 1:0.60, 2:1.00}
PREF_SCORE  = {0:0.20, 1:0.60, 2:1.00}
CAT_SCORE   = {"Clothes":0.30, "Food":0.70, "Electronics":0.80,
               "Documents":0.20, "Furniture":0.70, "Other":0.40}

W_WEIGHT, W_DIST, W_TYPE, W_SPEED, W_URGENCY, W_PREF, W_CAT = 0.18, 0.18, 0.20, 0.12, 0.22, 0.05, 0.05
TH_LOW, TH_MED = 0.40, 0.70

VOUCHERS = {
    "— Không dùng —": ("none", 0),
    "GIAM50": ("percent", 50),
    "LANDAUTRAINGHIEM": ("minus", 30_000),
    "TETTRUNGTHU": ("minus", 15_000),
    "DUOI20TUOI": ("minus", 20_000),
}
def apply_voucher(price: float, code: str):
    t, v = VOUCHERS.get(code, ("none", 0))
    disc = price*(v/100.0) if t=="percent" else float(v) if t=="minus" else 0.0
    return max(0.0, price - disc), max(0.0, disc)

# ===================== Geocoding / Landmarks (nguyên bản) =====================
geocoder = Nominatim(user_agent="colab-priority-app", timeout=10)
LANDMARKS = {
    "— Không chọn —": "",
    "Chợ Bến Thành": "Ben Thanh Market, Ho Chi Minh City, Vietnam",
    "Nhà thờ Đức Bà": "Saigon Notre-Dame Cathedral Basilica, Ho Chi Minh City, Vietnam",
    "Bưu điện Trung tâm": "Saigon Central Post Office, Ho Chi Minh City, Vietnam",
    "Dinh Độc Lập": "Independence Palace, Ho Chi Minh City, Vietnam",
    "Bitexco Financial Tower": "Bitexco Financial Tower, Ho Chi Minh City, Vietnam",
    "Landmark 81": "Landmark 81, Ho Chi Minh City, Vietnam",
    "Nhà hát Thành phố": "Saigon Opera House, Ho Chi Minh City, Vietnam",
    "Bảo tàng Chứng tích Chiến tranh": "War Remnants Museum, Ho Chi Minh City, Vietnam",
    "Phố đi bộ Nguyễn Huệ": "Nguyen Hue Walking Street, Ho Chi Minh City, Vietnam",
    "Công viên Tao Đàn": "Tao Dan Park, Ho Chi Minh City, Vietnam",
    "KDL Suối Tiên": "Suoi Tien Theme Park, Ho Chi Minh City, Vietnam",
    "Công viên Đầm Sen": "Dam Sen Cultural Park, Ho Chi Minh City, Vietnam",
    "Sân bay Tân Sơn Nhất": "Tan Son Nhat International Airport, Ho Chi Minh City, Vietnam",
    "Thảo Cầm Viên": "Saigon Zoo and Botanical Gardens, Ho Chi Minh City, Vietnam",
    "Chợ Bình Tây (Chợ Lớn)": "Binh Tay Market, Ho Chi Minh City, Vietnam",
    "Bến Nhà Rồng": "Ho Chi Minh Museum Nha Rong Harbor, Ho Chi Minh City, Vietnam",
    "Khu CNC Sài Gòn (SHTP)": "Saigon Hi-Tech Park, Ho Chi Minh City, Vietnam",
    "UEH – Cơ sở B": "279 Nguyen Tri Phuong, District 10, Ho Chi Minh City, Vietnam",
    "UEH – Cơ sở N": "232/6 Vo Thi Sau, District 3, Ho Chi Minh City, Vietnam",
}

@lru_cache(maxsize=256)
def geocode_addr(addr: str):
    if not addr or not addr.strip(): return None
    try:
        loc = geocoder.geocode(addr.strip())
        if loc: return (float(loc.latitude), float(loc.longitude))
    except Exception:
        pass
    return None

def urgency_multiplier(hours: float) -> float:
    return 1.50 if hours<=3 else 1.35 if hours<=6 else 1.20 if hours<=12 else 1.10 if hours<=24 else 1.00
def preference_multiplier(pref: int) -> float:
    return {0:0.97, 1:1.00, 2:1.05}[int(pref)]
def vnd(x: float) -> str:
    return f"{int(round(max(0,x)/1000)*1000):,} VND".replace(",", ".")
def mask_phone(p: str) -> str:
    if not p: return ""
    digits = re.sub(r"\D", "", str(p))
    return digits[:3] + "*****" + digits[-2:] if len(digits) >= 5 else digits
def validate_phone(p: str) -> bool:
    if not p: return False
    digits = re.sub(r"\D", "", str(p))
    return len(digits) in (9,10,11)

# ===================== Size preview (SVG) =====================
DIM_CM = {"S": (20,15,10), "M": (30,20,15), "L": (40,30,20), "XL": (60,40,40)}
def size_svg(size="M"):
    L, W, H = DIM_CM[size]
    return f"""
    <div style="display:flex;gap:14px;align-items:center;padding:10px;border:1px solid #e5e7eb;border-radius:12px;background:#f8fafc">
      <svg width="110" height="80" viewBox="0 0 110 80">
        <rect x="20" y="25" width="70" height="40" fill="#dbeafe" stroke="#6366f1" stroke-width="2"/>
        <polygon points="20,25 35,15 105,15 90,25" fill="#bfdbfe" stroke="#6366f1" stroke-width="2"/>
        <polygon points="90,25 105,15 105,55 90,65" fill="#93c5fd" stroke="#6366f1" stroke-width="2"/>
      </svg>
      <div style="font-size:14px;color:#334155">
        <div><b>Size {size}</b></div>
        <div>Kích thước tham khảo: <b>{L} × {W} × {H} cm</b></div>
        <div>Gợi ý: S ~ phụ kiện nhỏ · M ~ quần áo · L ~ cồng kềnh · XL ~ thùng lớn</div>
      </div>
    </div>
    """

# ===================== ROUTING (OSRM) =====================
def get_route_osrm(p_coord, d_coord):
    try:
        url = f"https://router.project-osrm.org/route/v1/driving/{p_coord[1]},{p_coord[0]};{d_coord[1]},{d_coord[0]}?overview=full&geometries=geojson"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "routes" in data and data["routes"]:
            route = data["routes"][0]
            coords = route["geometry"]["coordinates"]
            poly = [[lat, lon] for lon, lat in coords]
            dist_km = route["distance"] / 1000.0
            return dist_km, poly
    except Exception:
        pass
    return None, None

def folium_map_html(p_coord, d_coord, polyline=None):
    if not (p_coord and d_coord):
        return "Chưa đủ toạ độ để vẽ bản đồ."
    m = folium.Map(location=[(p_coord[0]+d_coord[0])/2, (p_coord[1]+d_coord[1])/2],
                   zoom_start=12, tiles="OpenStreetMap")
    folium.Marker(p_coord, tooltip="Điểm nhận", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(d_coord, tooltip="Điểm giao",  icon=folium.Icon(color="red")).add_to(m)
    if polyline: folium.PolyLine(polyline, weight=5, color="blue").add_to(m)
    else:        folium.PolyLine([p_coord, d_coord], weight=3, color="gray", dash_array="5,8").add_to(m)
    m.fit_bounds([p_coord, d_coord])
    gmaps = f"https://www.google.com/maps/dir/{p_coord[0]},{p_coord[1]}/{d_coord[0]},{d_coord[1]}"
    return m._repr_html_() + f"<div style='margin-top:8px'><a href='{gmaps}' target='_blank'>Mở chỉ đường Google Maps</a></div>"

# ===================== Perceptron: train/load & predict =====================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Perceptron

MODEL_PATH = "/tmp/prio_perceptron.joblib" # Changed path
USE_PERCEPTRON = True   # ← Bật/tắt dùng ML mà không đổi UI

NUM_COLS = ["weight","distance_km","fragile","speed","urgency_h","preference"]
CAT_COLS = ["category","size"]

def _rule_based_level(weight, distance_km, fragile, speed, urgency_h, preference, category, size):
    # Score y như app hiện tại
    w_n = min(max(float(weight)/120.0, 0), 1)
    d_n = min(max(float(distance_km)/400.0, 0), 1)
    t_n, s_n = TYPE_SCORE[int(fragile)], SPEED_SCORE[int(speed)]
    u_n = min(max((72.0-float(urgency_h))/72.0, 0), 1)
    p_n, c_n = PREF_SCORE[int(preference)], CAT_SCORE[category]
    score = (W_WEIGHT*w_n + W_DIST*d_n + W_TYPE*t_n + W_SPEED*s_n + W_URGENCY*u_n + W_PREF*p_n + W_CAT*c_n)
    if score < TH_LOW: return "Low"
    elif score < TH_MED: return "Medium"
    else: return "High"

def train_or_load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # ---------- DEMO TRAIN (synthetic) ----------
    # Sinh dữ liệu ngẫu nhiên + gắn nhãn theo quy tắc hiện tại
    N = 4000
    rng = np.random.default_rng(42)
    cats = list(CAT_SCORE.keys())
    sizes = list(SIZE_ADD.keys())
    rows = []
    labels = []
    for _ in range(N):
        weight = float(rng.integers(0, 121))
        distance_km = float(rng.integers(0, 401))
        fragile = int(rng.integers(0, 3))
        speed = int(rng.integers(0, 3))
        urgency_h = float(rng.integers(0, 73))
        preference = int(rng.integers(0, 3))
        category = random.choice(cats)
        size = random.choice(sizes)
        rows.append([weight,distance_km,fragile,speed,urgency_h,preference,category,size])
        labels.append(_rule_based_level(weight,distance_km,fragile,speed,urgency_h,preference,category,size))

    df = pd.DataFrame(rows, columns=NUM_COLS+CAT_COLS)
    y  = np.array(labels)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )
    clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    pipe = Pipeline([("prep", pre), ("clf", clf)])
    pipe.fit(df, y)
    joblib.dump(pipe, MODEL_PATH)
    return pipe

MODEL = train_or_load_model() if USE_PERCEPTRON else None

def predict_priority_with_model(distance_km, weight, fragile, speed, urgency_h, preference, category, size):
    if MODEL is None:
        return None  # signal to fall back
    X = pd.DataFrame([{
        "weight": float(weight),
        "distance_km": float(distance_km),
        "fragile": int(fragile),
        "speed": int(speed),
        "urgency_h": float(urgency_h),
        "preference": int(preference),
        "category": category,
        "size": size,
    }], columns=NUM_COLS+CAT_COLS)
    return MODEL.predict(X)[0]

# ===================== Core compute (UI giữ nguyên) =====================
def compute(weight, fragile, category, size, speed, urgency_h, preference, voucher_code,
            sender_name, sender_phone, sender_address, sender_landmark,
            recv_name, recv_phone, recv_address, recv_landmark,
            manual_distance):
    notes = []

    # Geocode
    p_coord = geocode_addr(LANDMARKS.get(sender_landmark, "") or "") or geocode_addr(sender_address or "")
    d_coord = geocode_addr(LANDMARKS.get(recv_landmark, "") or "") or geocode_addr(recv_address or "")
    if not p_coord: notes.append("Không geocode được điểm nhận (địa danh/địa chỉ).")
    if not d_coord: notes.append("Không geocode được điểm giao (địa danh/địa chỉ).")

    # Distance
    polyline = None
    if p_coord and d_coord:
        route_km, polyline = get_route_osrm(p_coord, d_coord)
        if route_km is not None:
            distance_km = round(route_km); dist_src = "road routing (OSRM)"
        else:
            distance_km = round(geodesic(p_coord, d_coord).km); dist_src = "geodesic (fallback)"
            notes.append("OSRM routing lỗi/giới hạn – dùng khoảng cách địa lý tạm thời.")
    else:
        distance_km = float(manual_distance or 0); dist_src = "manual override"

    # Pricing
    base = 15_000; per_km = 3_500
    distance_cost = distance_km * per_km
    weight_fee = max(0.0, float(weight) - 3.0) * 3_000.0
    size_fee = SIZE_ADD[size]
    subtotal = (base + distance_cost + weight_fee + size_fee)
    subtotal *= TYPE_MULT[int(fragile)] * CAT_MULT[category] * SPEED_MULT[int(speed)]
    subtotal *= urgency_multiplier(float(urgency_h)) * preference_multiplier(int(preference))
    final_price, discount = apply_voucher(subtotal, voucher_code)

    # Priority (Model -> fallback to rule)
    level_model = predict_priority_with_model(distance_km, weight, fragile, speed, urgency_h, preference, category, size)
    level_rule  = _rule_based_level(weight, distance_km, fragile, speed, urgency_h, preference, category, size)
    level = level_model or level_rule
    color = {"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444"}[level]

    # Validations
    if not sender_name: notes.append("Chưa nhập tên người gửi.")
    if not recv_name:   notes.append("Chưa nhập tên người nhận.")
    if not validate_phone(sender_phone): notes.append("SĐT người gửi không hợp lệ (9–11 số).")
    if not validate_phone(recv_phone):   notes.append("SĐT người nhận không hợp lệ (9–11 số).")

    # UI text (nguyên layout)
    chip = f"<div style='display:inline-block;padding:.65rem 1.1rem;border-radius:9999px;background:{color};color:#fff;font-weight:800'>PRIORITY: {level}</div>"
    price_html = (
        f"<div style='display:inline-block;margin-left:12px;padding:.55rem .9rem;border-radius:9999px;background:#334155;color:#fff;font-weight:800'>"
        f"SUBTOTAL: {vnd(subtotal)}</div>"
        f"<div style='display:inline-block;margin-left:8px;padding:.65rem 1.1rem;border-radius:9999px;background:#111827;color:#fff;font-weight:800'>"
        f"FINAL: {vnd(final_price)}</div>"
    )
    who = f"<div style='color:#334155;margin-top:.5rem;font-size:14px'><b>Sender:</b> {sender_name or '-'} • {mask_phone(sender_phone)}<br><b>Receiver:</b> {recv_name or '-'} • {mask_phone(recv_phone)}</div>"
    meta = (f"<div style='font-size:14px;color:#334155;margin-top:.4rem'>Distance: <b>{distance_km} km</b> ({dist_src}). "
            f"Weight: <b>{float(weight):.0f} kg</b>; Fragile: <b>{int(fragile)}</b>; Size: <b>{size}</b>; "
            f"Speed: <b>{int(speed)}</b>; Urgency: <b>{int(urgency_h)} h</b>; Pref: <b>{int(preference)}</b>; Category: <b>{category}</b>. "
            f"Voucher: <b>{voucher_code}</b> (tiết kiệm {vnd(discount)}). "
            + (f"Priority (ML): <b>{level_model}</b> • Rule: <b>{level_rule}</b>" if level_model else f"Priority (Rule): <b>{level_rule}</b>")
            + "</div>")
    warn_html = "" if not notes else "<div style='margin-top:6px;color:#b91c1c'>" + "<br>".join("⚠️ "+n for n in notes) + "</div>"

    breakdown = (f"<div style='margin-top:10px;padding:10px;border:1px solid #e5e7eb;border-radius:12px;background:#f8fafc'>"
                 f"<div style='font-weight:700;margin-bottom:6px;color:#334155'>Breakdown</div>"
                 f"<div style='display:flex;gap:16px;flex-wrap:wrap;color:#334155;font-size:14px'>"
                 f"<div>Base: {vnd(15_000)}</div>"
                 f"<div>Distance: {distance_km} km × 3.500 = {vnd(distance_cost)}</div>"
                 f"<div>Weight fee (&gt;3kg): {vnd(weight_fee)}</div>"
                 f"<div>Size add ({size}): {vnd(size_fee)}</div>"
                 f"<div>Multipliers: type×...×pref</div>"
                 f"<div>Voucher {voucher_code}: -{vnd(discount)}</div>"
                 f"</div>{warn_html}</div>")

    header = embedded_logo_html()
    map_html = folium_map_html(p_coord, d_coord, polyline)
    return header + chip + price_html + who + meta + breakdown, distance_km, level, int(round(final_price)), map_html

def update_size_preview(size):
    return size_svg(size)

# ===================== UI (giữ nguyên) =====================
theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate").set()

with gr.Blocks(theme=theme, css="""
.gradio-container {background: linear-gradient(180deg,#f8fafc,#ffffff);}
#card {max-width: 1100px; margin: 0 auto;}
""") as demo:
    with gr.Column(elem_id="card"):
        gr.HTML(embedded_logo_html())
        gr.Markdown("Định tuyến OSRM • Preview size • Voucher • QR thanh toán tích hợp sẵn.")

        with gr.Row():
            with gr.Column(scale=6):
                gr.Markdown("### Người gửi")
                sender_name  = gr.Textbox(label="Tên người gửi")
                sender_phone = gr.Textbox(label="SĐT người gửi", placeholder="0903xxxxxx")
                with gr.Row():
                    sender_addr = gr.Textbox(label="Địa chỉ nhận (tìm kiếm)", scale=3)
                    sender_land = gr.Dropdown(list(LANDMARKS.keys()), value="— Không chọn —", label="Hoặc chọn địa danh", scale=2)

                gr.Markdown("### Người nhận")
                recv_name  = gr.Textbox(label="Tên người nhận")
                recv_phone = gr.Textbox(label="SĐT người nhận", placeholder="0988xxxxxx")
                with gr.Row():
                    recv_addr = gr.Textbox(label="Địa chỉ giao (tìm kiếm)", scale=3)
                    recv_land = gr.Dropdown(list(LANDMARKS.keys()), value="— Không chọn —", label="Hoặc chọn địa danh", scale=2)

                gr.Markdown("### Hàng hoá & Tuỳ chọn")
                weight   = gr.Slider(0, 120, value=20, step=1, label="Khối lượng (kg)")
                fragile  = gr.Slider(0, 2, value=0, step=1, label="Fragility/Value (0=Normal, 1=Fragile, 2=High value)")
                category = gr.Dropdown(["Clothes","Food","Electronics","Documents","Furniture","Other"],
                                       value="Clothes", label="Loại hàng hoá")
                with gr.Row():
                    size = gr.Radio(["S","M","L","XL"], value="M", label="Kích cỡ", scale=1)
                    size_preview = gr.HTML(value=size_svg("M"))
                size.change(update_size_preview, inputs=size, outputs=size_preview)

                with gr.Row():
                    speed      = gr.Slider(0, 2, value=1, step=1, label="Tốc độ (0=Economy,1=Standard,2=Express)")
                    urgency_h  = gr.Slider(0, 72, value=24, step=1, label="Mức gấp (giờ đến deadline)")
                    preference = gr.Slider(0, 2, value=1, step=1, label="Ưu tiên của khách (0=Chi phí,1=Cân bằng,2=Tốc độ)")
                voucher = gr.Dropdown(list(VOUCHERS.keys()), value="— Không dùng —", label="Voucher giảm giá")

                with gr.Accordion("⚙️ Nâng cao: khoảng cách thủ công (fallback)", open=False):
                    manual_distance = gr.Slider(0, 1000, value=0, step=1, label="Manual distance (km)")
                btn = gr.Button("Tính giá & ưu tiên", variant="primary")

            with gr.Column(scale=5):
                result_html = gr.HTML()
                dist_used   = gr.Number(label="Khoảng cách (km)", interactive=False, precision=0)
                priority    = gr.Textbox(label="Mức độ ưu tiên", interactive=False)
                price_num   = gr.Number(label="Giá cước (VND, sau voucher)", interactive=False, precision=0)
                map_html    = gr.HTML(label="Bản đồ & Chỉ đường")
                gr.Image(value=EMBEDDED_QR, label="Quét QR để thanh toán", interactive=False, show_download_button=False)

        btn.click(
            fn=compute,
            inputs=[weight, fragile, category, size, speed, urgency_h, preference,
                    voucher, sender_name, sender_phone, sender_addr, sender_land,
                    recv_name, recv_phone, recv_addr, recv_land, manual_distance],
            outputs=[result_html, dist_used, priority, price_num, map_html]
        )

demo.launch()
