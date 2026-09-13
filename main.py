import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from typing import Dict, Any, List
from supabase import create_client, Client
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="NCD Progression AI Service")

# 1. เชื่อมต่อ Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ เชื่อมต่อ Supabase สำเร็จ")
    except Exception as e:
        print(f"⚠️ ไม่สามารถเชื่อมต่อ Supabase ได้: {e}")

# 2. โหลดโมเดล
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ncd_progression_xgb.joblib")
FEATURE_PATH = os.path.join(BASE_DIR, "model", "progression_features.joblib")

model = None
feature_cols = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_PATH):
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURE_PATH)
        print("✅ โหลดโมเดล XGBoost และ Features สำเร็จ")
    else:
        print("❌ ไม่พบไฟล์โมเดลในโฟลเดอร์ model/")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดขณะโหลดไฟล์โมเดล: {e}")

# ตัวติดตามสถานะการประมวลผล Batch
batch_tracker = {
    "is_running": False,
    "status": "idle",
    "total": 0,
    "processed": 0,
    "errors": 0,
    "message": "ยังไม่มีการประมวลผล Batch ในขณะนี้"
}

def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        s = str(val).strip().replace(",", "")
        if not s or s in ["-", "null", "None", "undefined"]:
            return default
        if "/" in s:
            s = s.split("/")[0].strip()
        return float(s)
    except Exception:
        return default

def calculate_prediction(record: Dict[str, Any]):
    age = safe_float(record.get("age"))
    h = safe_float(record.get("height"))
    w = safe_float(record.get("weight"))
    waist = safe_float(record.get("waist"))
    
    bmi = round(w / ((h / 100) ** 2), 2) if h > 0 else 0.0
    whtr = round(waist / h, 2) if h > 0 else 0.0

    sys = safe_float(record.get("sys"))
    dia = safe_float(record.get("dia"))
    bp = str(record.get("bp") or "").strip()
    
    if (sys == 0 or dia == 0) and "/" in bp:
        parts = bp.split("/")
        if len(parts) >= 2:
            sys = safe_float(parts[0])
            dia = safe_float(parts[1])

    pulse_pressure = sys - dia
    map_pressure = round(dia + (pulse_pressure / 3.0), 2)
    sugar = safe_float(record.get("sugar"))

    gender_code = 1 if str(record.get("gender") or "").strip() == "ชาย" else 0
    fasting_code = 1 if str(record.get("fasting") or "").strip().lower() in ["yes", "1", "true"] else 0

    sm = str(record.get("smoking") or "")
    if ("สูบ" in sm or "ประจำ" in sm) and "ไม่" not in sm and "เลิก" not in sm:
        smoking_code = 2
    elif "เลิก" in sm:
        smoking_code = 1
    else:
        smoking_code = 0

    al = str(record.get("alcohol") or "")
    if "3 เดือน" in al and "มากกว่า" not in al:
        alcohol_code = 2
    elif "มากกว่า 3" in al:
        alcohol_code = 1
    else:
        alcohol_code = 0

    fm = str(record.get("family") or "")
    if "ทั้งสอง" in fm:
        family_code = 3
    elif "เบาหวาน" in fm:
        family_code = 2
    elif "ความดัน" in fm:
        family_code = 1
    else:
        family_code = 0

    input_dict = {
        "age": age,
        "gender_code": gender_code,
        "bmi": bmi,
        "waist": waist,
        "whtr": whtr,
        "baseline_sys": sys,
        "baseline_dia": dia,
        "pulse_pressure": pulse_pressure,
        "mean_arterial_pressure": map_pressure,
        "baseline_sugar": sugar,
        "fasting_code": fasting_code,
        "smoking_code": smoking_code,
        "alcohol_code": alcohol_code,
        "family_code": family_code
    }
    
    input_df = pd.DataFrame([input_dict])[feature_cols]
    risk_probability = float(model.predict_proba(input_df)[:, 1][0]) * 100

    if risk_probability >= 70.0:
        tier = "วิกฤต/เสี่ยงสูงมาก (High Risk)"
    elif risk_probability >= 40.0:
        tier = "เฝ้าระวัง/เสี่ยงปานกลาง (Moderate Risk)"
    else:
        tier = "ความเสี่ยงต่ำ (Low Risk)"

    return round(risk_probability, 2), tier

def process_and_predict(record: Dict[str, Any]):
    if not model or not supabase:
        return
    try:
        rec_id = record.get("id")
        if not rec_id:
            return
        score, tier = calculate_prediction(record)
        supabase.table("records").update({
            "ai_risk_score": score,
            "ai_risk_tier": tier,
            "ai_predicted_at": "now()"
        }).eq("id", rec_id).execute()
        print(f"✅ บันทึกสำเร็จ {rec_id}: {score}% ({tier})")
    except Exception as e:
        print(f"❌ Error processing record: {e}")

def run_batch_worker(targets: List[Dict[str, Any]]):
    """ฟังก์ชันเบื้องหลังประมวลผลคู่ขนาน 15 Threads"""
    global batch_tracker
    batch_tracker["is_running"] = True
    batch_tracker["status"] = "running"
    batch_tracker["total"] = len(targets)
    batch_tracker["processed"] = 0
    batch_tracker["errors"] = 0
    batch_tracker["message"] = f"กำลังประมวลผล {len(targets)} รายการ..."

    def update_single(r):
        rec_id = r.get("id")
        if not rec_id:
            return False
        try:
            score, tier = calculate_prediction(r)
            supabase.table("records").update({
                "ai_risk_score": score,
                "ai_risk_tier": tier,
                "ai_predicted_at": "now()"
            }).eq("id", rec_id).execute()
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(update_single, targets))

    success_count = sum(1 for res in results if res)
    error_count = len(results) - success_count

    batch_tracker["is_running"] = False
    batch_tracker["status"] = "completed"
    batch_tracker["processed"] = success_count
    batch_tracker["errors"] = error_count
    batch_tracker["message"] = f"ประมวลผลเสร็จสิ้น {success_count} รายการ (ข้อผิดพลาด {error_count} รายการ)"
    print(f"🎉 Batch Process Completed: {success_count}/{len(targets)}")

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "supabase_connected": supabase is not None
    }

@app.post("/webhook/predict")
async def supabase_webhook(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    event_type = payload.get("type")
    record = payload.get("record")
    old_record = payload.get("old_record")

    if not record:
        raise HTTPException(status_code=400, detail="No record found in payload")

    if event_type == "UPDATE" and old_record:
        clinical_keys = ["age", "weight", "height", "waist", "sys", "dia", "bp", "sugar", "fasting", "smoking", "alcohol", "family", "gender"]
        has_clinical_change = any(str(record.get(k) or "").strip() != str(old_record.get(k) or "").strip() for k in clinical_keys)
        if not has_clinical_change:
            return {"status": "skipped", "message": "ข้ามการทำงาน: อัปเดตจาก AI"}

    background_tasks.add_task(process_and_predict, record)
    return {"status": "queued", "record_id": record.get("id")}

# ====================================================================
# ⚡ จุดประมวลผลย้อนหลัง (ความเร็วสูง + ไม่ติด Timeout)
# ====================================================================
@app.get("/batch/run")
@app.post("/batch/run")
def batch_run_prediction(force_all: bool = False, background_tasks: BackgroundTasks = None):
    global batch_tracker
    if batch_tracker["is_running"]:
        return {
            "status": "already_running",
            "message": "ระบบกำลังประมวลผล Batch อยู่ในขณะนี้ กรุณารอสักครู่",
            "progress": f"{batch_tracker['processed']}/{batch_tracker['total']}"
        }

    if not model or not supabase:
        raise HTTPException(status_code=500, detail="โมเดลหรือ Supabase ยังไม่พร้อมใช้งาน")

    try:
        res = supabase.table("records").select("*").limit(5000).execute()
        all_records = res.data or []

        def is_empty_score(val):
            if val is None:
                return True
            s = str(val).strip()
            return s in ["", "-", "null", "None"]

        targets = all_records if force_all else [r for r in all_records if is_empty_score(r.get("ai_risk_score"))]

        if not targets:
            return {
                "status": "success",
                "message": "ไม่มีรายการที่ต้องประมวลผล (มีคะแนนครบถ้วนแล้ว)",
                "total_in_db": len(all_records),
                "target_count": 0
            }

        # ส่งไปทำงานเบื้องหลังทันที ไม่รอให้เบราว์เซอร์ค้าง
        background_tasks.add_task(run_batch_worker, targets)

        return {
            "status": "started",
            "message": f"ระบบเริ่มประมวลผล {len(targets)} รายการในพื้นหลังเรียบร้อยแล้ว (ใช้ระบบ 15 Threads คู่ขนาน)",
            "total_targets": len(targets),
            "estimated_time": "ประมาณ 20-30 วินาที",
            "check_status_url": "https://web-service-u8aj.onrender.com/batch/status"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

@app.get("/batch/status")
def get_batch_status():
    """เปิดดูความคืบหน้าแบบ Real-time"""
    global batch_tracker
    pct = 0.0
    if batch_tracker["total"] > 0:
        pct = round((batch_tracker["processed"] / batch_tracker["total"]) * 100, 1)
    
    return {
        "status": batch_tracker["status"],
        "is_running": batch_tracker["is_running"],
        "progress_percent": f"{pct}%",
        "processed": batch_tracker["processed"],
        "total": batch_tracker["total"],
        "errors": batch_tracker["errors"],
        "message": batch_tracker["message"]
    }
