import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from typing import Dict, Any, List
from supabase import create_client, Client

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

# 2. โหลดโมเดลด้วย Absolute Path
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

def calculate_prediction(record: Dict[str, Any]):
    """ฟังก์ชันกลางแปลง Features และคำนวณความเสี่ยงด้วยโมเดล XGBoost"""
    age = float(record.get("age") or 0)
    h = float(record.get("height") or 0)
    w = float(record.get("weight") or 0)
    waist = float(record.get("waist") or 0)
    
    bmi = round(w / ((h / 100) ** 2), 2) if h > 0 else 0
    whtr = round(waist / h, 2) if h > 0 else 0

    sys = float(record.get("sys") or 0)
    dia = float(record.get("dia") or 0)
    bp = str(record.get("bp") or "")
    if (sys == 0 or dia == 0) and "/" in bp:
        parts = bp.split("/")
        if len(parts) >= 2:
            sys = float(parts[0]) if parts[0].strip().isdigit() else 0
            dia = float(parts[1]) if parts[1].strip().isdigit() else 0

    pulse_pressure = sys - dia
    map_pressure = round(dia + (pulse_pressure / 3.0), 2)
    sugar = float(record.get("sugar") or 0)

    gender_code = 1 if str(record.get("gender") or "").strip() == "ชาย" else 0
    fasting_code = 1 if str(record.get("fasting") or "").strip() == "yes" else 0

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
    """ใช้สำหรับ Webhook แบบเดี่ยว"""
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

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "supabase_connected": supabase is not None
    }

@app.post("/webhook/predict")
async def supabase_webhook(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    """จุดรับ Webhook จาก Supabase พร้อมระบบตัดลูปวนซ้ำ (Infinite Loop Guard)"""
    event_type = payload.get("type")
    record = payload.get("record")
    old_record = payload.get("old_record")

    if not record:
        raise HTTPException(status_code=400, detail="No record found in payload")

    # 🛡️ ตัดลูป: ถ้าเป็นการ UPDATE ให้เช็กว่าค่าสัญญาณชีพเปลี่ยนจริงหรือไม่
    if event_type == "UPDATE" and old_record:
        clinical_keys = [
            "age", "weight", "height", "waist", "sys", "dia", "bp", 
            "sugar", "fasting", "smoking", "alcohol", "family", "gender"
        ]
        # ถ้าสัญญาณชีพและพฤติกรรมเหมือนเดิมทุกประการ แปลว่าเป็นการ Update จากตัว AI เอง -> สั่งข้ามทันที
        has_clinical_change = any(
            str(record.get(k) or "").strip() != str(old_record.get(k) or "").strip() 
            for k in clinical_keys
        )

        if not has_clinical_change:
            return {"status": "skipped", "message": "ข้ามการทำงาน: เป็นการอัปเดตคะแนนจาก AI"}

    # สั่งประมวลผลเบื้องหลังตามปกติ
    background_tasks.add_task(process_and_predict, record)
    return {"status": "queued", "record_id": record.get("id")}

# ====================================================================
# ⚡ จุดประมวลผลย้อนหลังทั้งหมดในคลิกเดียว (Batch Run Endpoint)
# ====================================================================
@app.get("/batch/run")
@app.post("/batch/run")
def batch_run_prediction(force_all: bool = False):
    """
    ดึงข้อมูลทั้งหมดจาก Supabase มาให้ AI วิเคราะห์
    - force_all=False: คำนวณเฉพาะคนที่ ai_risk_score ยังเป็น NULL
    - force_all=True: คำนวณใหม่ทุกคนทั้งตาราง
    """
    if not model:
        raise HTTPException(status_code=500, detail="โมเดลยังไม่พร้อมใช้งาน")
    if not supabase:
        raise HTTPException(status_code=500, detail="ยังไม่ได้เชื่อมต่อ Supabase (ตรวจเช็ก Service Role Key บน Render)")

    try:
        # ดึงข้อมูลจากตาราง records สูงสุด 5,000 แถว
        res = supabase.table("records").select("*").limit(5000).execute()
        all_records = res.data or []

        # กรองเฉพาะแถวที่ต้องประมวลผล
        targets = all_records if force_all else [r for r in all_records if r.get("ai_risk_score") is None]

        updated_count = 0
        error_count = 0
        processed_logs = []

        for r in targets:
            rec_id = r.get("id")
            if not rec_id:
                continue
            try:
                score, tier = calculate_prediction(r)
                supabase.table("records").update({
                    "ai_risk_score": score,
                    "ai_risk_tier": tier,
                    "ai_predicted_at": "now()"
                }).eq("id", rec_id).execute()
                
                updated_count += 1
                name = (r.get("prefix") or "") + (r.get("name") or "")
                processed_logs.append(f"{name}: {score}% ({tier})")
            except Exception as item_err:
                error_count += 1
                print(f"Error record {rec_id}: {item_err}")

        return {
            "status": "success",
            "message": f"ประมวลผลสำเร็จ {updated_count} รายการ (เกิดข้อผิดพลาด {error_count} รายการ)",
            "total_found": len(all_records),
            "processed_count": updated_count,
            "sample_results": processed_logs[:10]  # แสดงตัวอย่าง 10 คนแรก
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการรัน Batch: {str(e)}")
