import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from supabase import create_client, Client

app = FastAPI(title="NCD Progression AI Service")

# 1. เชื่อมต่อ Supabase ผ่าน Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------------------------------------
# กำหนด Path แบบ Absolute อ้างอิงจากตำแหน่งของไฟล์ main.py เสมอ
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ncd_progression_xgb.joblib")
FEATURE_PATH = os.path.join(BASE_DIR, "model", "progression_features.joblib")

# ตรวจสอบการมีอยู่ของไฟล์ก่อนโหลด
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ไม่พบไฟล์โมเดลที่: {MODEL_PATH}")
if not os.path.exists(FEATURE_PATH):
    raise FileNotFoundError(f"ไม่พบไฟล์ Features ที่: {FEATURE_PATH}")

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURE_PATH)

def process_and_predict(record: Dict[str, Any]):
    """ฟังก์ชันประมวลผลเบื้องหลัง (Background Task) คำนวณความเสี่ยงและอัปเดตกลับ Supabase"""
    try:
        rec_id = record.get("id")
        if not rec_id:
            return

        # แปลงข้อมูลสรีรวิทยา
        age = float(record.get("age") or 0)
        h = float(record.get("height") or 0)
        w = float(record.get("weight") or 0)
        waist = float(record.get("waist") or 0)
        
        bmi = round(w / ((h / 100) ** 2), 2) if h > 0 else 0
        whtr = round(waist / h, 2) if h > 0 else 0

        # จัดการความดัน SYS / DIA (ดึงจาก bp หากช่องแยกไม่มีค่า)
        sys = float(record.get("sys") or 0)
        dia = float(record.get("dia") or 0)
        bp = str(record.get("bp") or "")
        if (sys == 0 or dia == 0) and "/" in bp:
            parts = bp.split("/")
            if len(parts) >= 2:
                sys = float(parts[0]) if parts[0].isdigit() else 0
                dia = float(parts[1]) if parts[1].isdigit() else 0

        pulse_pressure = sys - dia
        map_pressure = round(dia + (pulse_pressure / 3.0), 2)
        sugar = float(record.get("sugar") or 0)

        # การถอดรหัสพฤติกรรม (Encoding ให้ตรงกับตอนเทรน)
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

        # สร้าง DataFrame 1 แถวป้อนให้โมเดล
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

        # คำนวณค่าความน่าจะเป็นของการลุกลาม (0 - 100%)
        risk_probability = float(model.predict_proba(input_df)[:, 1][0]) * 100

        # แบ่งกลุ่มความเสี่ยงทางการแพทย์ (Risk Tiers)
        if risk_probability >= 70.0:
            tier = "วิกฤต/เสี่ยงสูงมาก (High Risk)"
        elif risk_probability >= 40.0:
            tier = "เฝ้าระวัง/เสี่ยงปานกลาง (Moderate Risk)"
        else:
            tier = "ความเสี่ยงต่ำ (Low Risk)"

        # เขียนผลลัพธ์กลับลงตาราง records ของ Supabase
        if supabase:
            supabase.table("records").update({
                "ai_risk_score": round(risk_probability, 2),
                "ai_risk_tier": tier,
                "ai_predicted_at": "now()"
            }).eq("id", rec_id).execute()

            print(f"Record {rec_id} ประมวลผลสำเร็จ: {risk_probability:.2f}% ({tier})")

    except Exception as e:
        print(f"Error processing prediction for record: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "online", "model": "XGBoost NCD Progression"}

@app.post("/webhook/predict")
async def supabase_webhook(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    """จุดรับ Webhook จาก Supabase"""
    record = payload.get("record")
    if not record:
        raise HTTPException(status_code=400, detail="No record found in payload")

    # สั่งประมวลผลเบื้องหลังทันทีเพื่อป้องกัน Timeout (ตอบกลับ Supabase ภายในไม่กี่ ms)
    background_tasks.add_task(process_and_predict, record)
    return {"status": "queued", "record_id": record.get("id")}