import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
import requests
from io import StringIO
import traceback

# הסתרת אזהרות מעצבנות
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ קונפיגורציה ופרמטרים קבועים
# ==========================================
# ה-URL הציבורי של הגוגל שיטס המיוצא כ-CSV (SOXL).
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRts2FKh1u-wF8aaLuaPqn0wfRJ6EarRMoTQ3HK2_95KJtBJwxQ5WoYOql9c6jsAZ3wcqz4jNdnim6z/pub?gid=0&single=true&output=csv"

# פרמטרים לכיוונון מודל הביטחון (Confidence)
BASE_CONFIDENCE_MULTIPLIER = 1000 # שמרני יותר במדדים
DISTANCE_PENALTY_FACTOR = 1.5 

app = Flask(__name__)
CORS(app) 

# משתנים גלובליים להחזקת המודלים והנתונים האחרונים
model_dip = None
model_peak = None
last_features_row = None

# ==========================================
# 📊 פונקציית אימון המודל עם תמיכה בלוגים
# ==========================================
def train_model():
    global model_dip, model_peak, last_features_row
    print("🔄 מושך נתונים ומאמן מודלים מתקדמים (עם הדפסת שגיאות ללוג)...")
    
    # 1. משיכת הנתונים בזמן אמת מהגוגל שיטס
    try:
        response = requests.get(CSV_URL)
        response.raise_for_status() 
        df = pd.read_csv(StringIO(response.text))
        print("✅ הנתונים נמשכו בהצלחה מהשיט.")
    except Exception as e:
        print(f"❌ שגיאה במשיכת הנתונים: {e}")
        traceback.print_exc() 
        return

    # 2. ניקוי ועיבוד נתונים בסיסי
    try:
        # המרת עמודות למספרים
        cols_to_convert = ['Open', 'High', 'Low', 'Close', 'VIX_Close', 'Oil_Close', 'NVDA_Close']
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        # 3. הנדסת מאפיינים (Feature Engineering)
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Prev_Vol'] = (df['High'].shift(1) - df['Low'].shift(1)) / df['Open'].shift(1)
        df['VIX_Chg'] = df['VIX_Close'].pct_change()
        df['Oil_Chg'] = df['Oil_Close'].pct_change()
        df['NVDA_Chg'] = df['NVDA_Close'].pct_change() 
        df['NVDA_Dist_MA20'] = (df['NVDA_Close'] - df['NVDA_Close'].rolling(20).mean()) / df['NVDA_Close'].rolling(20).mean()

        # RSI בסיסי
        delta = df['Close'].diff()
        gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
        rs = gain.ewm(com=13, min_periods=14).mean() / loss.ewm(com=13, min_periods=14).mean()
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # ממוצעים נעים (MAs)
        for ma in [20, 50, 100, 200]:
            df[f'MA_{ma}'] = df['Close'].rolling(window=ma).mean()
            df[f'Dist_from_MA{ma}'] = (df['Close'] - df[f'MA_{ma}']) / df[f'MA_{ma}']

        # הגדרת המטרות לחיזוי (Targets)
        df['Target_Dip'] = (df['Low'] - df['Open']) / df['Open']
        df['Target_Peak'] = (df['High'] - df['Open']) / df['Open']
        df = df.dropna()

        # בחירת המאפיינים למודל (Features)
        features = ['Open', 'Gap', 'Prev_Vol', 'VIX_Chg', 'Oil_Chg', 'RSI_14', 
                    'Dist_from_MA20', 'Dist_from_MA50', 'Dist_from_MA100', 'Dist_from_MA200',
                    'NVDA_Chg', 'NVDA_Dist_MA20']

        # 4. אימון המודלים (Random Forest Regressor)
        print("🤖 מאמן מודלי Random Forest...")
        model_dip = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
        model_peak = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
        
        model_dip.fit(df[features], df['Target_Dip'])
        model_peak.fit(df[features], df['Target_Peak'])
        
        # שמירת השורה האחרונה לצורך פיצ'רים של ה-Lag בחיזוי
        last_features_row = df[features].iloc[-1:].copy()
        print("✅ אימון המודל הושלם בהצלחה.")
    except Exception as e:
        print(f"❌ שגיאה באימון המודל או בעיבוד הנתונים: {e}")
        traceback.print_exc()

# ==========================================
# 🚀 הגדרת ה-API Endpoint לחיזוי
# ==========================================
@app.route('/predict', methods=['GET'])
def predict():
    if model_dip is None:
        return jsonify({"error": "Model not trained yet"}), 500
    
    # משיכת פרמטרים מה-URL
    open_price = request.args.get('open_price', type=float)
    # ברירת מחדל לסליידרים היא 50 (ניטרלי)
    buy_shaper = request.args.get('buy_shaper', default=50, type=int) 
    sell_shaper = request.args.get('sell_shaper', default=50, type=int) 
    
    if not open_price:
        return jsonify({"error": "Missing open_price"}), 400

    # עדכון שורת הפיצ'רים עם מחיר הפתיחה שהוזן
    current_features = last_features_row.copy()
    current_features['Open'] = open_price

    # ---------------------------------------------------------
    # 🧠 לוגיקת חישוב דינמית - כולל התיקון הקריטי
    # ---------------------------------------------------------

    # 1. טווח פקטורים אגרסיבי (0.3 עד 1.8)
    # 1 (אגרסיבי) -> 0.3, 100 (שמרני) -> 1.8
    def calculate_adjustment_factor(slider_val):
        return 0.3 + (slider_val / 100.0) * 1.5

    buy_safe_factor = calculate_adjustment_factor(buy_shaper)
    sell_safe_factor = calculate_adjustment_factor(sell_shaper)

    # 2. ביצוע החיזוי הבסיסי (Raw Prediction) מהמודל
    pred_dip_pct = model_dip.predict(current_features)[0] 
    pred_peak_pct = model_peak.predict(current_features)[0] 

    # --- !!! חידוש 2 !!! מנגנון Force Dip (סף מינימום) ---
    # נגדיר סף ירידה מינימלי למצב אגרסיבי (1.5% מתחת לפתיחה).
    MIN_DIP_PCT = -0.015 
    
    # !!! התיקון הקריטי כאן בשורה 137 !!!
    # משתמשים ב-buy_safe_factor (שהוגדר למעלה) במקום buy_adjustment.
    if buy_safe_factor < 1.0: # מצב אגרסיבי
        # נשנה את החיזוי הגולמי כך שיבטיח ירידה אמיתית
        pred_dip_pct = pred_dip_pct * 0.5 + MIN_DIP_PCT * 0.5 
    # ---------------------------------------------------

    # 3. חישוב יעדי המחיר הסופיים (Price Targets)
    # בקנייה: ככל ש buy_safe_factor נמוך (אגרסיבי), המחיר יורד דרסטית.
    buy_target = open_price * (1 + (pred_dip_pct / buy_safe_factor))
    sell_target = open_price * (1 + (pred_peak_pct / sell_safe_factor))

    # 4. חישוב רמות ביטחון דינמיות (Confidence) - עם קנס מרחק
    # א. חישוב שונות התחזיות בין כל העצים בפורסט (סטיית תקן)
    dip_tree_preds = np.array([tree.predict(current_features.values) for tree in model_dip.estimators_])
    peak_tree_preds = np.array([tree.predict(current_features.values) for tree in model_peak.estimators_])
    dip_std = np.std(dip_tree_preds)
    peak_std = np.std(peak_tree_preds)

    # ב. חישוב "קנס" על מרחק היעד ממחיר הפתיחה
    # ככל שהיעד רחוק יותר, המודל פחות בטוח.
    buy_dist_penalty = abs(buy_target - open_price) / open_price
    sell_dist_penalty = abs(sell_target - open_price) / open_price

    # Multiplier on STD (Inverse of safety factor)
    buy_risk_mult = BASE_CONFIDENCE_MULTIPLIER / buy_safe_factor
    sell_risk_mult = BASE_CONFIDENCE_MULTIPLIER / sell_safe_factor

    buy_conf_raw = 100 - (dip_std * buy_risk_mult) - (buy_dist_penalty * DISTANCE_PENALTY_FACTOR * 100)
    sell_conf_raw = 100 - (peak_std * sell_risk_mult) - (sell_dist_penalty * DISTANCE_PENALTY_FACTOR * 100)

    # ד. הגבלת התוצאות לטווח 0-100 ועיגול
    buy_confidence = int(np.clip(buy_conf_raw, 1, 99)) # מונע 0% או 100% מוחלטים
    sell_confidence = int(np.clip(sell_conf_raw, 1, 99))

    # החזרת התוצאות כ-JSON
    return jsonify({
        "buy_target": round(buy_target, 2),
        "buy_confidence": buy_confidence,
        "sell_target": round(sell_target, 2),
        "sell_confidence": sell_confidence,
        "raw_model_dip": round(pred_dip_pct, 4),
        "raw_model_peak": round(pred_peak_pct, 4)
    })

# ==========================================
# 🏁 הרצת האפליקציה
# ==========================================
if __name__ == '__main__':
    train_model() 
    app.run(host='0.0.0.0', port=5000)
else:
    train_model()
