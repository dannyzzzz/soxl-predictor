import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
import requests
from io import StringIO

# הסתרת אזהרות
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ הגדרות ופרמטרים קבועים
# ==========================================
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRts2FKh1u-wF8aaLuaPqn0wfRJ6EarRMoTQ3HK2_95KJtBJwxQ5WoYOql9c6jsAZ3wcqz4jNdnim6z/pub?gid=0&single=true&output=csv"

# מכפיל ביטחון בסיסי (עלינו מעט כדי להיות שמרניים יותר במדדים)
BASE_CONFIDENCE_MULTIPLIER = 1000 

app = Flask(__name__)
CORS(app)

# משתנים גלובליים למודלים
model_dip = None
model_peak = None
last_features_row = None

def train_model():
    global model_dip, model_peak, last_features_row
    print("🔄 מושך נתונים ומאמן מודלים מתקדמים...")
    
    # 1. משיכת ה-CSV של SOXL מהגוגל שייטס
    response = requests.get(CSV_URL)
    df = pd.read_csv(StringIO(response.text))
    
    # ניקוי והמרת נתונים
    cols_to_convert = ['Open', 'High', 'Low', 'Close', 'VIX_Close', 'Oil_Close', 'NVDA_Close']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    # הנדסת מאפיינים (Features)
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['Prev_Vol'] = (df['High'].shift(1) - df['Low'].shift(1)) / df['Open'].shift(1)
    df['VIX_Chg'] = df['VIX_Close'].pct_change()
    df['Oil_Chg'] = df['Oil_Close'].pct_change()
    df['NVDA_Chg'] = df['NVDA_Close'].pct_change() 
    df['NVDA_Dist_MA20'] = (df['NVDA_Close'] - df['NVDA_Close'].rolling(20).mean()) / df['NVDA_Close'].rolling(20).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    rs = gain.ewm(com=13, min_periods=14).mean() / loss.ewm(com=13, min_periods=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + rs))

    for ma in [20, 50, 100, 150, 200]:
        df[f'MA_{ma}'] = df['Close'].rolling(window=ma).mean()
        df[f'Dist_from_MA{ma}'] = (df['Close'] - df[f'MA_{ma}']) / df[f'MA_{ma}']

    df['Target_Dip'] = (df['Low'] - df['Open']) / df['Open']
    df['Target_Peak'] = (df['High'] - df['Open']) / df['Open']
    
    # ניקוי סופי לפני אימון
    df = df.dropna()

    features = ['Open', 'Gap', 'Prev_Vol', 'VIX_Chg', 'Oil_Chg', 'RSI_14', 
                'Dist_from_MA20', 'Dist_from_MA50', 'Dist_from_MA100', 
                'Dist_from_MA200', 'NVDA_Chg', 'NVDA_Dist_MA20']

    # אימון המודלים (Random Forest)
    model_dip = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42) # הגדלנו מעט את העצים והעומק
    model_peak = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model_dip.fit(df[features], df['Target_Dip'])
    model_peak.fit(df[features], df['Target_Peak'])
    
    last_features_row = df[features].iloc[-1:].copy()
    print("✅ המודלים אומנו בהצלחה! מחכה לסליידרים.")

@app.route('/predict', methods=['GET'])
def predict():
    if model_dip is None:
        return jsonify({"error": "Model not trained yet"}), 500
    
    open_price = request.args.get('open_price', type=float)
    buy_risk = request.args.get('buy_risk', default=50, type=int) 
    sell_risk = request.args.get('sell_risk', default=50, type=int) 
    
    if not open_price:
        return jsonify({"error": "Missing open_price"}), 400

    # עדכון פיצ'רים לחיזוי
    current_features = last_features_row.copy()
    current_features['Open'] = open_price

    # פונקציית התאמה דינמית (0.7 עד 1.3)
    def calculate_adjustment(risk_level):
        return 0.7 + (risk_level / 100.0) * 0.6
    
    # חישוב הפקטורים הנפרדים
    buy_adjustment = calculate_adjustment(buy_risk)
    sell_adjustment = calculate_adjustment(sell_risk)
    
    # --- חידוש 1: מנגנון "תיקון שמרנות" חזק (Force Dip) ---
    # נגדיר סף מינימום לירידה (Dip) במצב אגרסיבי. למשל, לפחות 1.5% מתחת לפתיחה.
    # זה מונע יעדים הזויים כמו "מעל מחיר הפתיחה" לקנייה.
    
    # תחזיות בסיסיות מהמודל
    pred_dip_pct = model_dip.predict(current_features)[0]
    pred_peak_pct = model_peak.predict(current_features)[0]
    
    # הגדרת סף מינימום ל-Dip (למשל, 1.5% מתחת לפתיחה)
    MIN_DIP_PCT = -0.015 
    
    # תיקון יעדי הקנייה במצב אגרסיבי
    if buy_adjustment < 1.0: # אנחנו במצב אגרסיבי
        # נשנה את החיזוי הגולמי כך שיבטיח ירידה אמיתית
        pred_dip_pct = pred_dip_pct * 0.5 + MIN_DIP_PCT * 0.5 
    # -----------------------------------------------

    # עדכון המרג'ינים (Margins) באופן דינמי
    current_buy_margin = 1.000 + (1 - buy_adjustment) * 0.05
    current_sell_margin = 0.990 - (1 - sell_adjustment) * 0.05
    
    # עדכון מכפילי הביטחון הנפרדים
    current_buy_multiplier = BASE_CONFIDENCE_MULTIPLIER * buy_adjustment
    current_sell_multiplier = BASE_CONFIDENCE_MULTIPLIER * sell_adjustment

    # חישוב היעדים הסופיים
    buy_target = open_price * (1 + pred_dip_pct) * current_buy_margin
    sell_target = open_price * (1 + pred_peak_pct) * current_sell_margin

    # חישוב ביטחון - שימוש במכפילים הדינמיים והנפרדים
    dip_tree_preds = np.array([tree.predict(current_features.values) for tree in model_dip.estimators_])
    peak_tree_preds = np.array([tree.predict(current_features.values) for tree in model_peak.estimators_])
    
    buy_conf = int(np.clip(100 - (np.std(dip_tree_preds) * current_buy_multiplier), 0, 100))
    sell_conf = int(np.clip(100 - (np.std(peak_tree_preds) * current_sell_multiplier), 0, 100))
    
    # --- חידוש 2: מנגנון "תיקון ביטחון" דינמי ליעדים רחוקים ---
    # ככל שהיעד רחוק יותר ממחיר הפתיחה, אנחנו מורידים את הביטחון באופן מלאכותי
    
    # חישוב המרחק של יעדי הקנייה והמכירה ממחיר הפתיחה
    buy_dist = abs(buy_target - open_price)
    sell_dist = abs(sell_target - open_price)
    
    # נגדיר סף מרחק לביטחון גבוה (למשל, 0.8 דולר). מעליו, הביטחון יורד.
    DISTANCE_THRESHOLD = 0.8 
    
    # תיקון ביטחון קנייה
    if buy_dist > DISTANCE_THRESHOLD:
        penalty = (buy_dist - DISTANCE_THRESHOLD) * 15 # הורדה של 15 נקודות לכל דולר מעל הסף
        buy_conf = int(np.clip(buy_conf - penalty, 0, 100))
        
    # תיקון ביטחון מכירה
    if sell_dist > DISTANCE_THRESHOLD:
        penalty = (sell_dist - DISTANCE_THRESHOLD) * 15
        sell_conf = int(np.clip(sell_conf - penalty, 0, 100))
    # -----------------------------------------------

    return jsonify({
        "buy_target": round(buy_target, 2),
        "buy_confidence": buy_conf,
        "sell_target": round(sell_target, 2),
        "sell_confidence": sell_conf,
        "calculated_buy_factor": round(buy_adjustment, 2),
        "calculated_sell_factor": round(sell_adjustment, 2) 
    })

if __name__ == '__main__':
    train_model() 
    app.run(host='0.0.0.0', port=5000)
else:
    train_model()
