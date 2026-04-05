import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
import requests
from io import StringIO

# ביטול אזהרות מעצבנות כדי לשמור על לוג נקי
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ קונפיגורציה ופרמטרים קבועים
# ==========================================
# ה-URL הציבורי של הגוגל שיטס שלך המיוצא כ-CSV
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRts2FKh1u-wF8aaLuaPqn0wfRJ6EarRMoTQ3HK2_95KJtBJwxQ5WoYOql9c6jsAZ3wcqz4jNdnim6z/pub?gid=0&single=true&output=csv"

# פרמטרים לכיוונון מודל הביטחון (Confidence)
# Base Multiplier: ככל שהמספר גבוה יותר, הביטחון הכללי ירד.
# Distance Penalty Factor: משקלו של ה"קנס" על המרחק ממחיר הפתיחה.
BASE_CONFIDENCE_MULTIPLIER = 800  # כיוונון עדין למניעת 0% קבוע
DISTANCE_PENALTY_FACTOR = 1.5      # "קנס" מתון יותר על מרחק

app = Flask(__name__)
CORS(app) # מאפשר לפורטנד (Lovable) לתקשר עם הבקאנד

# משתנים גלובליים להחזקת המודלים והנתונים האחרונים
model_dip = None
model_peak = None
last_features_row = None

# ==========================================
# 📊 פונקציית אימון המודל
# ==========================================
def train_model():
    global model_dip, model_peak, last_features_row
    print("🔄 מושך נתונים ומאמן מודלים מתקדמים...")
    
    # 1. משיכת הנתונים בזמן אמת מהגוגל שיטס
    try:
        response = requests.get(CSV_URL)
        response.raise_for_status() # בדיקה שהמשיכה הצליחה
        df = pd.read_csv(StringIO(response.text))
    except Exception as e:
        print(f"❌ שגיאה במשיכת הנתונים: {e}")
        return

    # 2. ניקוי ועיבוד נתונים בסיסי
    cols_to_convert = ['Open', 'High', 'Low', 'Close', 'VIX_Close', 'Oil_Close', 'NVDA_Close']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # הסרת שורות ריקות שנוצרו מההמרה
    df = df.dropna()

    # 3. הנדסת מאפיינים (Feature Engineering)
    print("🧠 מחשב אינדיקטורים טכניים...")
    # גאפ פתיחה
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    # תנודתיות יום קודם
    df['Prev_Vol'] = (df['High'].shift(1) - df['Low'].shift(1)) / df['Open'].shift(1)
    
    # שינויי אחוזים של נכסים מתואמים
    df['VIX_Chg'] = df['VIX_Close'].pct_change()
    df['Oil_Chg'] = df['Oil_Close'].pct_change()
    df['NVDA_Chg'] = df['NVDA_Close'].pct_change() 
    
    # מרחק NVDA מממוצע נע 20 (אינדיקטור עוצמה)
    df['NVDA_Dist_MA20'] = (df['NVDA_Close'] - df['NVDA_Close'].rolling(20).mean()) / df['NVDA_Close'].rolling(20).mean()

    # RSI בסיסי ל-SOXL
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.ewm(com=13, min_periods=14).mean() / loss.ewm(com=13, min_periods=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # ממוצעים נעים ומרחק מהם (MAs)
    for ma in [20, 50, 100, 200]:
        df[f'MA_{ma}'] = df['Close'].rolling(window=ma).mean()
        df[f'Dist_from_MA{ma}'] = (df['Close'] - df[f'MA_{ma}']) / df[f'MA_{ma}']

    # define variables to predict (Targets)
    # Target_Dip: כמה המחיר הנמוך היה מתחת לפתיחה (באחוזים)
    # Target_Peak: כמה המחיר הגבוה היה מעל לפתיחה (באחוזים)
    df['Target_Dip'] = (df['Low'] - df['Open']) / df['Open']
    df['Target_Peak'] = (df['High'] - df['Open']) / df['Open']
    
    # ניקוי סופי של שורות ה-Lag (תחילת הנתונים)
    df = df.dropna()

    # בחירת המאפיינים למודל (Features)
    features = ['Open', 'Gap', 'Prev_Vol', 'VIX_Chg', 'Oil_Chg', 'RSI_14', 
                'Dist_from_MA20', 'Dist_from_MA50', 'Dist_from_MA100', 'Dist_from_MA200',
                'NVDA_Chg', 'NVDA_Dist_MA20']

    # 4. אימון המודלים (Random Forest Regressor)
    # משתמשים ב-n_estimators נמוך יחסית (150) ו max_depth (8) כדי למנוע Overfitting
    # וכדי שסטיית התקן בין העצים תהיה משמעותית לצורך חישוב Confidence.
    print("🤖 מאמן מודלי Random Forest...")
    model_dip = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    model_peak = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    
    model_dip.fit(df[features], df['Target_Dip'])
    model_peak.fit(df[features], df['Target_Peak'])
    
    # שמירת השורה האחרונה לצורך פיצ'רים של ה-Lag בחיזוי הבא
    last_features_row = df[features].iloc[-1:].copy()
    print("✅ אימון המודל הושלם בהצלחה.")

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
    # 🧠 לוגיקת חישוב דינמית - פתרון הבאגים המרכזיים
    # ---------------------------------------------------------

    # 1. יצירת פקטורים לנרמול הסליידרים (Mapping 1-100 to dynamic multipliers)
    # אנו ממפים את הסליידר (1-100) לטווח משמעותי יותר (למשל 0.7 עד 1.3) כדי להגביר תגובתיות.
    def calculate_adjustment_factor(slider_val):
        # למשל: 1 (אגרסיבי) -> 0.7 (פחות בטוח), 100 (שמרני) -> 1.3 (יותר בטוח)
        return 0.7 + (slider_val / 100.0) * 0.6

    buy_safe_factor = calculate_adjustment_factor(buy_shaper)
    sell_safe_factor = calculate_adjustment_factor(sell_shaper)

    # 2. ביצוע החיזוי הבסיסי (Raw Prediction) מהמודל
    # אלו אחוזי התנועה הצפויים ללא כיוונון סיכון.
    pred_dip_pct = model_dip.predict(current_features)[0] # מספר שלילי (למשל -0.03 עבור -3%)
    pred_peak_pct = model_peak.predict(current_features)[0] # מספר חיובי (למשל 0.03 עבור +3%)

    # 3. חישוב יעדי המחיר הסופיים (Price Targets) - כולל תיקון הלוגיקה ההפוכה
    
    # קנייה (Buy/Dip Shaper):
    # שמרני (SafeFactor גבוה) -> רוצה יעד גבוה יותר (קרוב לפתיחה, פחות ירידה).
    # נחלק את התחזית (השלילית) בפקטור הבטיחות.
    buy_target = open_price * (1 + (pred_dip_pct / buy_safe_factor))
    
    # מכיָרה (Sell/Peak Shaper): !!! תיקון הבאג המרכזי !!!
    # שמרני (SafeFactor גבוה) -> רוצה יעד *נמוך יותר* (קרוב לפתיחה, הבטחת רווח).
    # לכן, נחלק את התחזית (החיובית) בפקטור הבטיחות.
    # (בעבר ביצענו כפל, מה שגרם להתנהגות ההפוכה).
    sell_target = open_price * (1 + (pred_peak_pct / sell_safe_factor))

    # 4. חישוב רמות ביטחון דינמיות (Confidence) - פתרון בעיית ה-0% הקפוא
    # הרעיון: ביטחון נמוך = שונות גבוהה בין העצים * פקטור סיכון * פקטור מרחק ממחיר הפתיחה.

    # א. חישוב שונות התחזיות בין כל העצים בפורסט (סטיית תקן)
    dip_tree_preds = np.array([tree.predict(current_features.values) for tree in model_dip.estimators_])
    peak_tree_preds = np.array([tree.predict(current_features.values) for tree in model_peak.estimators_])
    
    dip_std = np.std(dip_tree_preds)
    peak_std = np.std(peak_tree_preds)

    # ב. חישוב "קנס" על מרחק היעד ממחיר הפתיחה
    # ככל שהיעד רחוק יותר, המודל צריך להיות פחות בטוח (גם אם העצים מסכימים).
    buy_dist_penalty = abs(buy_target - open_price) / open_price
    sell_dist_penalty = abs(sell_target - open_price) / open_price

    # ג. שילוב המדדים הסופי (Confidence logic)
    # אנו משתמשים ב BASE_CONFIDENCE_MULTIPLIER מכוונן כדי למנוע נפילה ל-0% מהירה מדי במצב אגרסיבי.
    
    # Multiplier on STD (Inverse of safety factor: Aggressive=High Multiplier -> Lower Conf)
    buy_risk_mult = BASE_CONFIDENCE_MULTIPLIER / buy_safe_factor
    sell_risk_mult = BASE_CONFIDENCE_MULTIPLIER / sell_safe_factor

    # Buy Confidence
    buy_conf_raw = 100 - (dip_std * buy_risk_mult) - (buy_dist_penalty * DISTANCE_PENALTY_FACTOR * 100)
    # Sell Confidence
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
    # אימון ראשוני בעת ההפעלה
    train_model() 
    # הרצה על פורט 5000
    app.run(host='0.0.0.0', port=5000)
