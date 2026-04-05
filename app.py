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
# ⚙️ הגדרות ופרמטרים
# ==========================================
# הלינק הציבורי ששלחת (פורמט CSV)
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRts2FKh1u-wF8aaLuaPqn0wfRJ6EarRMoTQ3HK2_95KJtBJwxQ5WoYOql9c6jsAZ3wcqz4jNdnim6z/pub?gid=0&single=true&output=csv"

CONFIDENCE_MULTIPLIER = 800
BUY_MARGIN = 1.000
SELL_MARGIN = 0.990

app = Flask(__name__)
CORS(app)

# משתנים גלובליים למודלים
model_dip = None
model_peak = None
last_features_row = None

def train_model():
    global model_dip, model_peak, last_features_row
    print("🔄 מושך נתונים מהלינק הציבורי ומאמן מודלים...")
    
    # משיכת ה-CSV מהאינטרנט
    response = requests.get(CSV_URL)
    df = pd.read_csv(StringIO(response.text))
    
    # ניקוי והמרת נתונים
    cols_to_convert = ['Open', 'High', 'Low', 'Close', 'VIX_Close', 'Oil_Close']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    # הנדסת מאפיינים
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['Prev_Vol'] = (df['High'].shift(1) - df['Low'].shift(1)) / df['Open'].shift(1)
    df['VIX_Chg'] = df['VIX_Close'].pct_change()
    df['Oil_Chg'] = df['Oil_Close'].pct_change()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    rs = gain.ewm(com=13, min_periods=14).mean() / loss.ewm(com=13, min_periods=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + rs))

    for ma in [20, 50, 100, 150, 200]:
        df[f'MA_{ma}'] = df['Close'].rolling(window=ma).mean()
        df[f'Dist_from_MA{ma}'] = (df['Close'] - df[f'MA_{ma}']) / df[f'MA_{ma}']

    df['MA20_Slope'] = (df['MA_20'] - df['MA_20'].shift(3)) / df['MA_20'].shift(3)
    df['Target_Dip'] = (df['Low'] - df['Open']) / df['Open']
    df['Target_Peak'] = (df['High'] - df['Open']) / df['Open']
    df = df.dropna()

    features = ['Open', 'Gap', 'Prev_Vol', 'VIX_Chg', 'Oil_Chg', 'RSI_14', 
                'Dist_from_MA20', 'MA20_Slope', 'Dist_from_MA50', 'Dist_from_MA100', 
                'Dist_from_MA150', 'Dist_from_MA200']

    # אימון המודלים
    model_dip = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    model_peak = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    model_dip.fit(df[features], df['Target_Dip'])
    model_peak.fit(df[features], df['Target_Peak'])
    
    last_features_row = df[features].iloc[-1:].copy()
    print("✅ המודלים אומנו בהצלחה!")

@app.route('/predict', methods=['GET'])
def predict():
    if model_dip is None:
        return jsonify({"error": "Model not trained yet"}), 500
    
    open_price = request.args.get('open_price', type=float)
    if not open_price:
        return jsonify({"error": "Missing open_price"}), 400

    # עדכון פיצ'רים לחיזוי
    current_features = last_features_row.copy()
    current_features['Open'] = open_price

    # תחזיות
    pred_dip_pct = model_dip.predict(current_features)[0]
    pred_peak_pct = model_peak.predict(current_features)[0]

    buy_target = open_price * (1 + pred_dip_pct) * BUY_MARGIN
    sell_target = open_price * (1 + pred_peak_pct) * SELL_MARGIN

    # ביטחון
    dip_tree_preds = np.array([tree.predict(current_features.values) for tree in model_dip.estimators_])
    peak_tree_preds = np.array([tree.predict(current_features.values) for tree in model_peak.estimators_])
    
    buy_conf = int(np.clip(100 - (np.std(dip_tree_preds) * CONFIDENCE_MULTIPLIER), 0, 100))
    sell_conf = int(np.clip(100 - (np.std(peak_tree_preds) * CONFIDENCE_MULTIPLIER), 0, 100))

    return jsonify({
        "buy_target": round(buy_target, 2),
        "buy_confidence": buy_conf,
        "sell_target": round(sell_target, 2),
        "sell_confidence": sell_conf
    })

if __name__ == '__main__':
    train_model()
    # הרצה על פורט 5000. host='0.0.0.0' מאפשר גישה מחוץ למחשב
    app.run(host='0.0.0.0', port=5000)