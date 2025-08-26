# =========================================================
# 1. إنشاء بيانات كسلا (إذا لم تكن موجودة)
# =========================================================
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import os
import sys

sys.stdout.reconfigure(encoding='utf-8') # type: ignore

# إنشاء البيانات إذا لم يكن الملف موجود
if not os.path.exists("kasala_houses.csv"):
    locations = ["حي الميرغنية", "حي الختمية", "حي السواقي", "حي التاكا", "حي المربعات"]
    data = []
    for _ in range(200):
        area = random.randint(80, 400)
        bedrooms = random.randint(1, 6)
        bathrooms = random.randint(1, 4)
        location = random.choice(locations)
        age = random.randint(0, 30)
        base_price = area * 150000
        location_factor = (locations.index(location) + 1) * 0.05
        age_factor = max(0.5, 1 - (age * 0.02))
        price = int(base_price * (1 + location_factor) * age_factor)
        data.append([area, bedrooms, bathrooms, location, age, price])
    df = pd.DataFrame(data, columns=["Area", "Bedrooms", "Bathrooms", "Location", "House_Age", "Price"])
    df.to_csv("kasala_houses.csv", index=False, encoding="utf-8-sig")

# =========================================================
# 2. تنظيف ومعالجة البيانات
# =========================================================
df = pd.read_csv("kasala_houses.csv")
encoder = LabelEncoder()
df["Location"] = encoder.fit_transform(df["Location"])
X = df.drop("Price", axis=1)
y = df["Price"]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# حفظ أدوات المعالجة
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")

# =========================================================
# 3. تدريب النموذج إذا لم يكن موجود
# =========================================================
if not os.path.exists("house_price_model.h5"):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = Sequential([
        Dense(64, input_shape=(X_train.shape[1],), activation="relu"),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error", metrics=["mae"])
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=0)
    model.save("house_price_model.h5")

# =========================================================
# 4. واجهة Streamlit محسّنة مع رسم بياني
# =========================================================
model = load_model("house_price_model.h5")

# تصميم CSS مخصص
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }
    h1 {
        color: #007bff;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# العنوان
st.markdown("<h1>🏠 التنبؤ بأسعار المنازل - كسلا</h1>", unsafe_allow_html=True)
st.write("أدخل بيانات المنزل للحصول على السعر المتوقع:")

# إدخالات المستخدم
area = st.number_input("📏 المساحة (متر مربع)", min_value=50, max_value=1000, value=200)
bedrooms = st.number_input("🛏 عدد الغرف", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("🚿 عدد الحمامات", min_value=1, max_value=5, value=2)
location = st.selectbox("📍 الحي", encoder.classes_)
age = st.number_input("📅 عمر المنزل (بالسنوات)", min_value=0, max_value=50, value=5)

# زر التنبؤ
if st.button("🔍 تنبؤ بالسعر"):
    location_encoded = encoder.transform([location])[0]
    input_data = np.array([[area, bedrooms, bathrooms, location_encoded, age]])
    input_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_scaled)
    st.success(f"💰 السعر المتوقع: {int(predicted_price[0][0]):,} جنيه سوداني")

# =========================================================
# 5. رسم بياني للمقارنة بين الأسعار الحقيقية والمتوقعة
# =========================================================
st.markdown("---")
st.subheader("📊 مقارنة بين الأسعار الحقيقية والمتوقعة (اختبار النموذج)")

# إعادة تقسيم البيانات لعمل التقييم
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test).flatten()

# رسم المقارنة
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(range(len(y_test)), y_test, label="السعر الحقيقي", color="blue")
ax.scatter(range(len(y_pred)), y_pred, label="السعر المتوقع", color="red")
ax.set_title("المقارنة بين الأسعار الحقيقية والمتوقعة")
ax.set_xlabel("رقم العينة")
ax.set_ylabel("السعر")
ax.legend()
st.pyplot(fig)

# إضافة معلومات أسفل الصفحة
st.markdown("---")
st.caption("تم تطوير هذا التطبيق باستخدام Python + Streamlit + Keras كجزء من مشروع تخرج - آمنة عثمان 🌟")
