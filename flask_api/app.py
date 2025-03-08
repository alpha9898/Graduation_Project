from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# إنشاء تطبيق Flask
app = Flask(__name__)
CORS(app)  # لحل مشاكل CORS عند الاتصال من Flutter

# 🔹 تحميل البيانات الأصلية لاسترجاع المقياس الحقيقي
data = pd.read_csv("final_dataset.csv")

# 🔹 تجهيز StandardScaler لإعادة تحويل `predicted_bmi`
scaler_bmi = StandardScaler()
scaler_bmi.fit(data["BMI"].values.reshape(-1, 1))

# 🔹 تحميل النماذج الصحيحة
model_bmi = tf.keras.models.load_model("model_bmi.keras")
model_bmicas = tf.keras.models.load_model("model_bmicas.keras")
model_exercise_plan = tf.keras.models.load_model("model_exercise_plan.keras")

print("✅ تم تحميل النماذج بنجاح!")

# قائمة تصنيفات BMI
bmi_category_labels = ["Underweight", "Normal weight", "Overweight", "Obese"]

# قائمة أسماء خطط التمارين (تحتاج لإضافة التصنيفات الصحيحة)
exercise_labels = ["Plan A", "Plan B", "Plan C", "Plan D", "Plan E", "Plan F", "Plan G"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استقبال البيانات بصيغة JSON
        data = request.get_json()

        # استخراج المدخلات من JSON
        age = float(data["age"])
        height = float(data["height"])
        weight = float(data["weight"])
        gender = 1 if data["gender"].lower() == "male" else 0

        # تجهيز البيانات للنموذج (تحويلها إلى الشكل المطلوب)
        input_data = np.array([[age, height, weight, gender]]).reshape(1, 1, 4)

        # 🔹 إجراء التنبؤات باستخدام النماذج
        bmi_prediction_scaled = model_bmi.predict(input_data)[0][0]
        bmi_prediction = scaler_bmi.inverse_transform([[bmi_prediction_scaled]])[0][0]

        bmicas_prediction = bmi_category_labels[np.argmax(model_bmicas.predict(input_data)[0])]  # اسم تصنيف BMI
        exercise_plan_prediction = exercise_labels[np.argmax(model_exercise_plan.predict(input_data)[0])]  # اسم خطة التمارين

        # 🔹 تحويل النتائج إلى JSON
        response = {
            "predicted_bmi": round(float(bmi_prediction), 2),
            "bmi_category": bmicas_prediction,
            "exercise_plan": exercise_plan_prediction
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

# 🔹 تشغيل السيرفر
if __name__ == '__main__':
    app.run(debug=True)
