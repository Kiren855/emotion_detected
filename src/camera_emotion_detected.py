import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Tải mô hình đã huấn luyện
model = load_model("emotion_model.keras")

# Nhãn cảm xúc (tuỳ thuộc vào mô hình của bạn)
emotion_labels = ['Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Tải Haar Cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Khởi động camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Cắt vùng khuôn mặt
        face_roi = gray[y:y + h, x:x + w]

        # Resize và chuẩn hóa
        resized_face = cv2.resize(face_roi, (48, 48))  # Điều chỉnh kích thước phù hợp với mô hình
        input_face = np.expand_dims(resized_face, axis=0)
        input_face = np.expand_dims(input_face, axis=-1)
        input_face = input_face / 255.0  # Chuẩn hóa dữ liệu

        # Dự đoán cảm xúc
        prediction = model.predict(input_face)
        emotion_index = np.argmax(prediction)
        emotion_label = emotion_labels[emotion_index]

        # Vẽ khung vuông quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Hiển thị cảm xúc trên khung
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Hiển thị khung hình
    cv2.imshow("Emotion Recognition", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
