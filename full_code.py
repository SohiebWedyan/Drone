from ultralytics import YOLO
import cv2
import time
import math

# =========================
# تحميل الموديل
# =========================
model = YOLO("best.pt")

# تشغيل الكاميرا
cap = cv2.VideoCapture(1)

# تخزين البيانات
prev_positions = {}
track_history = {}

# =========================
# loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tracking
    results = model.track(frame, persist=True)

    for r in results:
        for box in r.boxes:

            # =========================
            # البيانات الأساسية
            # =========================
            track_id = int(box.id) if box.id is not None else -1
            x, y, w, h = box.xywh[0]

            x, y, w, h = float(x), float(y), float(w), float(h)
            current_time = time.time()

            # =========================
            # حساب Z (المسافة)
            # =========================
            real_height = 0.3  # متر
            focal_length = 800
            Z = (real_height * focal_length) / h

            # =========================
            # حفظ المسار
            # =========================
            if track_id not in track_history:
                track_history[track_id] = []

            track_history[track_id].append((int(x), int(y)))

            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            # رسم المسار
            for i in range(1, len(track_history[track_id])):
                cv2.line(frame,
                         track_history[track_id][i-1],
                         track_history[track_id][i],
                         (0,255,0), 2)

            # النقطة الحالية
            cv2.circle(frame, (int(x), int(y)), 5, (255,0,0), -1)

            # =========================
            # حساب السرعة + الاتجاه + prediction
            # =========================
            if track_id in prev_positions:
                prev_x, prev_y, prev_Z, prev_t = prev_positions[track_id]

                dx = x - prev_x
                dy = y - prev_y
                dz = Z - prev_Z
                dt = current_time - prev_t

                if dt > 0:

                    # 🎯 الاتجاه
                    dir_x = "Right" if dx > 0 else "Left"
                    dir_y = "Down" if dy > 0 else "Up"

                    # 🚀 السرعة (متر/ثانية)
                    meter_per_pixel = 0.3 / h
                    dist_pixels = math.sqrt(dx**2 + dy**2)
                    dist_m = dist_pixels * meter_per_pixel
                    speed = dist_m / dt

                    # 🔮 التوقع
                    vx = dx / dt
                    vy = dy / dt
                    vz = dz / dt

                    x_next = int(x + vx * dt)
                    y_next = int(y + vy * dt)
                    Z_next = Z + vz * dt

                    # رسم نقطة التوقع
                    cv2.circle(frame, (x_next, y_next), 6, (0,0,255), -1)

                    # سهم الاتجاه
                    cv2.arrowedLine(frame,
                                    (int(x), int(y)),
                                    (x_next, y_next),
                                    (0,255,255), 2)

                    # عرض البيانات
                    label = f"ID:{track_id} Z:{Z:.1f}m V:{speed:.2f}m/s"
                    cv2.putText(frame, label,
                                (int(x), int(y)-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0,255,0), 2)

                    print(f"ID {track_id} | Z={Z:.2f}m | Speed={speed:.2f}m/s | Dir={dir_x},{dir_y}")

            # تحديث البيانات السابقة
            prev_positions[track_id] = (x, y, Z, current_time)

    # =========================
    # Alert
    # =========================
    if len(results[0].boxes) > 0:
        cv2.putText(frame, "ALERT: Balloon Detected!",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,0,255), 3)

    # عرض الصورة
    cv2.imshow("AI System - Balloon Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()