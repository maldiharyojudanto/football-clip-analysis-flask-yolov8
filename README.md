# football-clip-analysis-flask-yolov8

## Demo
https://github.com/maldiharyojudanto/football-clip-analysis-flask-yolov8/assets/76139419/41a9adf7-aa21-4e57-9dec-6348e2b60758

## Sistem desain

Aplikasi football-clip-analysis-flask-yolov8 dapat **mendeteksi player (beserta warna jersey), bola, ball possession, dan passing** menggunakan model **Yolov8 dan traditional rule based** yang telah diintegrasikan dengan **framework Flask**.

## Keterangan file .py :
1. File 'main.py' adalah untuk deteksi objek player dan ball
    - Input video
    - Output video (.mp4)
    - Menggunakan model yang sudah di training sebelumnya yang disimpan pada folder 'weigths'
2. File 'keyframe.py' adalah untuk mengekstraksi gambar dalam video
    - Input video dan K, dimana K adalah jumlah frame yang akan di ekstrak
    - Output frame (.jpg) disimpan pada folder 'output/image/keyframes'
3. File 'socernetv2.py' adalah file untuk mengambil/mengunduh dataset pada server soccernetv2
    - Output video train dan test soccernetv2
4. File 'katna.py adalah untuk mengekstraksi gambar dalam video menggunakan library katna (...link...)
    - Input video dan jumlah citra yang akan diekstrak
    - Output frame (.jpg)
5. File 'sbd.py' adalah untuk memperpendek durasi video dengan menghitung perbedaan antar frame n dengan frame n+1
    - Input video dan treshold (default 11)
    - Output frame (.jpg) dan video (.mp4)
6. File 'oneframepersecond.py' adalah untuk mengekstraksi frame dari video dan diambil satu frame setiap detik
    - Input video hasil sbd (clip)
    - Output frame (.jpg)
7. File 'playerextract.py' adalah untuk megekspor object dari satu frame image untuk nantinya digunakan sebagai dataset warna jersey
    - Input frame (.jpg)
    - Output frame (.jpg)
8. File 'modeltraining.py' adalah untuk membuat model training (proses training)
    - Input dataset yang sudah dianotasi (dari roboflow)
    - Output model deteksi objek best.pt

## Keterangan model .pt yang telah di-train (link model terdapat di [petunjuk pemakaian](https://github.com/maldiharyojudanto/football-clip-analysis-flask-yolov8?tab=readme-ov-file#petunjuk-pemakaian-)) :
1. 'soccer-detection-v2-best-n-200-aug'
    - soccer-detection-v2 : Dataset yang diambil dari roboflow
    - best : Model yang terbaik hasil training
    - n : Model YOLOv8 nano
    - 200 : Jumlah citra dalam dataset sebanyak 200 citra
    - aug : Dataset dilakukan augmentasi
    - class/label : {0: '0', 1: '1', 2: '2', 3: 'Player', 4: 'coach', 5: 'person', 6: 'soccer-ball', 7: 'sports ball'}
    - hyperparameter : {epochs=100, batch_size=16, imgsz=640}
2. 'football-scouting-best-m-1000-noaug-segonlyplayer'
    - football-scouting : Dataset hasil anotasi manual menggunakan roboflow
    - best : Model yang terbaik hasil training
    - m : Model YOLOv8 medium
    - 1000 : Jumlah citra dalam dataset sebanyak 1000 citra
    - noaug : Dataset tidak dilakukan augmentasi
    - segonlyplayer : Objek yang dianotasi menggunakan segmentasi hanya player
    - class/label : {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
    - hyperparameter : {epochs=100, batch_size=16, imgsz=640}
3. 'football-scouting-best-m-1000-noaug-segplayerball'
    - football-scouting : Dataset hasil anotasi manual menggunakan roboflow
    - best : Model yang terbaik hasil training
    - m : Model YOLOv8 medium
    - 1000 : Jumlah citra dalam dataset sebanyak 1000 citra
    - noaug : Dataset tidak dilakukan augmentasi
    - segplayerball : Objek yang dianotasi menggunakan segmentasi player dan bola
    - class/label : {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
    - hyperparameter : {epochs=100, batch_size=16, imgsz=640}
4. 'football-scouting-best-x-1000-noaug-segplayerball'
    - football-scouting : Dataset hasil anotasi manual menggunakan roboflow
    - best : Model yang terbaik hasil training
    - x : Model YOLOv8 extra large
    - 1000 : Jumlah citra dalam dataset sebanyak 1000 citra
    - noaug : Dataset tidak dilakukan augmentasi
    - segplayerball : Objek yang dianotasi menggunakan segmentasi player dan bola
    - class/label : {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
    - hyperparameter : {epochs=100, batch_size=16, imgsz=640}
5. 'football-players-detection-best-x-663-aug-noseg'
    - football-players-detection : Dataset hasil mencari di roboflow
    - best : Model yang terbaik hasil training
    - x : Model YOLOv8 extra large
    - 663 : Jumlah citra dalam dataset sebanyak 663 citra
    - aug : Dataset dilakukan augmentasi
    - noseg : Objek yang dianotasi tidak menggunakan segmentasi (hanya bounding box)
    - class/label : {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
    - hyperparameter : {epochs=100, batch_size=16, imgsz=640}
6. 'football-scouting-best-m-1695-aug-segonlyplayer'
    - football-scouting : Dataset hasil anotasi manual menggunakan roboflow
    - best : Model yang terbaik hasil training
    - m : Model YOLOv8 medium
    - 1695 : Jumlah citra dalam dataset sebanyak 1695 citra
    - aug : Dataset dilakukan augmentasi (flip=horizontal, saturation, brightness)
    - segonlyplayer : Objek yang dianotasi menggunakan segmentasi hanya player
    - class/label : {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
    - hyperparameter : {epochs=100, batch_size=16, imgsz=640}
7. 'football-jersey-color-best-ncls-3192-noaug-noseg'
    - football-jersey-color : Dataset hasil anotasi manual menggunakan roboflow
    - best : Model yang terbaik hasil training
    - ncls : Model YOLOv8 nano classify
    - 3192 : Jumlah citra dalam dataset sebanyak 3192 citra
    - noaug : Dataset tidak dilakukan augmentasi
    - noseg : Dataset tidak dilakukan segmentasi (hanya bounding box)

## Petunjuk pemakaian :
1. Buka terminal
2. Pastikan CLI sudah mengarah ke folder root project ini
    contoh file folder project ini di 'Desktop/TA', 
    maka perintah untuk ke folder tersebut adalah 'cd Desktop/TA'
3. Buat folder 'weights' di root foolder
4. Download weights di [sini](https://drive.google.com/drive/folders/14HF1AErJAaSnmk8jtjDufTEiVnECrVBZ?usp=sharing)
5. Paste hasil download ke folder 'weights' yang sudah dibuat sebelumnya
6. Download video untuk dilakukan prediksi (link dataset di [bawah](https://github.com/maldiharyojudanto/football-clip-analysis-flask-yolov8?tab=readme-ov-file#link-dataset-))
7. Ekstrak video untuk dilakukan prediksi/testing
8. Jalankan file dengan perintah 'python app.py'
9. Aplikasi berjalan di localhost:8000 atau 127.0.0.1:8000 (bisa diatur di app.py)
10. Silakan upload video yang telah diekstrak dan enjoy inference

## Link dataset :
1. https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout
2. https://www.soccer-net.org/data
3. https://drive.google.com/file/d/1UYEurzB6ZRJUkn75yQJ_yh3YYfPpW4wh/view (SOD Dataset)
4. https://universe.roboflow.com/school-zlexb/football-scouting/model/1 (anotasi/label player dan bola)
5. https://universe.roboflow.com/school-zlexb/football-jersey-color/dataset/5 (anotasi/label warna jersey)

## File test dfl-bundesliga-data-shootout (untuk dilakukan prediksi) :
1. 0b1495d3_1.mp4 : cocok karena passing
2. 2f54ed1c_0.mp4 : cocok karena passing
3. 4dae79a9_0.mp4 : cocok karena passing
4. 9d3c239b_0.mp4 : cocok karena passing
5. ec9f4e2b_1.mp4 : cocok karena passing dan umpan lambung

## Output (contoh) : [di sini](https://drive.google.com/drive/folders/1ZMHMoCTyfX0gP7io4abAmOvHOJi87Gj2?usp=sharing)

Bandung, 17 Desember 2023

Aldi ❤️ Made with heart
