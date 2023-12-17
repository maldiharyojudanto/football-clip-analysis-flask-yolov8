from ultralytics import YOLO

from sklearn.cluster import KMeans
import numpy as np
import cv2

model = YOLO('weights/football-jersey-color-best-ncls-3192-noaug-noseg.pt')

# # Warna 
# pallete = {'biru': (0, 0, 128),
#         'hijau': (0, 128, 0),
#         'merah': (255, 0, 0),
#         'biru muda': (0, 192, 192),
#         'merah muda': (192, 0, 192),
#         'kuning': (192, 192, 0),
#         'hitam': (0, 0, 0),
#         'putih': (255, 255, 255)}

# # Fungsi deteksi warna dengan K-Means
# def detect_color(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     img = np.reshape(img, (img.shape[0]*img.shape[1], 3)) # Konversi dari 3D menjadi 2D
#     # print(img)

#     kmeans = KMeans(n_clusters=2, n_init='auto', max_iter=200) # Jalankan training model K-Means
#     prediksi = kmeans.fit(img) # Fitting ke img

#     labels = kmeans.labels_ 
#     centroid = kmeans.cluster_centers_ # Dapatkan titik tengah/centroid
#     labels = list(labels) # Dapatkan label (karena n_cluster = 2 maka labelnya range 0-1)
#     # print(labels)

#     persen = []

#     for i in range(len(centroid)): # centroid = n_cluster
#         total = labels.count(i) # Dijumlahkan setiap label pada setiap titik cluster
#         rata = total/(len(labels)) # Dirata-ratakan
#         persen.append(rata) # Dimasukkan ke array
#         # print(centroid)

#     # print(persen)

#     detected_color = centroid[np.argmin(persen)] # Centroid adalah nilai yang minimum dari hasil di atas
#     # print(detected_color)

#     list_of_colors = list(pallete.values())
#     assigned_color = closest_color(list_of_colors, detected_color)[0]
#     assigned_color = (int(assigned_color[2]), int(assigned_color[1]), int(assigned_color[0]))

#     if assigned_color == (0, 0, 0):
#         assigned_color = (128, 128, 128)

#     return assigned_color

# # Find the closest color to the detected one based on the predefined palette
# def closest_color(list_of_colors, color):
#     colors = np.array(list_of_colors)
#     color = np.array(color)
#     distances = np.sqrt(np.sum((colors-color)**2,axis=1))
#     index_of_shortest = np.where(distances==np.amin(distances))
#     shortest_distance = colors[index_of_shortest]
#     # print(shortest_distance)

#     return shortest_distance

def klasifikasi_warnajersey(imgplayer):
    """
    Daftar warna:
        {0: 'biru',
         1: 'biru muda',
         2: 'hijau',
         3: 'hitam',
         4: 'kuning',
         5: 'merah',
         6: 'putih'}
    """
    results = model(imgplayer, imgsz=64)
    kelas_pred = int(results[0].probs.top1)

    if kelas_pred == 0: # biru
        kelas = (0, 0, 128)
    elif kelas_pred == 1: # biru muda
        kelas = (0, 192, 192)
    elif kelas_pred == 2: # hijau
        kelas = (0, 128, 0)
    elif kelas_pred == 3: # hitam
        kelas = (0, 0, 0)
    elif kelas_pred == 4: # kuning
        kelas = (192, 192, 0)
    elif kelas_pred == 5: # merah
        kelas = (255, 0, 0)
    elif kelas_pred == 6: # putih
        kelas = (255, 255, 255)
    else: # merah muda
        kelas = (192, 0, 192)

    return kelas