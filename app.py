from ultralytics import YOLO
from modules.boundingbox_module import gambar_boundingbox_jersey, gambar_boundingbox_bola, deteksi_player_ballpossession, gambar_segitiga_pemain, keterangan_ballpossession, hitung_total_ballpossession, deteksi_player_passheading
from modules.jersey_module import klasifikasi_warnajersey
from modules.hardwarecheck import pythyoloversion, cudagpu

import cv2
import os
# import time
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


def inferenceVideo(video_path, jumlah_frame):
    print("\nSukses Upload Video!\n")

    pythyoloversion()
    cudagpu()

    # Load model
    model = YOLO('weights/football-scouting-best-m-1695-aug-segonlyplayer.pt')
    # results = model('sample_2.jpg', show=True, save=True) # Contoh prediksi pada single image

    # Membaca file video.mp4
    VIDEO_PATH = video_path # Source video dalam lokal
    cap = cv2.VideoCapture(VIDEO_PATH) # 0 = webcam, 1 = external webcam, VIDEO_PATH = lokasi video lokal

    # Membaca detail frame video
    file_base = os.path.basename(VIDEO_PATH) # Get nama direktori file dan nama file
    file_name = os.path.splitext(file_base) # Get nama file saja 
    frame_jumlah = cap.get(cv2.CAP_PROP_FRAME_COUNT) # Get total frame dalam file video masukan
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS (Frame Per Second) dalam file video masukan
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # Get ukuran width dari frame video masukan
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # Get ukuran height dari frame video masukan

    # Menampilkan informasi video
    print("============= DETAIL FILE VIDEO =============")
    print(f"Nama File Video :", file_name, 
        "\nTotal Frame :", int(frame_jumlah), 
        "\nFPS :", int(fps), 
        "\nDurasi Video (Detik) :", frame_jumlah/fps,
        "\nOriginal Ukuran Frame :", int(frame_width), int(frame_height),
        "\nModel Label/Kelas :", model.names),
    print("=============================================")

    # Simpan video ke ukuran 1280x720 (supaya lebih efisien & kompresi ukuran file)
    NEW_FRAME_WIDTH = 1280
    NEW_FRAME_HEIGHT = 720
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # Format video #H264 dll
    OUTPUT_PATH_FOLDER = os.path.join("static/output/videos\playerball", "football-scouting-best-m-1695-aug-segonlyplayer.pt") # Lokasi simpan video hasil inference
    try:
        os.makedirs(OUTPUT_PATH_FOLDER)
        print("Folder %s terbuat!" % OUTPUT_PATH_FOLDER)
    except FileExistsError:
        print("Folder %s telah tersedia" % OUTPUT_PATH_FOLDER)
    OUTPUT_PATH_VIDEOS = OUTPUT_PATH_FOLDER+f"/{file_name[0]} output.mp4"
    # out = cv2.VideoWriter(OUTPUT_PATH_VIDEOS, fourcc, fps, (NEW_FRAME_WIDTH, NEW_FRAME_HEIGHT)) # Untuk simpan
    out = cv2.VideoWriter(OUTPUT_PATH_VIDEOS, fourcc, fps, (int(frame_width), int(frame_height))) # Untuk simpan

    #################
    frame_nomor = 0

    # ball posession
    total_ballpossession = []
    warnajerseyterdeteksi_temp = []
    total_possession = []
    warnatext_possession = []

    # passing
    aktivitaswarna_temp = 0
    playerid_temp = 0
    riwayatpass = {}
    riwayatid = []
    aktivitaspassing = []
    passingtotal_temp = {}
    pass_temp = 0
    passwarnasekarang_temp = 0
    totalpasswarnasekarang = 0
    passingtotal_tim = []
    passwarnaberikut_temp = 0
    #################

    print("Inference dimulai ...")
    while frame_nomor < jumlah_frame and frame_nomor < int(frame_jumlah):
    # while cap.isOpened(): # True
        # start = time.time()
        success, frame = cap.read()     # Membaca frame saat ini yang telah diekstrak
        frame_nomor += 1

        if success:
            # Jalankan inference/prediksi pada frame saat ini dan persisting tracks between frames 
            # results = model(frame, imgsz=1280, conf=0.6)
            results = model.track(frame, imgsz=1280, conf=0.1, persist=True, tracker="bytetrack.yaml")

            # Menampilkan hasil prediksi (per bunding box) pada frame ini
            for r in results:
                boxes = r.boxes

                player_list = []
                bola_list = []
                playerwarnajersey_list = []

                bola_terdeksi = False

                jumlah_warnajerseyterdeteksi = 0

                # Setiap objek yang ada di frame ini
                for box in boxes:
                    # print(box)
                    # print(box.cls[0])
                    kelas = int(box.cls[0])
                    id_objek = int(box.id[0])
                    # print(id_objek)
                    # print(kelas)
                    # conf = int(box.conf[0])
                    x, y, w, h = box.xywh[0] # xcenter, ycenter, width, height
                    x1, y1, x2, y2 = box.xyxy[0] # x1 (xmin), y1 (ymin), x2 (xmax), y2 (ymax)
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert setiap value ke int
                    x_tengah, y_tengah, w, h = int(x), int(y), int(w), int(h) # convert setiap value ke int
                    # print(x_tengah, y_tengah)

                    if kelas == 2: # jika objek ini adalah player
                        playerwarnajersey = klasifikasi_warnajersey(frame[y1:y2, x1:x2]) # crop pemain sesuai ymin:ymax, xmin:xmax
                        # print(playerwarnajersey)
                        # Jika panjang jersey terdeteksi yang unique kurang dari 2 dan warna jersey tidak dalam warnajersey temp dan jumlah warna jersey terdeteksi < 2 maka 
                        if (len(warnajerseyterdeteksi_temp) < 2) and (playerwarnajersey not in warnajerseyterdeteksi_temp) and (jumlah_warnajerseyterdeteksi < 2):
                            jumlah_warnajerseyterdeteksi += 1
                            warnajerseyterdeteksi_temp.append(playerwarnajersey)
                        else:
                            pass
                        # print(warnajerseyterdeteksi_temp)

                        for i in warnajerseyterdeteksi_temp: # Jika warna terdapat dalam warnajersy
                            if i == playerwarnajersey: # maka warna i sama dengan tebakan warna jersey maka gambar
                                bbox_frame, player_xyxywh = gambar_boundingbox_jersey(frame, kelas, id_objek, i, x1, y1, x2, y2, x_tengah, y_tengah, w, h)
                                player_list.append(player_xyxywh)
                                playerwarnajersey_list.append(playerwarnajersey)
                            else:
                                pass

                    elif (kelas == 0) and (bola_terdeksi == False): # Jika objek ini adalah bola dan bola terdeteksi dalam frame ini masih false
                        bbox_frame, bola_xy = gambar_boundingbox_bola(frame, kelas, id_objek, x1, y1, x2, y2, x_tengah, y_tengah)
                        bola_list.append(bola_xy)
                        bola_terdeksi = True
                    else:
                        bbox_frame = frame

                frame_copy = bbox_frame.copy() # duplikat frame bbox_frame yang sudah ada boundingbox player dan bola (jika terdeteksi)
                text_player_ballposession = "-" # default value text ball posession
                aktivitas_player = "-" # default value text aktivitas player

                if bola_list != []: # jika list bola tidak kosong (bola terdeteksi)
                    player_ballposession, playerwarnajersey_ballpossession = deteksi_player_ballpossession(player_list, bola_list, playerwarnajersey_list, 35) # 35 = jarak antar objek dalam satuan pixel
                    text_player_ballposession, warnatext_player_ballposession = keterangan_ballpossession(playerwarnajersey_ballpossession) # keterangan siapa yang membawa bola sekarang (jika bola dan player terdeteksi membawa bola)
                    if (playerwarnajersey_ballpossession != []):
                        total_ballpossession.append(playerwarnajersey_ballpossession) # masukkan warna ke array
                    else:
                        pass
                    # print(f"Total Ball Posession : {total_ballpossession}")
                    total_possession, warnatext_possession = hitung_total_ballpossession(total_ballpossession) # hitung kemunculan dari warna
                    
                    if player_ballposession != []: # Jika terdapat player yang menguasai bola
                        bbox_frame_copy = gambar_segitiga_pemain(frame_copy, player_ballposession) # gambar segitiga diatas pemain
                        aktivitas_player, id_player = deteksi_player_passheading(player_ballposession, bola_list, 20, 35) # pemain sedang heading atau dribbling dan id pemainnya

                        if id_player not in riwayatpass:
                            riwayatpass[id_player] = 0

                        if playerid_temp == 0 and aktivitaswarna_temp == 0: # Jika player id temp masih 0 dan aktivitas warna temp masih -
                            playerid_temp = id_player
                            aktivitaswarna_temp = playerwarnajersey_ballpossession
                            aktivitaspassing.append(playerwarnajersey_ballpossession)
                            riwayatid.append(id_player)
                        elif playerid_temp != id_player and aktivitaswarna_temp == playerwarnajersey_ballpossession:
                            riwayatpass[playerid_temp] = riwayatpass[playerid_temp] + 1
                            playerid_temp = id_player
                            aktivitaswarna_temp = playerwarnajersey_ballpossession
                            aktivitaspassing.append(playerwarnajersey_ballpossession)
                            riwayatid.append(id_player)
                            # print("Passing/Heading sukses")
                        elif playerid_temp != id_player and aktivitaswarna_temp != playerwarnajersey_ballpossession:
                            playerid_temp = id_player
                            aktivitaswarna_temp = playerwarnajersey_ballpossession
                            aktivitaspassing = []
                            aktivitaspassing.append(playerwarnajersey_ballpossession)
                            riwayatid.append(id_player)
                            # print("Passing/Heading gagal")
                        else:
                            pass
                        cv2.putText(bbox_frame_copy, f"Ball Possession :", (50,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(bbox_frame_copy, f"{text_player_ballposession}", (285,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warnatext_player_ballposession, 2)
                        cv2.putText(bbox_frame_copy, f"Player Activity : {aktivitas_player}", (50,240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        bbox_frame_copy = frame_copy
                        cv2.putText(bbox_frame_copy, f"Ball Possession : {text_player_ballposession}", (50,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(bbox_frame_copy, f"Player Activity : {aktivitas_player}", (50,240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    bbox_frame_copy = frame_copy
                    cv2.putText(bbox_frame_copy, f"Ball Possession : {text_player_ballposession}", (50,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(bbox_frame_copy, f"Player Activity : {aktivitas_player}", (50,240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                for i in aktivitaspassing:
                    # print(i)
                    pass_temp = aktivitaspassing.count(i)-1
                    passwarnasekarang_temp = i
                
                if passwarnaberikut_temp != [] and passwarnaberikut_temp != passwarnasekarang_temp:
                    passingtotal_tim.append(totalpasswarnasekarang)
                    passingtotal_temp[passwarnaberikut_temp] = passingtotal_temp[passwarnaberikut_temp] + totalpasswarnasekarang
                else:
                    pass

                if aktivitaspassing != []:
                    passwarnaberikut_temp = aktivitaspassing[0]
                    totalpasswarnasekarang = pass_temp
                else:
                    pass

                if passwarnasekarang_temp not in passingtotal_temp:
                    passingtotal_temp[passwarnasekarang_temp] = 0
                else:
                    pass

                # print(passingtotal_temp[passwarnaberikut_temp])
                # print(passingtotal_temp[passwarnasekarang_temp])

                # print(f"\nPassing Total Tim : {passingtotal_tim}")
                # print(f"Passing Total Temporary : {passingtotal_temp}")
                print(f"Aktivitas Passing : {aktivitaspassing}")
                # print(f"Aktivitas Warna : {aktivitaswarna_temp}")
                # print(f"Player ID : {playerid_temp}")
                # print(f"Passing Total : {passingtotal_temp}")
                # print(f"Pass Tim Temporary : {pass_temp}")
                # print(f"Pass Warna Sekarang : {passwarnasekarang_temp}")
                # print(f"Total Pass Tim Sekarang : {totalpasswarnasekarang}")
                # print(f"Passing Total Tim : {passingtotal_tim}")
                # print(f"Passing Warna Berikutnya : {passwarnaberikut_temp}")
                print(f"\nBall Posession : {text_player_ballposession}")
                print(f"Ball Possession by ID : {riwayatid}")
                print(f"Total Posession : {total_possession}")
                print(f"Player Activity {aktivitas_player}")
                print(f"Passing : {passingtotal_temp}")
                print(f"Passing by ID: {riwayatpass}")
                print(f"Passing by Jersey Color : {totalpasswarnasekarang}")

                cv2.putText(bbox_frame_copy, f"Ball Possession by ID :", (50,160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if riwayatid != []:
                    cv2.putText(bbox_frame_copy, f"{riwayatid}", (350,160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.putText(bbox_frame_copy, f"Total Possession :", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if total_possession != [] and warnatext_possession != []:
                        pxke = 300
                        for i, j in enumerate(total_possession):
                            cv2.putText(bbox_frame_copy, f"{j}%", (pxke,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warnatext_possession[i], 2)
                            pxke += 100
                            
                cv2.putText(bbox_frame_copy, f"Passing  : ", (50,280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                pxke = 200
                for keys, values in passingtotal_temp.items():
                    if keys != 0:
                        cv2.putText(bbox_frame_copy, f"{values} Pass", (pxke,280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, keys, 2)
                        pxke += 150
                
                cv2.putText(bbox_frame_copy, f"Passing by ID :", (50,320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if riwayatid != []:
                    cv2.putText(bbox_frame_copy, f"{riwayatpass}", (280,320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Resize dari 1920x1080 ke 1280x720
            # resized_frame = cv2.resize(bbox_frame_copy, (NEW_FRAME_WIDTH, NEW_FRAME_HEIGHT))
            resized_frame = bbox_frame_copy
            # end = time.time()
            # fps_count = 1/(end-start)
            # cv2.putText(resized_frame, f"{int(fps)} FPS", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(resized_frame, f"{int(fps)} FPS", (50,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Simpan/add per frame ke format video
            print(f"Memproses Frame Urutan ke", frame_nomor)
            out.write(resized_frame)

            # # Visualize the results on the frame
            # annotated_frame = results[0].plot()

            # # Resize dari 1920x1080 ke 1280x720
            # resized_frame = cv2.resize(resized_frame, (NEW_FRAME_WIDTH, NEW_FRAME_HEIGHT))

            # # Tampilkan hasil di layar
            # cv2.imshow("{} Tracking".format(file_name[0]), resized_frame)
            # # Untuk menghentikan looping ekstraksi frame dari video dengan menekan 'q'
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
        else:
            # Berhenti ketika sampai frame terakhir
            break

    print("Inference berakhir ...")
    # Release the video capture object and close the display window
    print("\nOutput video telah berhasil disimpan pada '{}!'".format(OUTPUT_PATH_VIDEOS))
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

    return

EKSTENSI_DIPERBOLEH = ['mp4']
def ekstensiFile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EKSTENSI_DIPERBOLEH

@app.route('/upload', methods=['POST'])
def upload():
    # Jika video tidak ada dalam request
    if 'video' not in request.files:
        return "No video found"
    
    video = request.files['video'] # video di-assign dengan request.files['video'] atau kiriman video
    jumlahframe = request.form['numberframe'] # jumalhframe di-assign dengan request.form['numberframe'] atau kiriman dari field jumlah frame
    # print(f"{jumlahframe}")

    # Jika video filename kosong atau form jumlahframe kosong
    if video.filename == "" or jumlahframe == "":
        return "No video selected or number of frames not filled (invalid)"
    
    # Jika vide ada dan video ekstensi sesuai dalam daftar
    if video and ekstensiFile(video.filename):
        jf = int(jumlahframe)
        video.save('uploads/' + video.filename)
        video_path = os.path.join('uploads', video.filename)
        # print(video_path)
        inferenceVideo(video_path, jf)
        file_name = os.path.splitext(video.filename)
        # print(file_name[0])
        return render_template('result.html', video_name=file_name[0], jumlah_frame=jf) # Return result.html
    else:
        return "File type does not match"

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=8000) # Host dan Port bisa disesuaikan