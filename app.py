from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Inisialisasi penghubung Gemini cara baru
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

app = Flask(__name__)
model_ml = joblib.load('model_klasifikasi_finexa.pkl')

penerjemah_jawaban = {
    "Pemula": 1, "Menengah": 2, "Berpengalaman": 3,
    "Dana kurang dari 3 bulan": 1, "3-6 bulan": 2, "Dana lebih dari 6 bulan": 3,
    "Bisa disisihkan 10%": 1, "Bisa disisihkan 20%": 2, "Bisa disisihkan 30%": 3,
    "Tarik dana": 1, "Biarkan saja": 2, "Tambah investasi saat turun": 3,
    "< 1 tahun": 1, "1-3 tahun": 2, "3-5 tahun": 3, "Lebih dari 5 tahun": 4
}

@app.route('/predict-risk', methods=['POST'])
def predict_risk():
    try:
        data = request.json
        answers = data.get('answers')  # Menerima array angka: [3, 3, 3, 2, 3]
        
        # Langsung ambil angkanya dari array tanpa perlu diterjemahkan lewat teks
        angka_jawaban = [int(x) for x in answers]
        
        data_dict = {
            'skor_pengetahuan': angka_jawaban[0],
            'skor_kapasitas': angka_jawaban[1],
            'skor_hutang': angka_jawaban[2],
            'skor_risiko': angka_jawaban[3],
            'skor_waktu': angka_jawaban[4]
        }
        
        df_input = pd.DataFrame([data_dict])
        df_input = df_input[model_ml.feature_names_in_]
        
        # Biarkan model ML `.pkl` yang sudah kita latih bekerja secara murni!
        hasil_prediksi = model_ml.predict(df_input)
        
        print(f"Input Skor: {angka_jawaban} -> Hasil Prediksi ML: {hasil_prediksi[0]}")
        return jsonify({"risk_profile": hasil_prediksi[0]})
        
    except Exception as e:
        print("Error ML:", e)
        return jsonify({"risk_profile": "Moderat", "error": str(e)})
        
    except Exception as e:
        print("Error ML:", e)
        return jsonify({"risk_profile": "Moderat", "error": str(e)})

@app.route('/generate-advice', methods=['POST'])
def generate_advice():
    print("Menerima request LLM dari Express...")
    data = request.json
    risk_profile = data.get('risk_profile')
    
    # 1. Buka dan baca buku contekan katalog aset
    try:
        with open('katalog_aset.txt', 'r', encoding='utf-8') as file:
            buku_contekan = file.read()
    except FileNotFoundError:
        buku_contekan = "Gunakan pengetahuan umum untuk menyarankan instrumen investasi standar Indonesia."
    
    # 2. Sisipkan isi buku contekan ke instruksi Gemini
    perintah_untuk_gemini = f"""
    Kamu adalah penasihat keuangan profesional. Berikan saran alokasi aset untuk investor dengan profil risiko: {risk_profile}.
    
    PENTING: Kamu WAJIB memilih dan menyebutkan nama produk investasi secara spesifik berdasarkan katalog aset berikut ini:
    {buku_contekan}
    
    Kamu WAJIB mengembalikan jawaban HANYA dalam format JSON seperti contoh di bawah ini, tanpa tambahan teks apapun di awal atau di akhir:
    {{
        "expected_return": "tulis perkiraan return di sini, misal: 10 - 15% / tahun",
        "horizon_waktu": "tulis saran waktu di sini, misal: > 5 tahun",
        "alokasi_aset": [
            {{ "nama": "Sukuk Ritel (SR019)", "persentase": 60 }},
            {{ "nama": "Sucorinvest Money Market Fund", "persentase": 40 }}
        ]
    }}
    Pastikan total persentase di dalam alokasi_aset berjumlah persis 100.
    """
    
    try:
        respons_gemini = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=perintah_untuk_gemini
        )
        teks_jawaban = respons_gemini.text
        
        teks_bersih = teks_jawaban.replace('```json', '').replace('```', '').strip()
        hasil_akhir = json.loads(teks_bersih)

        print("Saran investasi selesai diracik!")
        return jsonify(hasil_akhir)
        
    except Exception as e:
        print("Error Gemini:", e)
        return jsonify({
            "expected_return": "Tidak tersedia",
            "horizon_waktu": "Tidak tersedia",
            "alokasi_aset": []
        })

if __name__ == '__main__':
    app.run(debug=True, port=5001)