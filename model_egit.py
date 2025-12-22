# Dosya adı: model_egit.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

# Dosya adı
dosya_adi = "column_2C_weka.arff"

def arff_oku_ve_turkcelestir(dosya_yolu):
    data = []
    veri_basladi = False
    with open(dosya_yolu, 'r', encoding='utf-8') as f:
        for satir in f:
            satir = satir.strip()
            if not satir: continue
            if not veri_basladi:
                if satir.lower().startswith("@data"):
                    veri_basladi = True
                continue
            data.append(satir.split(','))

    sutunlar = ['Pelvik_İnsidans', 'Pelvik_Eğim', 'Lumbar_Lordoz_Açısı', 
                'Sakral_Eğim', 'Pelvik_Yarıçap', 'Spondilolistezis_Derecesi', 'Durum']
    
    df = pd.DataFrame(data, columns=sutunlar)
    for col in sutunlar[:-1]:
        df[col] = pd.to_numeric(df[col])
    return df

print("3 Farklı Model eğitiliyor ve karşılaştırılıyor...")

if os.path.exists(dosya_adi):
    df = arff_oku_ve_turkcelestir(dosya_adi)
    X = df.drop('Durum', axis=1)
    y = df['Durum']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- SADECE 3 MODEL ---
    modeller = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (Destek Vektör)": SVC(probability=True),
        "KNN (En Yakın Komşu)": KNeighborsClassifier(n_neighbors=5)
    }

    sonuclar = {}
    
    # Hepsini tek tek eğitip skorunu ölçelim
    for isim, model in modeller.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        basari = accuracy_score(y_test, y_pred)
        sonuclar[isim] = basari
        print(f"👉 {isim} Başarısı: %{basari * 100:.2f}")

    # En iyi modeli bul ve kaydet
    en_iyi_model_ismi = max(sonuclar, key=sonuclar.get)
    en_iyi_model = modeller[en_iyi_model_ismi]
    
    print(f"\n🏆 EN İYİ MODEL: {en_iyi_model_ismi}")
    
    # Dosyaları kaydet
    joblib.dump(en_iyi_model, 'fiziktedavi_model.pkl')
    joblib.dump(sonuclar, 'model_skorlari.pkl')
    
    print("💾 Dosyalar güncellendi! Şimdi arayüzü çalıştırabilirsin.")
    
else:
    print("HATA: ARFF dosyası bulunamadı!")