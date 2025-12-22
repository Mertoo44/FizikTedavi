# Dosya adı: model_egit.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

# DOSYA ADI (Artık CSV kullanıyoruz)
dosya_adi = "column_2C.csv"

print("Model eğitimi başlıyor...")

if os.path.exists(dosya_adi):
    # 1. CSV DOSYASINI OKU
    df = pd.read_csv(dosya_adi)
    
    # 2. SÜTUN İSİMLERİNİ TÜRKÇELEŞTİR (Sırası standarttır)
    df.columns = [
        'Pelvik_İnsidans', 
        'Pelvik_Eğim', 
        'Lumbar_Lordoz_Açısı', 
        'Sakral_Eğim', 
        'Pelvik_Yarıçap', 
        'Spondilolistezis_Derecesi', 
        'Durum'
    ]
    
    # 3. VERİYİ HAZIRLA
    X = df.drop('Durum', axis=1)
    y = df['Durum']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. MODELLERİ EĞİT (3 Tane)
    modeller = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (Destek Vektör)": SVC(probability=True),
        "KNN (En Yakın Komşu)": KNeighborsClassifier(n_neighbors=5)
    }

    sonuclar = {}
    
    print(f"Toplam {len(df)} kayıt üzerinde eğitim yapılıyor...")
    
    for isim, model in modeller.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        basari = accuracy_score(y_test, y_pred)
        sonuclar[isim] = basari
        print(f"👉 {isim} Başarısı: %{basari * 100:.2f}")

    # En iyi modeli seç ve kaydet
    en_iyi_model_ismi = max(sonuclar, key=sonuclar.get)
    en_iyi_model = modeller[en_iyi_model_ismi]
    
    print(f"\n🏆 ŞAMPİYON MODEL: {en_iyi_model_ismi}")
    
    joblib.dump(en_iyi_model, 'fiziktedavi_model.pkl')
    joblib.dump(sonuclar, 'model_skorlari.pkl')
    print("💾 Model ve skorlar başarıyla kaydedildi!")
    
else:
    print(f"HATA: '{dosya_adi}' dosyası klasörde bulunamadı! Lütfen ismini kontrol et.")