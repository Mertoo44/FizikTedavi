# Dosya adı: app.py
import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. AYARLAR VE FONKSİYONLAR ---
st.set_page_config(page_title="Fizik Tedavi KDS", page_icon="🏥", layout="wide")

# ARFF okuma fonksiyonu
def arff_oku_ve_turkcelestir(dosya_yolu):
    data = []
    veri_basladi = False
    if not os.path.exists(dosya_yolu):
        return None
        
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

# Modelleri yükle
try:
    model = joblib.load('fiziktedavi_model.pkl')
    skorlar = joblib.load('model_skorlari.pkl')
except:
    st.error("Model dosyaları bulunamadı! Lütfen önce 'model_egit.py' dosyasını çalıştırın.")
    st.stop()

# --- 2. BAŞLIK VE SEKME YAPISI ---
st.title("🏥 Ortopedik Anomali Tespit ve Analiz Sistemi")

# Sekmeleri oluşturuyoruz
tab1, tab2 = st.tabs(["🩺 Tahmin Sistemi", "📊 Veri Analizi"])

# ==========================================
# SEKME 1: TAHMİN SİSTEMİ
# ==========================================
with tab1:
    st.markdown("### Hasta Durumu Tahmini")
    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.subheader("Veri Girişi")
        p_insidans = st.slider('Pelvik İnsidans', 26.0, 130.0, 60.0)
        p_egim = st.slider('Pelvik Eğim', -6.0, 50.0, 20.0)
        l_lordoz = st.slider('Lumbar Lordoz Açısı', 14.0, 126.0, 50.0)
        s_egim = st.slider('Sakral Eğim', 13.0, 122.0, 40.0)
        p_yaricap = st.slider('Pelvik Yarıçap', 70.0, 164.0, 110.0)
        s_derece = st.slider('Spondilolistezis Derecesi', -11.0, 419.0, 10.0)

        input_data = {
            'Pelvik_İnsidans': p_insidans,
            'Pelvik_Eğim': p_egim,
            'Lumbar_Lordoz_Açısı': l_lordoz,
            'Sakral_Eğim': s_egim,
            'Pelvik_Yarıçap': p_yaricap,
            'Spondilolistezis_Derecesi': s_derece
        }
        input_df = pd.DataFrame(input_data, index=[0])

    with col_result:
        st.subheader("Analiz Sonucu")
        st.info("Girilen Değerler:")
        # BURAYI DÜZELTTİM: Eski kod uyarı veriyordu, sadeleştirdik.
        st.dataframe(input_df)
        
        if st.button("Hastalığı Tahmin Et", type="primary"):
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
            durum = prediction[0]
            
            st.divider()
            if durum == 'Normal':
                st.success(f"✅ SONUÇ: {durum}")
                st.write("Hastanın değerleri **Normal** görünüyor.")
            else:
                if durum == 'Abnormal': durum = 'ANORMAL (Riskli)'
                st.error(f"⚠️ SONUÇ: {durum}")
                st.write("Hastada ortopedik bir anomali riski tespit edildi.")

            st.write("**Güven Oranı:**")
            probs_df = pd.DataFrame(probability, columns=model.classes_)
            probs_df = probs_df.rename(columns={'Abnormal': 'Anormal', 'Normal': 'Normal'})
            st.bar_chart(probs_df.T)

    # Model Karşılaştırma Grafiği
    st.divider()
    st.subheader("📈 Algoritma Performans Karşılaştırması")
    skor_df = pd.DataFrame(list(skorlar.items()), columns=['Algoritma', 'Başarı Oranı'])
    skor_df = skor_df.set_index('Algoritma')
    st.bar_chart(skor_df)
    st.caption("Bu proje kapsamında 3 farklı makine öğrenmesi algoritması test edilmiştir.")

# ==========================================
# SEKME 2: VERİ ANALİZİ
# ==========================================
with tab2:
    st.header("Veri Seti İstatistikleri ve Görselleştirme")
    
    # Veriyi tekrar oku
    df = arff_oku_ve_turkcelestir("column_2C_weka.arff")
    
    if df is not None:
        # 1. Veri Önizleme
        st.subheader("1. Veri Setine Genel Bakış")
        st.write(f"Veri setinde toplam **{df.shape[0]}** hasta kaydı ve **{df.shape[1]}** özellik bulunmaktadır.")
        
        # BURAYI DÜZELTTİM: Eski 'use_container_width' parametresini kaldırdım.
        st.dataframe(df.head(10)) 
        st.caption("İlk 10 satır gösterilmektedir.")

        # 2. İstatistiksel Özet
        st.subheader("2. İstatistiksel Özellikler")
        st.write("Ortalama, standart sapma, min-max değerleri:")
        st.write(df.describe())

        # 3. Sınıf Dağılımı
        st.subheader("3. Hasta Dağılımı (Normal vs Anormal)")
        col_pie1, col_pie2 = st.columns([1, 2])
        
        dagilim = df['Durum'].value_counts()
        dagilim = dagilim.rename(index={'Abnormal': 'Anormal', 'Normal': 'Normal'})
        
        with col_pie1:
            st.dataframe(dagilim)
        with col_pie2:
            st.bar_chart(dagilim)
            st.caption("Veri setindeki Anormal ve Normal hasta sayıları.")

        # 4. Korelasyon Analizi
        st.subheader("4. Değişkenler Arası İlişki Analizi")
        st.info("İki özellik arasındaki ilişkiyi görmek için aşağıdan seçim yapın.")
        
        ozellikler = df.columns[:-1].tolist()
        
        c1, c2 = st.columns(2)
        x_ekseni = c1.selectbox("X Ekseni", ozellikler, index=0)
        y_ekseni = c2.selectbox("Y Ekseni", ozellikler, index=5)
        
        st.scatter_chart(df, x=x_ekseni, y=y_ekseni, color='Durum', size=20)
        st.caption(f"{x_ekseni} ile {y_ekseni} arasındaki ilişki.")
        
    else:
        st.error("Veri dosyası (column_2C_weka.arff) bulunamadı! Lütfen dosyanın klasörde olduğundan emin olun.")