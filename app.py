# Dosya adı: app.py
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image 

# --- AYARLAR ---
st.set_page_config(page_title="Fizik Tedavi KDS", page_icon="🏥", layout="wide")

# RESİM YÜKLEME FONKSİYONU
def resim_goster(dosya_adi, genislik=None, altyazi=None):
    if os.path.exists(dosya_adi):
        img = Image.open(dosya_adi)
        if genislik:
            st.image(img, width=genislik, caption=altyazi)
        else:
            st.image(img, use_column_width=True, caption=altyazi)
    else:
        pass

# Modelleri yükle
try:
    model = joblib.load('fiziktedavi_model.pkl')
    skorlar = joblib.load('model_skorlari.pkl')
except:
    st.error("Model dosyaları bulunamadı! Lütfen önce 'model_egit.py' dosyasını çalıştırın.")
    st.stop()

# --- BAŞLIK KISMI ---
col_logo, col_baslik = st.columns([1, 4])
with col_logo:
    resim_goster("banner.jpg", genislik=150) 
with col_baslik:
    st.title("🏥 Ortopedik Anomali Tespit Sistemi")
    st.markdown("**Makine Öğrenmesi Destekli Karar Destek Sistemi**")

tab1, tab2 = st.tabs(["🩺 Tahmin Sistemi", "📊 Veri Analizi ve Performans"])

# ==========================================
# SEKME 1: TAHMİN SİSTEMİ
# ==========================================
with tab1:
    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.subheader("Hasta Verileri")
        resim_goster("anatomi.jpg", altyazi="Omurga Açıları Referans Görseli")
        
        st.info("Lütfen hastanın radyolojik ölçümlerini giriniz:")
        
        p_insidans = st.slider('Pelvik İnsidans', 26.0, 130.0, 60.0)
        p_egim = st.slider('Pelvik Eğim', -6.0, 50.0, 20.0)
        l_lordoz = st.slider('Lumbar Lordoz Açısı', 14.0, 126.0, 50.0)
        s_egim = st.slider('Sakral Eğim', 13.0, 122.0, 40.0)
        p_yaricap = st.slider('Pelvik Yarıçap', 70.0, 164.0, 110.0)
        s_derece = st.slider('Spondilolistezis Derecesi', -11.0, 419.0, 10.0)

        input_df = pd.DataFrame({
            'Pelvik_İnsidans': [p_insidans],
            'Pelvik_Eğim': [p_egim],
            'Lumbar_Lordoz_Açısı': [l_lordoz],
            'Sakral_Eğim': [s_egim],
            'Pelvik_Yarıçap': [p_yaricap],
            'Spondilolistezis_Derecesi': [s_derece]
        })

    with col_result:
        st.subheader("Analiz Sonucu")
        
        if st.button("Hastalığı Tahmin Et", type="primary"):
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
            durum = prediction[0]
            
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                if durum == 'Normal':
                    st.success(f"✅ SONUÇ: {durum}")
                    st.write("Hastanın omurga yapısı **Sağlıklı** sınıfında değerlendirilmiştir.")
                else:
                    if durum == 'Abnormal': durum = 'ANORMAL (Riskli)'
                    st.error(f"⚠️ SONUÇ: {durum}")
                    st.write("Hastada **Disk Kayması veya Fıtık** riski tespit edilmiştir. Uzman hekim kontrolü önerilir.")
                
                st.write("**Yapay Zeka Güven Oranı:**")
                probs_df = pd.DataFrame(probability, columns=model.classes_)
                probs_df = probs_df.rename(columns={'Abnormal': 'Anormal', 'Normal': 'Normal'})
                st.bar_chart(probs_df.T)

            with res_col2:
                if durum == 'Normal':
                    resim_goster("saglikli.jpg", altyazi="Sağlıklı Omurga Örneği")
                else:
                    resim_goster("hasta.jpg", altyazi="Spondilolistezis (Kayma) Örneği")

    st.divider()
    st.subheader("📈 Algoritma Performans Karşılaştırması")
    skor_df = pd.DataFrame(list(skorlar.items()), columns=['Algoritma', 'Başarı Oranı'])
    skor_df = skor_df.set_index('Algoritma')
    st.bar_chart(skor_df)
    st.caption("Bu grafik, eğitim sırasında farklı algoritmaların test verisi üzerindeki başarı oranlarını gösterir.")

# ==========================================
# SEKME 2: VERİ ANALİZİ VE CONFUSION MATRIX
# ==========================================
with tab2:
    st.header("Veri Seti Analizi ve Model Performansı")
    
    dosya_yolu = "column_2C.csv"
    
    if os.path.exists(dosya_yolu):
        df = pd.read_csv(dosya_yolu)
        df.columns = ['Pelvik_İnsidans', 'Pelvik_Eğim', 'Lumbar_Lordoz_Açısı', 
                      'Sakral_Eğim', 'Pelvik_Yarıçap', 'Spondilolistezis_Derecesi', 'Durum']
        
        st.subheader("1. Veri Setine Genel Bakış")
        st.write(f"Toplam Kayıt: **{df.shape[0]}** | Özellik Sayısı: **{df.shape[1]}**")
        st.dataframe(df.head(10)) 

        st.subheader("2. İstatistiksel Özellikler")
        st.write(df.describe())

        st.subheader("3. Hasta Dağılımı")
        col_pie1, col_pie2 = st.columns([1, 2])
        dagilim = df['Durum'].value_counts().rename(index={'Abnormal': 'Anormal'})
        with col_pie1: st.dataframe(dagilim)
        with col_pie2: st.bar_chart(dagilim)

        st.subheader("4. Değişken İlişkileri")
        ozellikler = df.columns[:-1].tolist()
        c1, c2 = st.columns(2)
        x_val = c1.selectbox("X Ekseni", ozellikler, index=0)
        y_val = c2.selectbox("Y Ekseni", ozellikler, index=5)
        st.scatter_chart(df, x=x_val, y=y_val, color='Durum', size=20)

        st.divider()

        st.subheader("5. Karmaşıklık Matrisi (Tüm Veri Seti)")
        X_all = df.drop('Durum', axis=1)
        y_all = df['Durum']
        y_pred_all = model.predict(X_all)
        cm = confusion_matrix(y_all, y_pred_all, labels=model.classes_)
        
        # --- GÜNCELLEME BURADA ---
        # Grafiği küçültmek için sütun kullandık
        col_cm1, col_cm2 = st.columns([1, 2]) # 1 birim grafik, 2 birim boşluk
        
        with col_cm1:
            # figsize=(5, 4) yaparak fiziksel boyutunu küçülttük
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
            plt.ylabel('Gerçek Durum')
            plt.xlabel('Modelin Tahmini')
            st.pyplot(fig)
        
    else:
        st.error(f"'{dosya_yolu}' dosyası bulunamadı! Lütfen CSV dosyasını klasöre atın.")