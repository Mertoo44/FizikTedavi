import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Fizik Tedavi KDS", page_icon="🏥", layout="wide")

try:
    model = joblib.load('fiziktedavi_model.pkl')
    skorlar = joblib.load('model_skorlari.pkl')
except:
    st.error("Model dosyaları bulunamadı! Lütfen önce 'model_egit.py' dosyasını çalıştırın.")
    st.stop()

st.title("🏥 Ortopedik Anomali Tespit ve Analiz Sistemi")
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
        st.info("Girilen Değerler:")
        st.dataframe(input_df, hide_index=True)
        
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

    st.divider()
    st.subheader("📈 Algoritma Performans Karşılaştırması")
    skor_df = pd.DataFrame(list(skorlar.items()), columns=['Algoritma', 'Başarı Oranı'])
    skor_df = skor_df.set_index('Algoritma')
    st.bar_chart(skor_df)
    st.caption("Bu grafik, veri seti üzerinde eğitilen 3 farklı algoritmanın başarı oranlarını kıyaslamaktadır.")

# ==========================================
# SEKME 2: VERİ ANALİZİ 
# ==========================================
with tab2:
    st.header("Veri Seti İstatistikleri")
    
    dosya_yolu = "column_2C.csv"
    
    if os.path.exists(dosya_yolu):
        df = pd.read_csv(dosya_yolu)
        
        df.columns = ['Pelvik_İnsidans', 'Pelvik_Eğim', 'Lumbar_Lordoz_Açısı', 
                      'Sakral_Eğim', 'Pelvik_Yarıçap', 'Spondilolistezis_Derecesi', 'Durum']
        
        # 1. BÖLÜM
        st.subheader("1. Veri Setine Genel Bakış")
        st.write(f"Toplam Kayıt: **{df.shape[0]}** | Özellik Sayısı: **{df.shape[1]}**")
        
        st.dataframe(df.head(10)) 
        
        st.caption("ℹ️ Tabloda veri setinin ilk 10 satırı örnek olarak gösterilmektedir.")

        # 2. BÖLÜM
        st.subheader("2. İstatistiksel Özellikler")
        st.write(df.describe())
        st.caption("ℹ️ **count:** Veri sayısı, **mean:** Ortalama, **std:** Standart sapma, **min-max:** En düşük ve en yüksek değerler.")

        # 3. BÖLÜM
        st.subheader("3. Hasta Dağılımı")
        col_pie1, col_pie2 = st.columns([1, 2])
        dagilim = df['Durum'].value_counts().rename(index={'Abnormal': 'Anormal'})
        
        with col_pie1:
            st.dataframe(dagilim)
        with col_pie2:
            st.bar_chart(dagilim)
        st.caption("ℹ️ Veri setindeki Anormal (Hasta) ve Normal (Sağlıklı) bireylerin sayısal dağılımı.")

        # 4. BÖLÜM
        st.subheader("4. Değişken İlişkileri")
        ozellikler = df.columns[:-1].tolist()
        c1, c2 = st.columns(2)
        x_val = c1.selectbox("X Ekseni", ozellikler, index=0)
        y_val = c2.selectbox("Y Ekseni", ozellikler, index=5)
        st.scatter_chart(df, x=x_val, y=y_val, color='Durum', size=20)
        st.caption(f"ℹ️ Yukarıdaki grafik **{x_val}** ile **{y_val}** arasındaki ilişkiyi gösterir. Noktaların rengi hastalık durumunu belirtir.")
        
    else:
        st.error(f"'{dosya_yolu}' dosyası bulunamadı! Lütfen CSV dosyasını klasöre atın.")