import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veriyi Okuma ve Train Setini Ayırma
# fabrika_clean.csv dosyasının kod ile aynı dizinde olduğu varsayılmaktadır.
df = pd.read_csv("fabrika_clean.csv")

# İlk 132 ay Train verisi (Ocak 2011 - Aralık 2019)
df_train = df.iloc[:108].copy()

# Korelasyon hesabına katılmaması için tarih sütunlarını çıkarıyoruz
cols_to_drop = ["Year", "Month"]
df_train_features = df_train.drop(columns=cols_to_drop, errors="ignore")

# Tüm figürler için Times New Roman font ailesini zorunlu kılıyoruz
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# =============================================================================
# HARİTA 1: TÜM DEĞİŞKENLER İÇİN KORELASYON HARİTASI (600 DPI)
# =============================================================================
corr_all = df_train_features.corr()

plt.figure(figsize=(18, 14))
sns.heatmap(corr_all, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
            vmin=-1, vmax=1, annot_kws={"size": 10}, square=True, linewidths=.5)

plt.title("Correlation Heatmap - All Features\n(Train Data Only: Jan 2011 - Dec 2019)", 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()

# 600 dpi olarak kaydet
plt.savefig("heatmap_all_features_600dpi.png", dpi=600, bbox_inches='tight')
plt.close()
print("1. Tüm değişkenler için korelasyon haritası oluşturuldu: 'heatmap_all_features_600dpi.png'")


# =============================================================================
# HARİTA 2: SEÇİLİ 3 ÖZELLİK + HEDEF DEĞİŞKEN İÇİN KORELASYON HARİTASI (600 DPI)
# =============================================================================
# Kullanmak istediğiniz 3 özellik ve tahmin edilecek hedef (W-Water)
selected_columns = [
    "TotalDyeingAmount", 
    "TotalPackingAmaount", 
    "YarnConsumptionPerKgGas", 
    "W-Water"
]

# Sadece bu sütunları filtrele (yine sadece Train seti üzerinden)
df_train_selected = df_train[selected_columns]
corr_selected = df_train_selected.corr()

# Harita 2'nin çizimi
plt.figure(figsize=(8, 6))
# annot_kws boyutunu ve formatını Harita 1 ile birebir AYNI yaptık (size: 10, fmt: .2f)
sns.heatmap(corr_selected, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
            vmin=-1, vmax=1, annot_kws={"size": 10}, 
            square=True, linewidths=.5)

# Başlık ve eksen font boyutlarını Harita 1 ile birebir AYNI yaptık
plt.title("Correlation Heatmap - Selected Features vs Target\n(Train Data Only)", 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=25, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()

# 600 dpi olarak kaydet
plt.savefig("heatmap_selected_features_600dpi.png", dpi=600, bbox_inches='tight')
plt.close()
print("2. Seçili özellikler için korelasyon haritası oluşturuldu: 'heatmap_selected_features_600dpi.png'")