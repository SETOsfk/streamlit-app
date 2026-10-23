import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
from scipy.stats import chi2_contingency, chisquare
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

d = pd.read_csv("Consumer_Shopping_Trends_2026.csv", sep=',', header=0, encoding="ISO-8859-1")
print(d.head())       # İlk 5 satırı göster
print(d.tail())       # Son 5 satırı göster
print(d.dtypes)       # Değişken türleri
print(d.describe())   # İstatistiksel özet
#Sütunlara Erişim
print(d['age'])
print(d[['age', 'gender']])
#Filtreleme
print(d[d['age'] > 25])
#Yeni Sütun Ekleme
d['yeni_yas'] = d['age'] + 5
print(d['yeni_yas'])
d = d.drop(columns=['yeni_yas'])
#Eksik değerlerin ve yüzdelerinin tespiti
print("Eksik Değer Sayısı")
kayıp = d.isnull().sum()
print(kayıp)
print("Eksik Değer Yüzdesi")
kayıpyuzde = (kayıp / len(d))
print(kayıpyuzde)
print(d.isnull().mean().round(3))
print("Tüm veri setindeki eksik değer sayısı:", d.isnull().sum().sum())
#Tekrar eden verilerin silinmesi
d = d.drop_duplicates()
print("Tekrarlar silindikten sonra boyut:", d.shape)
#Eksik değerleri doldurma
d['age'] = d['age'].fillna(d['age'].mean())
d['monthly_income'] = d['monthly_income'].fillna(d['monthly_income'].median())
d['daily_internet_hours'] = d['daily_internet_hours'].fillna(d['daily_internet_hours'].mean())
#Kategorik verileri en sık görülen değer ile doldurma
d['gender'] = d['gender'].fillna(d['gender'].mode()[0])
print(d.isna().sum())
#Interpolate ile doldurma
tarih = pd.date_range("2025-01-01", periods=10)
s = pd.Series([10, np.nan, 38, np.nan, 78, 45, 56, 34, np.nan, 65], index=tarih)
print("Orijinal seri:", s)
s_doldurulmus = s.interpolate()
print("Lineer interpolate:", s_doldurulmus)
s_doldurulmus = s.interpolate(method="time")
print("Time interpolate:", s_doldurulmus)
s_doldurulmus = s.interpolate(method="polynomial", order=2)
print("Polinom interpolate:", s_doldurulmus)
#Aykırı Değerleri Tespit Etme (IQR)
q1 = d['age'].quantile(0.25)
q3 = d['age'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print(d[d['age'] < lower].index)
print(d[d['age'] > upper].index)
#Her sayısal değişken için aykırı değer sayısı
aykırı_indeks = {}
sayısal = d.select_dtypes(include='number').columns
for col in sayısal:
    Q1 = d[col].quantile(0.25)
    Q3 = d[col].quantile(0.75)
    IQR = Q3 - Q1
    alt = Q1 - 1.5 * IQR
    ust = Q3 + 1.5 * IQR
    aykırı = d[(d[col] < alt) | (d[col] > ust)].index.tolist()
    aykırı_indeks[col] = aykırı
    print(col, "aykırı değer sayısı:", len(aykırı_indeks[col]))
#Aykırı değerleri NaN yapıp silme
d_temiz = d.dropna()
for col in sayısal:
    Q1 = d_temiz[col].quantile(0.25)
    Q3 = d_temiz[col].quantile(0.75)
    IQR = Q3 - Q1
    alt = Q1 - 1.5 * IQR
    ust = Q3 + 1.5 * IQR
    d_temiz.loc[(d_temiz[col] < alt) | (d_temiz[col] > ust), col] = np.nan
d_temiz1 = d_temiz.dropna()
print(d_temiz1.isnull().sum())
print("Temizlenmiş veri boyutu:", d_temiz1.shape)
d_temiz1.to_csv("d_temiz.csv")
#Aykırı değerleri ortalama ile değiştirme
d_temiz = d.copy()
for col in sayısal:
    Q1 = d[col].quantile(0.25)
    Q3 = d[col].quantile(0.75)
    IQR = Q3 - Q1
    alt = Q1 - 1.5 * IQR
    ust = Q3 + 1.5 * IQR
    ortalama = d[(d[col] >= alt) & (d[col] <= ust)][col].mean()
    d_temiz[col] = d_temiz[col].astype(float)
    d_temiz.loc[(d_temiz[col] < alt) | (d_temiz[col] > ust), col] = ortalama
print(d_temiz)

d = pd.read_csv("Consumer_Shopping_Trends_2026.csv", sep=',', header=0, encoding="ISO-8859-1")
#Aykırı Değerleri Tespit Etme-Z-Score Yöntemi (silme)
d_temiz = d.copy()
sayısal = d.select_dtypes(include='number').columns
for col in sayısal:
    ortalama = d[col].mean()
    std = d[col].std()
    z_skor = (d[col] - ortalama) / std
    d_temiz = d_temiz[z_skor.abs() <= 3]
print("Temizlenmiş veri:", d_temiz)
#Z-Score Yöntemi (ortalama ile düzeltme)
d_duzeltilmis = d.copy()
for col in sayısal:
    ortalama = d[col].mean()
    std = d[col].std()
    z_skor = (d[col] - ortalama) / std
    ortalama = d_duzeltilmis.loc[z_skor.abs() < 3, col].mean()
    d_duzeltilmis[col] = d_duzeltilmis[col].astype(float)
    d_duzeltilmis.loc[z_skor.abs() > 3, col] = ortalama
print("Düzeltilmiş veri:", d_duzeltilmis)
#Normalizasyon
scaler = MinMaxScaler()
d_normallestirilmis1 = pd.DataFrame(scaler.fit_transform(d[sayısal]), columns=sayısal)
print(d_normallestirilmis1)
scaler = StandardScaler()
d_normallestirilmis2 = pd.DataFrame(scaler.fit_transform(d[sayısal]), columns=sayısal)
print(d_normallestirilmis2)
#Tanımlayıcı İstatistikler
ist = d.describe(include='all')
print(ist)
ist.to_csv("tanımlayıcı_istatistikler.csv", index=True)
count = d.count()
mean = d.mean(numeric_only=True)
median = d.median(numeric_only=True)
std = d.std(numeric_only=True)
min_ = d.min(numeric_only=True)
max_ = d.max(numeric_only=True)
var = d.var(numeric_only=True)
mode = d.mode().iloc[0]
ist = pd.DataFrame({
    "count": count,
    "mean": mean,
    "median": median,
    "std": std,
    "min": min_,
    "max": max_,
    "variance": var,
    "mode": mode
})
print(ist)
ist.to_csv("tanımlayıcı_istatistikler1.csv", index=True)
#Gruplara göre tanımlayıcı istatistikler
print(d.groupby("shopping_preference")["age"].mean())
print(d.groupby("shopping_preference")["age"].sum())
print(d.groupby("shopping_preference")["age"].median())
print(d.groupby("shopping_preference")["age"].count())
print(d.groupby("shopping_preference")["age"].describe())
print(d.groupby("shopping_preference")["avg_online_spend"].describe())
print(d[sayısal].agg(["sum", "mean", 'min', 'max', 'median']))
print(d["age"].agg(["sum", "mean", 'min', 'max', 'median']))

d = pd.read_csv("Consumer_Shopping_Trends_2026.csv", sep=',', header=0, encoding="ISO-8859-1")
sayısal = d.select_dtypes(include=["number"]).columns
for sutun in sayısal:
    if d[sutun].nunique() < 11:
        d[sutun] = d[sutun].astype("category")
sayısal = d.select_dtypes(include=["number"]).columns
#Korelasyon matrisi (Pearson)
d1 = d[sayısal]
corr_matrix = d1.corr()
print(corr_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", square=True)
plt.title("Pearson Korelasyon Matrisi")
plt.show()
#Sıralı (ordinal) değişkenler için Spearman ve Kendall
kategorik = d.select_dtypes(include=["category"]).columns
d2 = d[kategorik].astype(int)
corr_matrix = d2.corr(method="spearman")
print(corr_matrix)
corr_matrix = d2.corr(method="kendall")
print(corr_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", square=True)
plt.title("Kendall Korelasyon Matrisi")
plt.show()
#Ki-kare Bağımsızlık ve Homojenlik testi
veri = pd.crosstab(d["gender"], d["shopping_preference"])
chi2, p_val, dof, expected = chi2_contingency(veri)
print(veri)
print(chi2, p_val, dof, expected)
#Uygunluk testi
gözlenen = [58, 62, 50, 55, 54]
beklenen = [58, 60, 52, 57, 52]
chi2, p = chisquare(f_obs=gözlenen, f_exp=beklenen)
print(chi2, p)
#Tüm kategorik değişken çiftleri için ki-kare testi
kategorik = d.select_dtypes(include="object").columns
d1 = d[kategorik]
for i in range(0, len(kategorik)):
    for j in range(i + 1, len(kategorik)):
        col1 = kategorik[i]
        col2 = kategorik[j]
        veri = pd.crosstab(d1[col1], d1[col2])
        chi, p, dof, expected = chi2_contingency(veri)
        print(veri)
        print(col1, "-", col2, ":", chi, p, dof)
#Mc Nemar testi
tablo = np.array([[40, 50],
                  [10, 35]])
sonuc = mcnemar(tablo, exact=False, correction=True)
print(sonuc)

d = pd.read_csv("Consumer_Shopping_Trends_2026.csv", sep=',', header=0, encoding="ISO-8859-1")
#Tek örneklem t-testi (One-Sample t-test)
t_stat, p_val = stats.ttest_1samp(d["age"], 40)
print("t-istatistiği:", t_stat, "p-değeri:", p_val)
#Bağımsız iki örneklem t-testi
group1, group2 = d["age"].loc[d["shopping_preference"] == "Online"], d["age"].loc[d["shopping_preference"] == "Store"]
t_stat, p_val = stats.ttest_ind(group1, group2)
print("t-istatistiği:", t_stat, "p-değeri:", p_val)
#Eşleştirilmiş t-testi (Paired Samples t-test)
once, sonra = [10, 20, 12, 23, 45, 23, 56, 48, 10, 12, 21, 22], [8, 17, 9, 21, 41, 20, 53, 39, 7, 9, 19, 19]
t_stat, p_val = stats.ttest_rel(once, sonra)
print("t-istatistiği:", t_stat, "p-değeri:", p_val)
#ANOVA (Tek Yönlü Varyans Analizi)
group1 = d["age"].loc[d["shopping_preference"] == "Online"]
group2 = d["age"].loc[d["shopping_preference"] == "Store"]
group3 = d["age"].loc[d["shopping_preference"] == "Hybrid"]
f_stat, p_val = stats.f_oneway(group1, group2, group3)
print("F-istatistiği:", f_stat)
print("p-değeri:", p_val)
#Wilcoxon (One-Sample)
f = d["age"] - 40
stat, p = stats.wilcoxon(f)
print(stat, p)
#Mann-Whitney testi
grup1, grup2 = d["age"].loc[d["shopping_preference"] == "Online"], d["age"].loc[d["shopping_preference"] == "Store"]
stat, p = stats.mannwhitneyu(grup1, grup2, alternative='two-sided')
print(stat, p)
#Eşleştirilmiş Wilcoxon testi
stat, p = stats.wilcoxon(once, sonra)
print(stat, p)
#Kruskal-Wallis testi
stat, p = stats.kruskal(group1, group2, group3)
print(stat, p)
#Görselleştirme
sayısal = d.select_dtypes(include=["number"]).columns
for sutun in sayısal:
    if d[sutun].nunique() < 11:
        d[sutun] = d[sutun].astype("category")
sayısal = d.select_dtypes(include=['number']).columns
#Box-plot grafikleri
for sutun in sayısal:
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(d[sutun].dropna(),
                      patch_artist=True,
                      boxprops=dict(facecolor="lightblue", color="blue"),
                      medianprops=dict(color="red", linewidth=2),
                      whiskerprops=dict(color="blue"),
                      capprops=dict(color="blue"),
                      flierprops=dict(marker='o', markerfacecolor='orange', markersize=8, linestyle='none')
                      )
    plt.title(f"Boxplot - {sutun}", fontsize=14, fontweight="bold")
    plt.ylabel(sutun, fontsize=12)
    plt.show()
#Tüm sayısal değişkenlerin boxplotu (normalleştirilmiş)
scaler = MinMaxScaler()
d_olcek = pd.DataFrame(scaler.fit_transform(d[sayısal]), columns=sayısal)
plt.figure(figsize=(10, 6))
sns.boxplot(data=d_olcek, palette="Set3")
plt.title("Normalleştirilmiş Değişkenlerin Boxplotu", fontsize=12)
plt.xticks(rotation=45, fontsize=12)
plt.ylabel("Normalized Değerler (0-1)", fontsize=14)
plt.show()
#Countplot grafiği
kategorik = ["gender", "city_tier", "shopping_preference"]
for sutun in kategorik:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=sutun, hue=sutun, data=d, palette="Set2")
    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), fontsize=10, color="red")
    plt.show()
#Countplot grafiği (yüzde)
for sutun in kategorik:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=sutun, hue=sutun, data=d, palette="Set2")
    toplamfrekans = len(d[sutun])
    for p in ax.patches:
        yukseklik = p.get_height()
        yüzde = yukseklik / toplamfrekans * 100
        ax.annotate(f"{yüzde:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), fontsize=10, color="red")
    plt.show()
#Scatter plot grafiği
x = range(1, len(d) + 1)
for sutun in sayısal:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, d[sutun], alpha=0.4, color="red")
    ax.set_xlabel("X ekseni", fontsize=10, color="blue")
    ax.set_ylabel(d[sutun].name, fontsize=10, color="blue")
    ax.tick_params(axis='x', colors='blue')
    ax.tick_params(axis='y', colors='blue')
    plt.show()
#Pair plot grafiği
d1 = d[sayısal]
sns.pairplot(d1.iloc[:, 0:2])
plt.show()
sns.pairplot(d1.iloc[:, 2:4])
plt.show()
sns.pairplot(d1.iloc[:, 4:6])
plt.show()

d = pd.read_csv("Consumer_Shopping_Trends_2026.csv", sep=',', header=0, encoding="ISO-8859-1")
print("Eksik Değer Sayısı")
kayıp = d.isnull().sum()
print(kayıp)
x = d.iloc[:, :22]
y = d.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
#Karar Ağacı - Hold-out
print("Karar Ağacı - Hold-out")
clf = DecisionTreeClassifier(random_state=42, max_depth=3)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test)
accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average='macro')
y_test_bin = label_binarize(y_test, classes=np.unique(y))
auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score (macro): {f1:.3f}")
print(f"AUC (OvR): {auc:.3f}")
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Karar Ağacı - Hold-out Confusion Matrix")
plt.show()
plt.figure(figsize=(14, 8))
plot_tree(clf, feature_names=x_train.columns, class_names=[str(c) for c in sorted(y_train.unique())], filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree Classifier")
plt.show()
#Karar Ağacı - 5 Katlı Çapraz Doğrulama
print("Karar Ağacı - Cross-Validation")
clf = DecisionTreeClassifier(random_state=42, max_depth=3)
acc = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
f1 = cross_val_score(clf, x, y, cv=5, scoring='f1_macro')
auc = cross_val_score(clf, x, y, cv=5, scoring='roc_auc_ovr')
print("Doğruluk :", acc.mean())
print("F skor :", f1.mean())
print("AUC :", auc.mean())
y_pred_cv = cross_val_predict(clf, x, y, cv=5)
cm = confusion_matrix(y, y_pred_cv)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap="Blues", values_format='d')
plt.title("Karar Ağacı - Cross-Validation Confusion Matrix")
plt.show()
#K-En Yakın Komşu - Hold-out
print("K-En Yakın Komşu - Hold-out")
clf = KNeighborsClassifier()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test)
accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average='macro')
y_test_bin = label_binarize(y_test, classes=np.unique(y))
auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score (macro): {f1:.3f}")
print(f"AUC (OvR): {auc:.3f}")
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("K-En Yakın Komşu - Hold-out Confusion Matrix")
plt.show()
#K-En Yakın Komşu - 5 Katlı Çapraz Doğrulama
print("K-En Yakın Komşu - Cross-Validation")
clf = KNeighborsClassifier()
acc = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
f1 = cross_val_score(clf, x, y, cv=5, scoring='f1_macro')
auc = cross_val_score(clf, x, y, cv=5, scoring='roc_auc_ovr')
print("Doğruluk :", acc.mean())
print("F skor :", f1.mean())
print("AUC :", auc.mean())
y_pred_cv = cross_val_predict(clf, x, y, cv=5)
cm = confusion_matrix(y, y_pred_cv)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap="Blues", values_format='d')
plt.title("K-En Yakın Komşu - Cross-Validation Confusion Matrix")
plt.show()
#Naive Bayes - Hold-out
print("Naive Bayes - Hold-out")
clf = GaussianNB()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test)
accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average='macro')
y_test_bin = label_binarize(y_test, classes=np.unique(y))
auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score (macro): {f1:.3f}")
print(f"AUC (OvR): {auc:.3f}")
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Naive Bayes - Hold-out Confusion Matrix")
plt.show()
#Naive Bayes - 5 Katlı Çapraz Doğrulama
print("Naive Bayes - Cross-Validation")
clf = GaussianNB()
acc = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
f1 = cross_val_score(clf, x, y, cv=5, scoring='f1_macro')
auc = cross_val_score(clf, x, y, cv=5, scoring='roc_auc_ovr')
print("Doğruluk :", acc.mean())
print("F skor :", f1.mean())
print("AUC :", auc.mean())
y_pred_cv = cross_val_predict(clf, x, y, cv=5)
cm = confusion_matrix(y, y_pred_cv)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap="Blues", values_format='d')
plt.title("Naive Bayes - Cross-Validation Confusion Matrix")
plt.show()
#Lojistik Regresyon - Hold-out
print("Lojistik Regresyon - Hold-out")
clf = LogisticRegression()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test)
accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average='macro')
y_test_bin = label_binarize(y_test, classes=np.unique(y))
auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score (macro): {f1:.3f}")
print(f"AUC (OvR): {auc:.3f}")
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Lojistik Regresyon - Hold-out Confusion Matrix")
plt.show()
#Lojistik Regresyon - 5 Katlı Çapraz Doğrulama
print("Lojistik Regresyon - Cross-Validation")
clf = LogisticRegression()
acc = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
f1 = cross_val_score(clf, x, y, cv=5, scoring='f1_macro')
auc = cross_val_score(clf, x, y, cv=5, scoring='roc_auc_ovr')
print("Doğruluk :", acc.mean())
print("F skor :", f1.mean())
print("AUC :", auc.mean())
y_pred_cv = cross_val_predict(clf, x, y, cv=5)
cm = confusion_matrix(y, y_pred_cv)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap="Blues", values_format='d')
plt.title("Lojistik Regresyon - Cross-Validation Confusion Matrix")
plt.show()
#Tüm modellerin karşılaştırılması (Hold-out)
clf = DecisionTreeClassifier(random_state=42, max_depth=3)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test)
accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average='macro')
y_test_bin = label_binarize(y_test, classes=np.unique(y))
auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
print(f"DecisionTreeClassifier(Accuracy): {accuracy:.3f}")
print(f"DecisionTreeClassifier(F1-score) (macro): {f1:.3f}")
print(f"DecisionTreeClassifier(AUC (OvR)): {auc:.3f}")
clf = KNeighborsClassifier()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test)
accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average='macro')
y_test_bin = label_binarize(y_test, classes=np.unique(y))
auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
print(f"KNeighborsClassifier(Accuracy): {accuracy:.3f}")
print(f"KNeighborsClassifier(F1-score) (macro): {f1:.3f}")
print(f"KNeighborsClassifier(AUC (OvR)): {auc:.3f}")
clf = GaussianNB()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test)
accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average='macro')
y_test_bin = label_binarize(y_test, classes=np.unique(y))
auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
print(f"GaussianNB(Accuracy): {accuracy:.3f}")
print(f"GaussianNB(F1-score) (macro): {f1:.3f}")
print(f"GaussianNB(AUC (OvR)): {auc:.3f}")
clf = LogisticRegression()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test)
accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average='macro')
y_test_bin = label_binarize(y_test, classes=np.unique(y))
auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
print(f"LogisticRegression(Accuracy): {accuracy:.3f}")
print(f"LogisticRegression(F1-score) (macro): {f1:.3f}")
print(f"LogisticRegression(AUC (OvR)): {auc:.3f}")
