import streamlit as st
import matplotlib.pyplot as plt

@st.cache_data
def vize():
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
    st.write(d.head())       # İlk 5 satırı göster
    st.write(d.tail())       # Son 5 satırı göster
    st.write(d.dtypes)       # Değişken türleri
    st.write(d.describe())   # İstatistiksel özet
    #Sütunlara Erişim
    st.write(d['age'])
    st.write(d[['age', 'gender']])
    #Filtreleme
    st.write(d[d['age'] > 25])
    #Yeni Sütun Ekleme
    d['yeni_yas'] = d['age'] + 5
    st.write(d['yeni_yas'])
    d = d.drop(columns=['yeni_yas'])
    #Eksik değerlerin ve yüzdelerinin tespiti
    st.write("Eksik Değer Sayısı")
    kayıp = d.isnull().sum()
    st.write(kayıp)
    st.write("Eksik Değer Yüzdesi")
    kayıpyuzde = (kayıp / len(d))
    st.write(kayıpyuzde)
    st.write(d.isnull().mean().round(3))
    st.write("Tüm veri setindeki eksik değer sayısı:", d.isnull().sum().sum())
    #Tekrar eden verilerin silinmesi
    d = d.drop_duplicates()
    st.write("Tekrarlar silindikten sonra boyut:", d.shape)
    #Eksik değerleri doldurma
    d['age'] = d['age'].fillna(d['age'].mean())
    d['monthly_income'] = d['monthly_income'].fillna(d['monthly_income'].median())
    d['daily_internet_hours'] = d['daily_internet_hours'].fillna(d['daily_internet_hours'].mean())
    #Kategorik verileri en sık görülen değer ile doldurma
    d['gender'] = d['gender'].fillna(d['gender'].mode()[0])
    st.write(d.isna().sum())
    #Interpolate ile doldurma
    tarih = pd.date_range("2025-01-01", periods=10)
    s = pd.Series([10, np.nan, 38, np.nan, 78, 45, 56, 34, np.nan, 65], index=tarih)
    st.write("Orijinal seri:", s)
    s_doldurulmus = s.interpolate()
    st.write("Lineer interpolate:", s_doldurulmus)
    s_doldurulmus = s.interpolate(method="time")
    st.write("Time interpolate:", s_doldurulmus)
    s_doldurulmus = s.interpolate(method="polynomial", order=2)
    st.write("Polinom interpolate:", s_doldurulmus)
    #Aykırı Değerleri Tespit Etme (IQR)
    q1 = d['age'].quantile(0.25)
    q3 = d['age'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    st.write(d[d['age'] < lower].index)
    st.write(d[d['age'] > upper].index)
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
        st.write(col, "aykırı değer sayısı:", len(aykırı_indeks[col]))
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
    st.write(d_temiz1.isnull().sum())
    st.write("Temizlenmiş veri boyutu:", d_temiz1.shape)
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
    st.write(d_temiz)

    d = pd.read_csv("Consumer_Shopping_Trends_2026.csv", sep=',', header=0, encoding="ISO-8859-1")
    #Aykırı Değerleri Tespit Etme-Z-Score Yöntemi (silme)
    d_temiz = d.copy()
    sayısal = d.select_dtypes(include='number').columns
    for col in sayısal:
        ortalama = d[col].mean()
        std = d[col].std()
        z_skor = (d[col] - ortalama) / std
        d_temiz = d_temiz[z_skor.abs() <= 3]
    st.write("Temizlenmiş veri:", d_temiz)
    #Z-Score Yöntemi (ortalama ile düzeltme)
    d_duzeltilmis = d.copy()
    for col in sayısal:
        ortalama = d[col].mean()
        std = d[col].std()
        z_skor = (d[col] - ortalama) / std
        ortalama = d_duzeltilmis.loc[z_skor.abs() < 3, col].mean()
        d_duzeltilmis[col] = d_duzeltilmis[col].astype(float)
        d_duzeltilmis.loc[z_skor.abs() > 3, col] = ortalama
    st.write("Düzeltilmiş veri:", d_duzeltilmis)
    #Normalizasyon
    scaler = MinMaxScaler()
    d_normallestirilmis1 = pd.DataFrame(scaler.fit_transform(d[sayısal]), columns=sayısal)
    st.write(d_normallestirilmis1)
    scaler = StandardScaler()
    d_normallestirilmis2 = pd.DataFrame(scaler.fit_transform(d[sayısal]), columns=sayısal)
    st.write(d_normallestirilmis2)
    #Tanımlayıcı İstatistikler
    ist = d.describe(include='all')
    st.write(ist)
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
    st.write(ist)
    ist.to_csv("tanımlayıcı_istatistikler1.csv", index=True)
    #Gruplara göre tanımlayıcı istatistikler
    st.write(d.groupby("shopping_preference")["age"].mean())
    st.write(d.groupby("shopping_preference")["age"].sum())
    st.write(d.groupby("shopping_preference")["age"].median())
    st.write(d.groupby("shopping_preference")["age"].count())
    st.write(d.groupby("shopping_preference")["age"].describe())
    st.write(d.groupby("shopping_preference")["avg_online_spend"].describe())
    st.write(d[sayısal].agg(["sum", "mean", 'min', 'max', 'median']))
    st.write(d["age"].agg(["sum", "mean", 'min', 'max', 'median']))

    d = pd.read_csv("Consumer_Shopping_Trends_2026.csv", sep=',', header=0, encoding="ISO-8859-1")
    sayısal = d.select_dtypes(include=["number"]).columns
    for sutun in sayısal:
        if d[sutun].nunique() < 11:
            d[sutun] = d[sutun].astype("category")
    sayısal = d.select_dtypes(include=["number"]).columns
    #Korelasyon matrisi (Pearson)
    d1 = d[sayısal]
    corr_matrix = d1.corr()
    st.write(corr_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", square=True)
    plt.title("Pearson Korelasyon Matrisi")
    st.pyplot(plt.gcf())
    plt.close()
    #Sıralı (ordinal) değişkenler için Spearman ve Kendall
    kategorik = d.select_dtypes(include=["category"]).columns
    d2 = d[kategorik].astype(int)
    corr_matrix = d2.corr(method="spearman")
    st.write(corr_matrix)
    corr_matrix = d2.corr(method="kendall")
    st.write(corr_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", square=True)
    plt.title("Kendall Korelasyon Matrisi")
    st.pyplot(plt.gcf())
    plt.close()
    #Ki-kare Bağımsızlık ve Homojenlik testi
    veri = pd.crosstab(d["gender"], d["shopping_preference"])
    chi2, p_val, dof, expected = chi2_contingency(veri)
    st.write(veri)
    st.write(chi2, p_val, dof, expected)
    #Uygunluk testi
    gözlenen = [58, 62, 50, 55, 54]
    beklenen = [58, 60, 52, 57, 52]
    chi2, p = chisquare(f_obs=gözlenen, f_exp=beklenen)
    st.write(chi2, p)
    #Tüm kategorik değişken çiftleri için ki-kare testi
    kategorik = d.select_dtypes(include="object").columns
    d1 = d[kategorik]
    for i in range(0, len(kategorik)):
        for j in range(i + 1, len(kategorik)):
            col1 = kategorik[i]
            col2 = kategorik[j]
            veri = pd.crosstab(d1[col1], d1[col2])
            chi, p, dof, expected = chi2_contingency(veri)
            st.write(veri)
            st.write(col1, "-", col2, ":", chi, p, dof)
    #Mc Nemar testi
    tablo = np.array([[40, 50],
                      [10, 35]])
    sonuc = mcnemar(tablo, exact=False, correction=True)
    st.write(sonuc)

    d = pd.read_csv("Consumer_Shopping_Trends_2026.csv", sep=',', header=0, encoding="ISO-8859-1")
    #Tek örneklem t-testi (One-Sample t-test)
    t_stat, p_val = stats.ttest_1samp(d["age"], 40)
    st.write("t-istatistiği:", t_stat, "p-değeri:", p_val)
    #Bağımsız iki örneklem t-testi
    group1, group2 = d["age"].loc[d["shopping_preference"] == "Online"], d["age"].loc[d["shopping_preference"] == "Store"]
    t_stat, p_val = stats.ttest_ind(group1, group2)
    st.write("t-istatistiği:", t_stat, "p-değeri:", p_val)
    #Eşleştirilmiş t-testi (Paired Samples t-test)
    once, sonra = [10, 20, 12, 23, 45, 23, 56, 48, 10, 12, 21, 22], [8, 17, 9, 21, 41, 20, 53, 39, 7, 9, 19, 19]
    t_stat, p_val = stats.ttest_rel(once, sonra)
    st.write("t-istatistiği:", t_stat, "p-değeri:", p_val)
    #ANOVA (Tek Yönlü Varyans Analizi)
    group1 = d["age"].loc[d["shopping_preference"] == "Online"]
    group2 = d["age"].loc[d["shopping_preference"] == "Store"]
    group3 = d["age"].loc[d["shopping_preference"] == "Hybrid"]
    f_stat, p_val = stats.f_oneway(group1, group2, group3)
    st.write("F-istatistiği:", f_stat)
    st.write("p-değeri:", p_val)
    #Wilcoxon (One-Sample)
    f = d["age"] - 40
    stat, p = stats.wilcoxon(f)
    st.write(stat, p)
    #Mann-Whitney testi
    grup1, grup2 = d["age"].loc[d["shopping_preference"] == "Online"], d["age"].loc[d["shopping_preference"] == "Store"]
    stat, p = stats.mannwhitneyu(grup1, grup2, alternative='two-sided')
    st.write(stat, p)
    #Eşleştirilmiş Wilcoxon testi
    stat, p = stats.wilcoxon(once, sonra)
    st.write(stat, p)
    #Kruskal-Wallis testi
    stat, p = stats.kruskal(group1, group2, group3)
    st.write(stat, p)
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
        st.pyplot(plt.gcf())
        plt.close()
    #Tüm sayısal değişkenlerin boxplotu (normalleştirilmiş)
    scaler = MinMaxScaler()
    d_olcek = pd.DataFrame(scaler.fit_transform(d[sayısal]), columns=sayısal)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=d_olcek, palette="Set3")
    plt.title("Normalleştirilmiş Değişkenlerin Boxplotu", fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.ylabel("Normalized Değerler (0-1)", fontsize=14)
    st.pyplot(plt.gcf())
    plt.close()
    #Countplot grafiği
    kategorik = ["gender", "city_tier", "shopping_preference"]
    for sutun in kategorik:
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(x=sutun, hue=sutun, data=d, palette="Set2")
        for p in ax.patches:
            ax.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), fontsize=10, color="red")
        st.pyplot(plt.gcf())
        plt.close()
    #Countplot grafiği (yüzde)
    for sutun in kategorik:
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(x=sutun, hue=sutun, data=d, palette="Set2")
        toplamfrekans = len(d[sutun])
        for p in ax.patches:
            yukseklik = p.get_height()
            yüzde = yukseklik / toplamfrekans * 100
            ax.annotate(f"{yüzde:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), fontsize=10, color="red")
        st.pyplot(plt.gcf())
        plt.close()
    #Scatter plot grafiği
    x = range(1, len(d) + 1)
    for sutun in sayısal:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, d[sutun], alpha=0.4, color="red")
        ax.set_xlabel("X ekseni", fontsize=10, color="blue")
        ax.set_ylabel(d[sutun].name, fontsize=10, color="blue")
        ax.tick_params(axis='x', colors='blue')
        ax.tick_params(axis='y', colors='blue')
        st.pyplot(plt.gcf())
        plt.close()
    #Pair plot grafiği
    d1 = d[sayısal]
    sns.pairplot(d1.iloc[:, 0:2])
    st.pyplot(plt.gcf())
    plt.close()
    sns.pairplot(d1.iloc[:, 2:4])
    st.pyplot(plt.gcf())
    plt.close()
    sns.pairplot(d1.iloc[:, 4:6])
    st.pyplot(plt.gcf())
    plt.close()

    d = pd.read_csv("Consumer_Shopping_Trends_2026.csv", sep=',', header=0, encoding="ISO-8859-1")
    st.write("Eksik Değer Sayısı")
    kayıp = d.isnull().sum()
    st.write(kayıp)
    x = d.iloc[:, :22]
    y = d.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    #Karar Ağacı - Hold-out
    st.write("Karar Ağacı - Hold-out")
    clf = DecisionTreeClassifier(random_state=42, max_depth=3)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
    st.write(f"Accuracy: {accuracy:.3f}")
    st.write(f"F1-score (macro): {f1:.3f}")
    st.write(f"AUC (OvR): {auc:.3f}")
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Karar Ağacı - Hold-out Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.close()
    plt.figure(figsize=(14, 8))
    plot_tree(clf, feature_names=x_train.columns, class_names=[str(c) for c in sorted(y_train.unique())], filled=True, rounded=True, fontsize=8)
    plt.title("Decision Tree Classifier")
    st.pyplot(plt.gcf())
    plt.close()
    #Karar Ağacı - 5 Katlı Çapraz Doğrulama
    st.write("Karar Ağacı - Cross-Validation")
    clf = DecisionTreeClassifier(random_state=42, max_depth=3)
    acc = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
    f1 = cross_val_score(clf, x, y, cv=5, scoring='f1_macro')
    auc = cross_val_score(clf, x, y, cv=5, scoring='roc_auc_ovr')
    st.write("Doğruluk :", acc.mean())
    st.write("F skor :", f1.mean())
    st.write("AUC :", auc.mean())
    y_pred_cv = cross_val_predict(clf, x, y, cv=5)
    cm = confusion_matrix(y, y_pred_cv)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Karar Ağacı - Cross-Validation Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.close()
    #K-En Yakın Komşu - Hold-out
    st.write("K-En Yakın Komşu - Hold-out")
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
    st.write(f"Accuracy: {accuracy:.3f}")
    st.write(f"F1-score (macro): {f1:.3f}")
    st.write(f"AUC (OvR): {auc:.3f}")
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("K-En Yakın Komşu - Hold-out Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.close()
    #K-En Yakın Komşu - 5 Katlı Çapraz Doğrulama
    st.write("K-En Yakın Komşu - Cross-Validation")
    clf = KNeighborsClassifier()
    acc = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
    f1 = cross_val_score(clf, x, y, cv=5, scoring='f1_macro')
    auc = cross_val_score(clf, x, y, cv=5, scoring='roc_auc_ovr')
    st.write("Doğruluk :", acc.mean())
    st.write("F skor :", f1.mean())
    st.write("AUC :", auc.mean())
    y_pred_cv = cross_val_predict(clf, x, y, cv=5)
    cm = confusion_matrix(y, y_pred_cv)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap="Blues", values_format='d')
    plt.title("K-En Yakın Komşu - Cross-Validation Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.close()
    #Naive Bayes - Hold-out
    st.write("Naive Bayes - Hold-out")
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
    st.write(f"Accuracy: {accuracy:.3f}")
    st.write(f"F1-score (macro): {f1:.3f}")
    st.write(f"AUC (OvR): {auc:.3f}")
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Naive Bayes - Hold-out Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.close()
    #Naive Bayes - 5 Katlı Çapraz Doğrulama
    st.write("Naive Bayes - Cross-Validation")
    clf = GaussianNB()
    acc = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
    f1 = cross_val_score(clf, x, y, cv=5, scoring='f1_macro')
    auc = cross_val_score(clf, x, y, cv=5, scoring='roc_auc_ovr')
    st.write("Doğruluk :", acc.mean())
    st.write("F skor :", f1.mean())
    st.write("AUC :", auc.mean())
    y_pred_cv = cross_val_predict(clf, x, y, cv=5)
    cm = confusion_matrix(y, y_pred_cv)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Naive Bayes - Cross-Validation Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.close()
    #Lojistik Regresyon - Hold-out
    st.write("Lojistik Regresyon - Hold-out")
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
    st.write(f"Accuracy: {accuracy:.3f}")
    st.write(f"F1-score (macro): {f1:.3f}")
    st.write(f"AUC (OvR): {auc:.3f}")
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Lojistik Regresyon - Hold-out Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.close()
    #Lojistik Regresyon - 5 Katlı Çapraz Doğrulama
    st.write("Lojistik Regresyon - Cross-Validation")
    clf = LogisticRegression()
    acc = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
    f1 = cross_val_score(clf, x, y, cv=5, scoring='f1_macro')
    auc = cross_val_score(clf, x, y, cv=5, scoring='roc_auc_ovr')
    st.write("Doğruluk :", acc.mean())
    st.write("F skor :", f1.mean())
    st.write("AUC :", auc.mean())
    y_pred_cv = cross_val_predict(clf, x, y, cv=5)
    cm = confusion_matrix(y, y_pred_cv)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Lojistik Regresyon - Cross-Validation Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.close()
    #Tüm modellerin karşılaştırılması (Hold-out)
    clf = DecisionTreeClassifier(random_state=42, max_depth=3)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
    st.write(f"DecisionTreeClassifier(Accuracy): {accuracy:.3f}")
    st.write(f"DecisionTreeClassifier(F1-score) (macro): {f1:.3f}")
    st.write(f"DecisionTreeClassifier(AUC (OvR)): {auc:.3f}")
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
    st.write(f"KNeighborsClassifier(Accuracy): {accuracy:.3f}")
    st.write(f"KNeighborsClassifier(F1-score) (macro): {f1:.3f}")
    st.write(f"KNeighborsClassifier(AUC (OvR)): {auc:.3f}")
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
    st.write(f"GaussianNB(Accuracy): {accuracy:.3f}")
    st.write(f"GaussianNB(F1-score) (macro): {f1:.3f}")
    st.write(f"GaussianNB(AUC (OvR)): {auc:.3f}")
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
    st.write(f"LogisticRegression(Accuracy): {accuracy:.3f}")
    st.write(f"LogisticRegression(F1-score) (macro): {f1:.3f}")
    st.write(f"LogisticRegression(AUC (OvR)): {auc:.3f}")

@st.cache_data
def final():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, StackingClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.preprocessing import label_binarize, StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    import scipy.cluster.hierarchy as sch
    from sklearn_extra.cluster import KMedoids
    from fcmeans import FCM

    #AdaBoostClassifier
    d = pd.read_csv("Consumer_Shopping_Trends_2026.csv", sep=',', header=0, encoding="ISO-8859-1")
    x = d.iloc[:, :22]
    y = d.iloc[:, -1]
    clfs = [DecisionTreeClassifier(max_depth=4), LogisticRegression(max_iter=100), LinearSVC(), GaussianNB()]
    st.write("Cross-Validation Test Kümesi Seçim Yöntemi")
    for i in clfs:
        clf = AdaBoostClassifier(estimator=i, n_estimators=50, random_state=123)
        skor = cross_val_score(clf, x, y, cv=5).mean()
        st.write(type(i).__name__, ":", skor)
    st.write("********************************************")
    st.write("Hold-out Test Kümesi Seçim Yöntemi")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123, shuffle=True)
    for i in clfs:
        clf = AdaBoostClassifier(estimator=i, n_estimators=50, random_state=123)
        clf.fit(x_train, y_train)
        pred, predp, y_test1 = clf.predict(x_test), clf.predict_proba(x_test), label_binarize(y_test, classes=np.unique(y))
        st.write("********************************************")
        st.write(type(i).__name__, " (Doğruluk):", accuracy_score(y_test, pred))
        st.write(type(i).__name__, " (F-Skor):", f1_score(y_test, pred, average="weighted"))
        st.write(type(i).__name__, " (AUC):", roc_auc_score(y_test1, predp, multi_class='ovr'))
    clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=4), n_estimators=50, random_state=123)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap='Blues')
    plt.title("AdaBoostClassifier Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.close()
    #BaggingClassifier
    d = pd.read_csv("Consumer_Shopping_Trends_2026.csv", sep=',', header=0, encoding="ISO-8859-1")
    x = d.iloc[:, :22]
    y = d.iloc[:, -1]
    clfs = [DecisionTreeClassifier(max_depth=4), LogisticRegression(max_iter=100), LinearSVC(), GaussianNB()]
    st.write("Cross-Validation Test Kümesi Seçim Yöntemi")
    for i in clfs:
        clf = BaggingClassifier(estimator=i, n_estimators=50, random_state=123)
        skor = cross_val_score(clf, x, y, cv=5).mean()
        st.write(type(i).__name__, ":", skor)
    st.write("********************************************")
    st.write("Hold-out Test Kümesi Seçim Yöntemi")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123, shuffle=True)
    for i in clfs:
        clf = BaggingClassifier(estimator=i, n_estimators=50, random_state=123)
        clf.fit(x_train, y_train)
        pred, predp, y_test1 = clf.predict(x_test), clf.predict_proba(x_test), label_binarize(y_test, classes=np.unique(y))
        st.write("********************************************")
        st.write(type(i).__name__, " (Doğruluk):", accuracy_score(y_test, pred))
        st.write(type(i).__name__, " (F-Skor):", f1_score(y_test, pred, average="weighted"))
        st.write(type(i).__name__, " (AUC):", roc_auc_score(y_test1, predp, multi_class='ovr'))
    clf = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=4), n_estimators=50, random_state=123)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap='Blues')
    plt.title("BaggingClassifier Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.close()
    #StackingClassifier
    d = pd.read_csv("Consumer_Shopping_Trends_2026.csv", sep=',', header=0, encoding="ISO-8859-1")
    x = d.iloc[:, :22]
    y = d.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
    temel = [('DT', DecisionTreeClassifier(max_depth=4)), ('LSVC', LinearSVC()), ('KNN', KNeighborsClassifier())]
    clf = StackingClassifier(estimators=temel, final_estimator=LogisticRegression(), cv=5, passthrough=False)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    predp = clf.predict_proba(x_test)
    y_test1 = label_binarize(y_test, classes=np.unique(y))
    st.write(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    st.write(f"F1-score: {f1_score(y_test, pred, average='weighted'):.3f}")
    st.write(f"AUC (OvR): {roc_auc_score(y_test1, predp, multi_class='ovr'):.3f}")
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Stacking Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.close()
    #Random Forest, Extra Trees, Gradient Boosting, Hist Gradient Boosting
    d = pd.read_csv("Consumer_Shopping_Trends_2026.csv", sep=',', header=0, encoding="ISO-8859-1")
    x, y = d.iloc[:, :22], d.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
    clfs = {"Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42)}
    for isim, clf in clfs.items():
        clf.fit(x_train, y_train)
        pred, predp, y_test1 = clf.predict(x_test), clf.predict_proba(x_test), label_binarize(y_test, classes=np.unique(y))
        st.write("********************************************")
        st.write(isim, " (Doğruluk):", accuracy_score(y_test, pred))
        st.write(isim, " (F-Skor):", f1_score(y_test, pred, average="weighted"))
        st.write(isim, " (AUC):", roc_auc_score(y_test1, predp, multi_class='ovr'))

    #Hiyerarşik Kümeleme
    df = pd.read_csv("IRIS.csv", sep=',', header=0, encoding="ISO-8859-1")
    x, y = np.shape(df)
    dataset = df.iloc[:, 0:y-1]
    st.write("*************Tanımlayıcı istatistikler******************")
    st.write(dataset.describe())
    st.write("***********Gruplara Göre Tanımlayıcı istatistikler*****************")
    st.write(df.groupby("species").describe())
    #Gruplara Göre sepal_length değişkeninin bar grafiği
    ax = sns.barplot(data=df, x="species", y="sepal_length")
    for i in ax.containers:
        ax.bar_label(i,)
    st.pyplot(plt.gcf())
    plt.close()
    #Subplot Çizilmesi
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    sns.set(style="darkgrid")
    ax1, ax2 = sns.barplot(ax=axes[0, 0], data=df, x='species', y='sepal_length'), sns.barplot(ax=axes[0, 1], data=df, x='species', y='sepal_width')
    ax3, ax4 = sns.barplot(ax=axes[1, 0], data=df, x='species', y='petal_length'), sns.barplot(ax=axes[1, 1], data=df, x='species', y='petal_width')
    for i in ax1.containers:
        ax1.bar_label(i,)
    for i in ax2.containers:
        ax2.bar_label(i,)
    for i in ax3.containers:
        ax3.bar_label(i,)
    for i in ax4.containers:
        ax4.bar_label(i,)
    st.pyplot(plt.gcf())
    plt.close()
    #Nokta (Scatter) grafiği
    sns.scatterplot(data=df, x="sepal_width", y="sepal_length", hue="species")
    st.pyplot(plt.gcf())
    plt.close()
    #Noktalı boxplot
    sns.boxplot(data=df, x='species', y='sepal_length')
    sns.stripplot(data=df, x='species', y='sepal_length')
    st.pyplot(plt.gcf())
    plt.close()
    #İkili karşılaştırma grafikleri
    sns.pairplot(df, palette='hls')
    st.pyplot(plt.gcf())
    plt.close()
    #Gruplara göre pairplot
    sns.pairplot(df, hue="species", markers=["o", "s", "D"])
    st.pyplot(plt.gcf())
    plt.close()
    #Violin plot
    sns.violinplot(data=df, x='species', y='sepal_length')
    sns.stripplot(data=df, x='species', y='sepal_length')
    st.pyplot(plt.gcf())
    plt.close()
    #Çizgi grafiği
    sns.lineplot(data=df, x="sepal_width", y="sepal_length", hue="species")
    st.pyplot(plt.gcf())
    plt.close()
    #Hiyerarşik kümeleme (Ward)
    cr = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
    predictions = cr.fit_predict(dataset)
    st.write(predictions)
    #Dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram = sch.dendrogram(sch.linkage(dataset, method='ward'))
    st.pyplot(plt.gcf())
    plt.close()
    plt.figure(figsize=(6, 6))
    plt.scatter(dataset['sepal_length'], dataset['sepal_width'], c=predictions, cmap='rainbow')
    plt.ylabel("Sepal_Width")
    plt.xlabel("Sepal_Length")
    st.pyplot(plt.gcf())
    plt.close()
    plt.figure(figsize=(6, 6))
    plt.scatter(dataset['petal_length'], dataset['petal_width'], c=predictions, cmap='rainbow')
    plt.ylabel("Petal_Width")
    plt.xlabel("Petal_Length")
    st.pyplot(plt.gcf())
    plt.close()
    plt.figure(figsize=(6, 6))
    plt.scatter(dataset['petal_length'], dataset['sepal_width'], c=predictions, cmap='rainbow')
    plt.ylabel("Sepal_Width")
    plt.xlabel("Petal_Length")
    st.pyplot(plt.gcf())
    plt.close()
    plt.figure(figsize=(6, 6))
    plt.scatter(dataset['sepal_length'], dataset['petal_width'], c=predictions, cmap='rainbow')
    plt.ylabel("Petal_Width")
    plt.xlabel("Sepal_Length")
    st.pyplot(plt.gcf())
    plt.close()
    #K-Ortalamalar
    dataset = df.iloc[:, [0, 1, 2, 3]].values
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
    predictions = kmeans.fit_predict(dataset)
    st.write(predictions)
    st.write(kmeans.cluster_centers_)
    plt.figure(figsize=(8, 6))
    plt.scatter(dataset[predictions == 0, 0], dataset[predictions == 0, 1], s=50, c='purple', label='Küme 0')
    plt.scatter(dataset[predictions == 1, 0], dataset[predictions == 1, 1], s=50, c='orange', label='Küme 1')
    plt.scatter(dataset[predictions == 2, 0], dataset[predictions == 2, 1], s=50, c='green', label='Küme 2')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
    plt.legend()
    plt.title("K-Ortalamalar")
    st.pyplot(plt.gcf())
    plt.close()
    #K-Medoids
    clf = KMedoids(n_clusters=3, random_state=0)
    predictions = clf.fit_predict(dataset)
    st.write(clf.cluster_centers_)
    plt.figure(figsize=(8, 6))
    plt.scatter(dataset[predictions == 0, 0], dataset[predictions == 0, 1], s=50, c='purple', label='Küme 0')
    plt.scatter(dataset[predictions == 1, 0], dataset[predictions == 1, 1], s=50, c='orange', label='Küme 1')
    plt.scatter(dataset[predictions == 2, 0], dataset[predictions == 2, 1], s=50, c='green', label='Küme 2')
    plt.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], s=100, c='red', label='Medoids')
    plt.legend()
    plt.title("K-Medoids")
    st.pyplot(plt.gcf())
    plt.close()
    #Bulanık C-Ortalamalar
    scaler = StandardScaler()
    Xnormalize = scaler.fit_transform(dataset)
    pca = PCA(n_components=2)
    X_tba = pca.fit_transform(Xnormalize)
    clf = FCM(n_clusters=3, m=2)
    clf.fit(Xnormalize)
    kumemerkezi = pca.transform(clf.centers)
    predictions = clf.predict(Xnormalize)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tba[:, 0], X_tba[:, 1], c=predictions)
    plt.scatter(kumemerkezi[:, 0], kumemerkezi[:, 1], s=100, c='red', label='Centroids')
    plt.title("Bulanık C-Ortalamalar")
    plt.xlabel("Temel Bileşen 1")
    plt.ylabel("Temel Bileşen 2")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

st.set_page_config(page_title="Veri Madenciliği Ödevi", layout="wide")
st.title("Veri Madenciliği Ödevi")
st.write("Veri seti: Consumer_Shopping_Trends_2026.csv (kümeleme: IRIS.csv)")

odev = st.sidebar.radio("Ödev", ["Vize Ödevi", "Final Ödevi"])
st.header(odev)
dosya = "vize.py" if odev == "Vize Ödevi" else "final.py"
with st.expander("Python kodunu göster"):
    st.code(open(dosya, encoding="utf-8").read(), language="python")
if odev == "Vize Ödevi":
    with st.spinner("Kod çalışıyor..."):
        vize()
else:
    st.warning("Final ödevi ilk açılışta birkaç dakika sürebilir.")
    with st.spinner("Kod çalışıyor..."):
        final()
