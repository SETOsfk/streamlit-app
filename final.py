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
print("Cross-Validation Test Kümesi Seçim Yöntemi")
for i in clfs:
    clf = AdaBoostClassifier(estimator=i, n_estimators=50, random_state=123)
    skor = cross_val_score(clf, x, y, cv=5).mean()
    print(type(i).__name__, ":", skor)
print("********************************************")
print("Hold-out Test Kümesi Seçim Yöntemi")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123, shuffle=True)
for i in clfs:
    clf = AdaBoostClassifier(estimator=i, n_estimators=50, random_state=123)
    clf.fit(x_train, y_train)
    pred, predp, y_test1 = clf.predict(x_test), clf.predict_proba(x_test), label_binarize(y_test, classes=np.unique(y))
    print("********************************************")
    print(type(i).__name__, " (Doğruluk):", accuracy_score(y_test, pred))
    print(type(i).__name__, " (F-Skor):", f1_score(y_test, pred, average="weighted"))
    print(type(i).__name__, " (AUC):", roc_auc_score(y_test1, predp, multi_class='ovr'))
clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=4), n_estimators=50, random_state=123)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues')
plt.title("AdaBoostClassifier Confusion Matrix")
plt.show()
#BaggingClassifier
d = pd.read_csv("Consumer_Shopping_Trends_2026.csv", sep=',', header=0, encoding="ISO-8859-1")
x = d.iloc[:, :22]
y = d.iloc[:, -1]
clfs = [DecisionTreeClassifier(max_depth=4), LogisticRegression(max_iter=100), LinearSVC(), GaussianNB()]
print("Cross-Validation Test Kümesi Seçim Yöntemi")
for i in clfs:
    clf = BaggingClassifier(estimator=i, n_estimators=50, random_state=123)
    skor = cross_val_score(clf, x, y, cv=5).mean()
    print(type(i).__name__, ":", skor)
print("********************************************")
print("Hold-out Test Kümesi Seçim Yöntemi")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123, shuffle=True)
for i in clfs:
    clf = BaggingClassifier(estimator=i, n_estimators=50, random_state=123)
    clf.fit(x_train, y_train)
    pred, predp, y_test1 = clf.predict(x_test), clf.predict_proba(x_test), label_binarize(y_test, classes=np.unique(y))
    print("********************************************")
    print(type(i).__name__, " (Doğruluk):", accuracy_score(y_test, pred))
    print(type(i).__name__, " (F-Skor):", f1_score(y_test, pred, average="weighted"))
    print(type(i).__name__, " (AUC):", roc_auc_score(y_test1, predp, multi_class='ovr'))
clf = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=4), n_estimators=50, random_state=123)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues')
plt.title("BaggingClassifier Confusion Matrix")
plt.show()
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
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"F1-score: {f1_score(y_test, pred, average='weighted'):.3f}")
print(f"AUC (OvR): {roc_auc_score(y_test1, predp, multi_class='ovr'):.3f}")
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap="Blues", values_format='d')
plt.title("Stacking Confusion Matrix")
plt.show()
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
    print("********************************************")
    print(isim, " (Doğruluk):", accuracy_score(y_test, pred))
    print(isim, " (F-Skor):", f1_score(y_test, pred, average="weighted"))
    print(isim, " (AUC):", roc_auc_score(y_test1, predp, multi_class='ovr'))

#Hiyerarşik Kümeleme
df = pd.read_csv("IRIS.csv", sep=',', header=0, encoding="ISO-8859-1")
x, y = np.shape(df)
dataset = df.iloc[:, 0:y-1]
print("*************Tanımlayıcı istatistikler******************")
print(dataset.describe())
print("***********Gruplara Göre Tanımlayıcı istatistikler*****************")
print(df.groupby("species").describe())
#Gruplara Göre sepal_length değişkeninin bar grafiği
ax = sns.barplot(data=df, x="species", y="sepal_length")
for i in ax.containers:
    ax.bar_label(i,)
plt.show()
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
plt.show()
#Nokta (Scatter) grafiği
sns.scatterplot(data=df, x="sepal_width", y="sepal_length", hue="species")
plt.show()
#Noktalı boxplot
sns.boxplot(data=df, x='species', y='sepal_length')
sns.stripplot(data=df, x='species', y='sepal_length')
plt.show()
#İkili karşılaştırma grafikleri
sns.pairplot(df, palette='hls')
plt.show()
#Gruplara göre pairplot
sns.pairplot(df, hue="species", markers=["o", "s", "D"])
plt.show()
#Violin plot
sns.violinplot(data=df, x='species', y='sepal_length')
sns.stripplot(data=df, x='species', y='sepal_length')
plt.show()
#Çizgi grafiği
sns.lineplot(data=df, x="sepal_width", y="sepal_length", hue="species")
plt.show()
#Hiyerarşik kümeleme (Ward)
cr = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
predictions = cr.fit_predict(dataset)
print(predictions)
#Dendrogram
plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(dataset, method='ward'))
plt.show()
plt.figure(figsize=(6, 6))
plt.scatter(dataset['sepal_length'], dataset['sepal_width'], c=predictions, cmap='rainbow')
plt.ylabel("Sepal_Width")
plt.xlabel("Sepal_Length")
plt.show()
plt.figure(figsize=(6, 6))
plt.scatter(dataset['petal_length'], dataset['petal_width'], c=predictions, cmap='rainbow')
plt.ylabel("Petal_Width")
plt.xlabel("Petal_Length")
plt.show()
plt.figure(figsize=(6, 6))
plt.scatter(dataset['petal_length'], dataset['sepal_width'], c=predictions, cmap='rainbow')
plt.ylabel("Sepal_Width")
plt.xlabel("Petal_Length")
plt.show()
plt.figure(figsize=(6, 6))
plt.scatter(dataset['sepal_length'], dataset['petal_width'], c=predictions, cmap='rainbow')
plt.ylabel("Petal_Width")
plt.xlabel("Sepal_Length")
plt.show()
#K-Ortalamalar
dataset = df.iloc[:, [0, 1, 2, 3]].values
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
predictions = kmeans.fit_predict(dataset)
print(predictions)
print(kmeans.cluster_centers_)
plt.figure(figsize=(8, 6))
plt.scatter(dataset[predictions == 0, 0], dataset[predictions == 0, 1], s=50, c='purple', label='Küme 0')
plt.scatter(dataset[predictions == 1, 0], dataset[predictions == 1, 1], s=50, c='orange', label='Küme 1')
plt.scatter(dataset[predictions == 2, 0], dataset[predictions == 2, 1], s=50, c='green', label='Küme 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
plt.legend()
plt.title("K-Ortalamalar")
plt.show()
#K-Medoids
clf = KMedoids(n_clusters=3, random_state=0)
predictions = clf.fit_predict(dataset)
print(clf.cluster_centers_)
plt.figure(figsize=(8, 6))
plt.scatter(dataset[predictions == 0, 0], dataset[predictions == 0, 1], s=50, c='purple', label='Küme 0')
plt.scatter(dataset[predictions == 1, 0], dataset[predictions == 1, 1], s=50, c='orange', label='Küme 1')
plt.scatter(dataset[predictions == 2, 0], dataset[predictions == 2, 1], s=50, c='green', label='Küme 2')
plt.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], s=100, c='red', label='Medoids')
plt.legend()
plt.title("K-Medoids")
plt.show()
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
plt.show()
