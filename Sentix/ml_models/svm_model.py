import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

# %%
df = pd.read_excel('mt_sentimento.xlsx')
# %%
X = df.Caracteristicas
y = df.Target
# %%
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(X)
vocabulario = vectorizer.get_feature_names_out()
X_final = bow.toarray()
# %%
y_indexado = []
for i in y:
    if i == "Negativo":
        y_indexado.append(0)
    else:
        y_indexado.append(1)
# %%
y_final = pd.DataFrame(y_indexado, columns=["Target"])
# %%
clf = svm.SVC()
clf.fit(X_final, y_final.values.ravel())


# %%
def predict(texto):
    vectorizer = CountVectorizer(vocabulary=vocabulario)
    nova_entrada = vectorizer.transform(texto).toarray()

    resposta_vetorizada = clf.predict(nova_entrada)
    if resposta_vetorizada == 1:
        return "Positivo"
    else:
        return "Negativo"


# %%
print(predict([""]))