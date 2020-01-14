# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# %%
data = pd.read_csv('nfr.csv')
data

# %%
interest = ['US', 'SE', 'PE', 'O']
data['class'].unique()
data['req_length'] = data['RequirementText'].str.len()

# %%
data = data.loc[data['class'].isin(interest)]
data

# %% [markdown]
# ## Feature Engineering
#
# El procedimiento siguiente incluye el preprocesamiento del texto para el entrenamiento de los modelo de _machine learning_. Los pasos son lo siguientes:
# 1. **Limpieza y preparación de texto:** Se retiran caracteres especiales, signos de puntuación, stop words y pronombres posesivos. Se hace una conversión a minúsculas y lemanización.
# 2. **Label coding:** Se crea un diccionario para mapear cada categoría en un código específico.
# 3. **Separación de _Train_ y _Test_ sets:** Preparación de conjuntos para uso de los modelos de ML
# 4. **Representación de Texto:** Uso de puntuaciones TF-IDF para la representación de texto.

# %% [markdown]
# ### Limpieza y preparación de texto
#
# #### 1.1 Limpiado de caracteres especiales
#
# Se retiran los caracteres que se identifican de lasiguiente forma:
#
# * ``\r``
# * ``\n``
# * ``\`` before possessive pronouns (`government's = government\'s`)
# * ``\`` before possessive pronouns 2 (`Yukos'` = `Yukos\'`)
# * ``"`` when quoting text

# %%
data['Content_Parsed_1'] = data['RequirementText'].str.replace(r'\\r',' ', regex=True)
data['Content_Parsed_1'] = data['Content_Parsed_1'].str.replace(r'\\n',' ', regex=True)
data['Content_Parsed_1'] = data['Content_Parsed_1'].str.replace(r'\\t',' ', regex=True)
data['Content_Parsed_1'] = data['Content_Parsed_1'].str.replace(r'    ',' ', regex=True)
data['Content_Parsed_1'] = data['Content_Parsed_1'].str.replace(r'""','', regex=True)

# %% [markdown]
# #### 1.2 Upcase/downcase
#
# Conversión de todo el texto a minúsculas

# %%
data['Content_Parsed_2'] = data['Content_Parsed_1'].str.lower()

# %% [markdown]
# #### 1.3 Signos de puntuación
#
# Remoción de signos de puntuación, pues no agregan significado a la clasificación

# %%
punctuation_signs = list("?:!.,;")
data['Content_Parsed_3'] = data['Content_Parsed_2']

for punct_sign in punctuation_signs:
    data['Content_Parsed_3'] = data['Content_Parsed_3'].str.replace(punct_sign, '')

# %% [markdown]
# #### 1.4 Pronombres posesivos
#
# Se retiran todos los casos de poseción, pues no agregan significdo adicional, casos como `Luke's` se tornan `Luke`

# %%
data['Content_Parsed_4'] = data['Content_Parsed_3'].str.replace("'s'","", regex=True)

# %% [markdown]
# #### 1.5 Stemming y Lematization
#
# En este caso, se recurre únicamente a lemmatization, pues es normal que el proceso de stemming retorne palabras inexistentes. En el caso del análisis morfológico que utiliza la segunda opción, se obtienen lemmas, u orígenes morfológiocos de palabras.

# %%
# Downloading punkt and wordnet from NLTK
nltk.download('punkt')
print("------------------------------------------------------------")
nltk.download('wordnet')

# %%
text = data.iloc[50]['Content_Parsed_4']
text

# %%
wordnet_lemmatizer = WordNetLemmatizer()
nrows = len(data)
lemmatized_text_list = []

for row in range(0, nrows):
    
    # Create an empty list containing lemmatized words
    lemmatized_list = []
    
    # Save the text and its words into an object
    text = data.iloc[row]['Content_Parsed_4']
    text_words = text.split(" ")

    # Iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
    # Join the list
    lemmatized_text = " ".join(lemmatized_list)
    
    # Append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)

# %%
data['Content_Parsed_5'] = lemmatized_text_list

# %% [markdown]
# #### 1.6 Stop words
#
# Remoción de palabras que no agregan significado extra

# %%
# Downloading the stop words list
nltk.download('stopwords')

# %%
# Loading the stop words in english
stop_words = list(stopwords.words('english'))
stop_words[0:10]

# %%
data['Content_Parsed_6'] = data['Content_Parsed_5']

for stop_word in stop_words:

    regex_stopword = r"\b" + stop_word + r"\b"
    data['Content_Parsed_6'] = data['Content_Parsed_6'].str.replace(regex_stopword, '')

# %% [markdown]
# Es importante recordar, que aunque este preceso resulta con múltiples espacios entre oraciones, estos no significan semánticamente información extra. sin embargo también serán procesados en el proceso de `tokenize`.
#
# ### Resultado
#
# Los siguientes son ejemplos de los cambios que sufre un requisito a lo largo de este procedimeinto:

# %% [markdown]
# #### Requisito Original

# %%
data.loc[5]['RequirementText']

# %% [markdown]
# #### Remoción de caracteres especiales

# %%
data.loc[5]['Content_Parsed_1']

# %% [markdown]
# #### Minúsculas

# %%
data.loc[5]['Content_Parsed_2']

# %% [markdown]
# #### Signos de puntuación

# %%
data.loc[5]['Content_Parsed_3']

# %% [markdown]
# #### Pronombres posesivos

# %%
data.loc[5]['Content_Parsed_4']

# %% [markdown]
# #### Steeming y Lemmatization

# %%
data.loc[5]['Content_Parsed_5']

# %% [markdown]
# #### Stop words

# %%
data.loc[5]['Content_Parsed_6']

# %% [markdown]
# ### Limpiado de pasos intermedios

# %%
data

# %%
list_columns = ["ProjectID", "RequirementText", "class", "req_length", 'Content_Parsed_6']
data = data[list_columns]

data = data.rename(columns={'Content_Parsed_6': 'Content_Parsed'})

# %%
data

# %% [markdown]
# ### Label coding
#
# Cración de un diccionario para discretizar las categorías:
#
# ```python
# category_codes = {
#     'O': 1,
#     'PE': 2,
#     'SE': 3,
#     'US': 4,
# }
# ```

# %%
category_codes = {
    'O': 1,
    'PE': 2,
    'SE': 3,
    'US': 4,
}

# %%
# Category mapping
data['Category_Code'] = data['class']
data = data.replace({'Category_Code':category_codes})

# %%
data

# %%
X_train = data['Content_Parsed']
y_train = data['Category_Code']

# %% [markdown]
# ### Representación de Texto
#
# Se identificán mútiples técnicas de representación textulal:
#
# * Count Vectors as features
# * TF-IDF Vectors as features
# * Word Embeddings as features
# * Text / NLP based features
# * Topic Models as features
#
# Se usará para esta práctica `TF-IDF Vectors as features`
#
# Está técnica requirede de múltiples parámetros:
#
# * `ngram_range`: We want to consider both unigrams and bigrams.
# * `max_df`: When building the vocabulary ignore terms that have a document
#     frequency strictly higher than the given threshold
# * `min_df`: When building the vocabulary ignore terms that have a document
#     frequency strictly lower than the given threshold.
# * `max_features`: If not None, build a vocabulary that only consider the top
#     max_features ordered by term frequency across the corpus.
#
# See `TfidfVectorizer?` for further detail.
#
# Es importante remarcar que se pueden experimentar con diferentes parámetros para maximizar la capacidad de predicción. Los parámetros iniciales son los siguientes:

# %%
# Parameter election
ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300

# %%
tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
print(features_train.shape)

# %% [markdown]
# Se puede utilizar el método de feature selection de Chi squared para identificar cómo se correlacionan los unigramas y bigramas de cada categoría:

# %%
from sklearn.feature_selection import chi2

for Product, category_id in sorted(category_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' category:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")

# %% [markdown]
# Se puede notar que los bigramas más característicos coresponden a la estructura básica de un requisito: _"El sistema deberá ...", "El producto deberá ...", "Los usuarios podrán..."._
#
# De esta forma se identifican los unigramas como los más característicos de una categoría.

# %%
bigrams

# %% [markdown]
# ## Entrenamiento de Modelo de _Machine Learning_
#
# Una vez creados los _feature vectors_, se provarán con diferentes modelos de clasificación de ML, para identificar los que tengan el mejor rendimeinto. Se probarán:
#
# * Multinomial Naïve Bayes
#
# La metodología para el entrenamiento de cada modelo será la siguiente:
#
# 1. Se deciden los hiperparámetros para afinar el modelo.
# 2. Se definen las métricas con las que se entranará el modelo. Se probará `accuracy`, `precision`, `recall`y `F-measure`.
# 3. Se realizará un proceso de `Randomized Search Cross Validation` para identificar la región de hiperparámetros con los que se logre la mayor `accuracy`.
# 4. Una vez identificada la región, se usará un proceso de `Grid Search Cross Validation` para identificar de forma exhaustiva la mejor combinación de hiperparámetros.
# 5. Conseguida la mejor combinación de hiperparámetros, se obtendrá el `accuracy` tanto del _training set_ como del _test set_, el reporte de la clasificación y la matriz de confusión.
# 6. Finalmente, se calculará el `accuracy` de un modelo con hiperparámetros _default_, para identificar si es posible obtener mejores resultados a través de la afinación de estos parámetros.
#
# Es importante remarcar, que los modelos sólo tienen conocimiento de las _categorias_ que define el dataset, mas no es una taxonomía específica. Así mismo, de encontrarse atributos de calidad con otra categoría externa a las identificadas en el entrenamiento, es seguro que será mal clasificado.

# %%
from sklearn.naive_bayes import MultinomialNB
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

# %%
print(features_train.shape)

# %% [markdown]
# ## Cross-Validation for Hyperparameter tuning

# %% [markdown]
# In the case of the Multinomial Naïve Bayes, we will not tune any hyperparameter.

# %%
mnbc = MultinomialNB()
mnbc

# %% [markdown]
# Let's fit it and see how it performs:

# %% [markdown]
# ## Model fit and performance

# %% [markdown]
# Now, we can fit the model to our training data:

# %%
mnbc.fit(features_train, labels_train)

# %% [markdown]
# For performance analysis, we will use the confusion matrix, the classification report and the accuracy on both training and test data:

# %% [markdown]
# #### Training accuracy

# %%
# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, mnbc.predict(features_train)))

# %%
from sklearn.model_selection import cross_val_score
print(cross_val_score(mnbc, features_train, labels_train, cv=10))

# %%
