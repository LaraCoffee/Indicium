import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import numpy as np
import joblib

#treinamento do modelo a ser utilizado para previsao de preço de imoveis

#carregar dataset
data = pd.read_csv('teste_indicium_precificacao.csv')

#selecionar as features e o alvo
features = [
    'bairro_group', 'bairro', 'latitude', 'longitude', 'room_type',
    'minimo_noites', 'numero_de_reviews', 'reviews_por_mes',
    'calculado_host_listings_count', 'disponibilidade_365'
]
alvo = 'price'

X = data[features]
y = data[alvo]

#tratamento de pontos fora do padrão no alvo pra remover valores muito altos
y = np.clip(y, None, y.quantile(0.95))

#preencher valores ausentes com os valors mais frequentes
imputer = SimpleImputer(strategy='most_frequent')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

#codificar variaveis categoricas
X = pd.get_dummies(X, columns=['bairro_group', 'bairro', 'room_type'], drop_first=True)

#dividir em conjuntos de treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#modelo random forrest 
randomf_model = RandomForestRegressor(n_estimators=100, random_state=42)
randomf_model.fit(X_train, y_train)
randomf_y_pred = randomf_model.predict(X_test)

#avaliaçao do modelo
radomf_mae = mean_absolute_error(y_test, randomf_y_pred)

print(f"MAE em dolares: ${radomf_mae:.2f}")


#salvar o modelo ja treinado, o imputer e as colunas
joblib.dump(randomf_model, 'treino_modelo.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(X_train.columns, 'colunas_modelo.pkl')

