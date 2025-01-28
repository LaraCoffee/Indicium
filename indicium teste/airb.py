import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

st.set_page_config(page_title="Imoveis.com", layout="wide")


st.markdown("<h1 style='text-align: center;'>Analise de preços de imoveis</h1>", unsafe_allow_html=True)

#carregar arquivo
df = pd.read_csv("teste_indicium_precificacao.csv")

col4, col5 = st.columns(2)
col1, col2, col3 = st.columns(3)

#calculo da media de preços por bairro
preco_medio = df.groupby('bairro').agg(media_preco=('price', 'mean'),count=('id', 'size')).reset_index()
preco_medio['media_preco'] = preco_medio['media_preco'].round(2)
# preco_medio = preco_medio.sort_values(by='price', ascending=False)

preco_medio.columns = ['bairro', 'preco_medio', 'quantidade de ocorrencias']

#criar gráfico da media de preços overall
barra_preco_bairro = px.bar(preco_medio,x='bairro',y='preco_medio',text='quantidade de ocorrencias',
    title='Gráfico com Média de Preços e Ocorrências por Bairro',
    labels={'preco_medio': 'Preço Médio ($)','bairro': 'Bairro','quantidade': 'Número de Ocorrências'},color='preco_medio')

#grafico com media por distrito
preco_medio_bairro = df.groupby('bairro_group')["price"].mean().reset_index()
barra_media_distrito = px.bar(preco_medio_bairro, x="bairro_group", y="price", title="Preço Médio por Distrito", color="bairro_group")


#grafico com medeia de preços por tipo de quarto
preco_type = df[["room_type","price"]]
barra_tipo_quarto = px.bar(preco_type.groupby("room_type", as_index=False).mean(), x="room_type", y="price",
                 title="Preço Médio por Tipo de Quarto", labels={"room_type": "Tipo de Quarto", "price": "Preço Médio ($)"},color="room_type")



#grafico com preço medio e quantidade de categoria de quartos 
room_type_df = df.groupby("room_type").agg(quantidade=("room_type", "count"),preco_medio_room=("price", "mean")).reset_index()
barra_preco_room = px.bar(room_type_df, x="room_type", y="quantidade", text="preco_medio_room", title="Quantidade Ocorrências de Quartos por Tipo e Preço Médio", color="room_type")
barra_preco_room.update_traces(texttemplate='Preço Médio: $%{text:.2f}', textposition='outside')



with col4:
    st.plotly_chart(barra_media_distrito)
with col5:
    st.plotly_chart(barra_preco_room)
    pass    


with col1:
    st.write("Media de Preços por Bairro em New York:")
    st.dataframe(preco_medio)#dataframe com media dos valores por bairro e ocorreências
with col2:
    #mostrar grafico 
    st.plotly_chart(barra_preco_bairro)
with col3:
    st.plotly_chart(barra_tipo_quarto)




#modelo de perdiçao ja treinado 

#carrega o modelo
def carregar_modelo_treinado():
    modelo_carregado = joblib.load('treino_modelo.pkl')
    imputer_carregado = joblib.load('imputer.pkl')
    colunas_modelo = joblib.load('colunas_modelo.pkl')
    return modelo_carregado, imputer_carregado, colunas_modelo

#preve o preço aproximado do imovel
def prever_preco_imovel(imovel):
    modelo_carregado, imputer_carregado, colunas_modelo = carregar_modelo_treinado()
    imovel_df = pd.DataFrame([imovel])
    imovel_df = pd.DataFrame(imputer_carregado.transform(imovel_df), columns=imovel_df.columns)
    imovel_df = pd.get_dummies(imovel_df, columns=['bairro_group', 'bairro', 'room_type'], drop_first=True)
    imovel_df = imovel_df.reindex(columns=colunas_modelo, fill_value=0)
    preco_previsto = modelo_carregado.predict(imovel_df)[0]
    return preco_previsto


#imovel novo a ser previsto o valor
novo_imovel = {
    'bairro_group': 'Manhattan',
    'bairro': 'Midtown',
    'latitude': 40.75362,
    'longitude': -73.98377,
    'room_type': 'Entire home/apt',
    'minimo_noites': 1,
    'numero_de_reviews': 45,
    'reviews_por_mes': 0.38,
    'calculado_host_listings_count': 2,
    'disponibilidade_365': 355
}



df_novo_imovel = pd.DataFrame([novo_imovel])

preco = prever_preco_imovel(novo_imovel)#previsao do preço aproximado


st.write("referencias do imovel a ser testado:")
st.dataframe(df_novo_imovel)
st.write(f"Preço aproximado previsto para o imóvel: ${preco:.2f}")
