import pandas as pd
import pandas.io.sql as sqlio
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

import plotly.express as px
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(
    page_title='⛽ Petróleo Brent',
    layout='wide'
)

#carregando os dados
dados = pd.read_csv('ipea_tudo.csv', parse_dates = ['ds'])

#carregando os dados para a previsão, com os dados a partir de 2018 para melhorar a precisão do modelo
dados_prev = pd.read_csv('ipea_prev.csv', parse_dates = ['ds'])

st.markdown("<h1 style='font-family: Roboto Light, sans-serif;'>Análise dos preços do petróleo brent</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='font-family: Roboto Light, sans-serif;'>Esse é um projeto elaborado por Isabela Maria Evangelista RM350099</h3>", unsafe_allow_html=True)

st.markdown("<h6 style='font-family: Roboto Light, sans-serif; text-align:justify;'>A maioria dos combustível utilizados em automóveis hoje em dia é derivada do petróleo, um combustível fóssil capaz de gerar diversos subprodutos. O petróleo brent é um pouco diferente do pretóleo comum, se trata da principal referência para a indústria petrolífera. Vindo do Mar Norte, entre a Noruega e a Dinamarca, esse tipo de petróleo tem densidade média e baixo enxofre na sua composição, o que o torna mais fácil de ser refinado e menos poluente ao ser queimado. E esse produto é facilmente influenciado por quatro grandes fatores: instabilidade geopolítica, eventos naturais e mudanças na política energética, além da influência das decisões da OPEP(Organização dos Países Exportadores de Petróleo).</h6>", unsafe_allow_html=True)
st.markdown("<h6 style='font-family: Roboto Light, sans-serif; text-align:justify;'>A instabilidade geopolítica, no caso do petróleo brent, não afeta tanto o preço por barril, porém afeta outros tipos de petróleo extraídos da região do Oriente Médio e Norte da África, onde os conflitos político-religiosos ocorrem com maior frequência e intensidade. Já os eventos naturais são mais complicados de lidar, porque muitas vezes não é possível controlar esses acontecimentos. Terremotos, maremotos, vazamentos no mar, inacessibilidade e falta de evidências de prova de reservas são alguns dos mais comuns eventos. A política energética é algo igualmente volátil e depende muito da sociedade, pois se trata da necessidade e do tipo de energia que a população mais consome ou deseja consumir. Recentemente, os apelos pela energia limpa aumentaram, como a energia de usinas hidrelétricas ou energia eólica, o que pode diminuir a demanda pelos combustíveis fósseis e, consequentemente, aumentar o preço destes.</h6>", unsafe_allow_html=True)
st.markdown("<h6 style='font-family: Roboto Light, sans-serif; text-align:justify;'>Mesmo assim, a descoberta e comercialização do petróleo brent é algo que demonstra a evolução tecnológica em busca de meios de continuarmos usando o petróleo, porém com o foco em poluir menos, o que não resolve para aqueles que buscam formas de combustíveis que não poluam nada. Por fim, a OPEP é quem mais tem influência direta e efetiva sobre o petróleo, porque é um órgão internacional que administra todos os assuntos relacionados à política petrolífera mundial. Alguns países que fazem parte da organização são: Kuwait, Venezuela, Argélia, Equador, Líbia, Nigéria e Emirados Árabes Unidos.</h6>", unsafe_allow_html=True)

st.markdown("<h3 style='font-family: Roboto Light, sans-serif;'>Qual período você deseja visualizar?</h3>", unsafe_allow_html=True)

filtro_ano = st.select_slider('nada',
                              options=dados['ds'].dt.year,
                              value=((dados['ds'].dt.year.min(), dados['ds'].dt.year.max())),
                              label_visibility='collapsed')

#aplicando os filtros
dados = dados[(dados['ds'].dt.year >= filtro_ano[0]) & (dados['ds'].dt.year <= filtro_ano[1])]

#separando a primeira linha em duas colunas
col1, col2 = st.columns(2)

#primeiro gráfico
dados_media_ano = dados.copy()

dados_media_ano['ano'] = dados_media_ano['ds'].dt.year

dados_media_ano = dados_media_ano.groupby(['ano'])['y'].mean().reset_index()

graf_media_ano = px.line(dados_media_ano, 
                         x='ano', 
                         y='y', 
                         title='Valor médio do petróleo brent por ano',
                         labels={
                            'y': 'Valor médio',
                            'ano': 'Ano'
                         },
                         color_discrete_sequence=px.colors.qualitative.Safe)


graf_media_ano.update_layout(
    font_family='Courier New',
    title_font_family='Roboto',
    title_font_size=23
)

col1.plotly_chart(graf_media_ano, use_container_width=True)

#segundo gráfico
dados_meses = dados.copy()

dados_meses['ano'] = dados_meses['ds'].dt.year
dados_meses['mes'] = dados_meses['ds'].dt.month

dados_meses = dados_meses.groupby(['mes'])['y'].mean().reset_index()

graf_media_mensal = px.bar(dados_meses,
                           x='mes',
                           y='y',
                           title='Média do valor do barril de petróleo brent por mês',
                           labels={
                               'mes': 'Mês',
                               'y': 'Média do valor do barril'
                           },
                           color_discrete_sequence=px.colors.qualitative.Safe)

graf_media_mensal.update_layout(
    font_family='Courier New',
    title_font_family='Roboto',
    title_font_size=23
)

col2.plotly_chart(graf_media_mensal, use_container_width=True)

#terceiro gráfico
dados_box = dados.copy()

dados_box['ano'] = dados_box['ds'].dt.year

graf_box = px.box(dados_box,
                  x='ano',
                  y='y',
                  title='Boxplots com a mediana, máximos, mínimos e quartis dos anos selecionados',
                  labels={
                      'ano': 'Ano',
                      'y': 'Valores'
                  },
                  color_discrete_sequence=px.colors.qualitative.Safe)

graf_box.update_layout(
    font_family='Courier New',
    title_font_family='Roboto',
    title_font_size=23
)

st.plotly_chart(graf_box, use_container_width=True)

modelo_prophet = Prophet(daily_seasonality=True)

modelo_prophet.fit(dados_prev)

future = modelo_prophet.make_future_dataframe(periods=365)

prev = modelo_prophet.predict(future)

graf_prev = plot_plotly(modelo_prophet, prev, uncertainty=True)

colors = px.colors.qualitative.Safe

graf_prev.update_traces(line=dict(color=colors[0]))

graf_prev.update_layout(
    font_family='Courier New',
    title_font_family='Roboto',
    title='Previsão dos preços do barril de petróleo brent',
    title_font_size=23 
)

st.plotly_chart(graf_prev, use_container_width=True)

st.markdown("<h6 style='font-family: Roboto Light, sans-serif; text-align:justify;'>AVISO! As informações apresentadas nesse site são apenas previsões e não necessariamente a verdade. Não nos responsabilizamos por quaisquer ações tomadas com base nos dados aqui exibidos.</h6>", unsafe_allow_html=True)

st.markdown("</br>", unsafe_allow_html=True)

st.markdown("<h3 style='font-family: Roboto Light, sans-serif; text-align:justify;'>Um pouco mais sobre a construção do projeto...</h3>", unsafe_allow_html=True)
st.markdown("<h6 style='font-family: Roboto Light, sans-serif; text-align:justify;'>Esse projeto tem origem no tech challenge da fase 4 da pós tech de Data Analytics da FIAP. Foram utilizados os dados da base da <a href='http://www.ipeadata.gov.br/Default.aspx'>IPEA</a>.</h6>", unsafe_allow_html=True)
st.image('ipea.jpg', caption='Site do IPEA')

st.markdown("<h6 style='font-family: Roboto Light, sans-serif; text-align:justify;'>Com a base de dados carregadas, os dados foram transformados e tratados via jupyter notebook, para remover, por exemplo, os dias de fim de semana em que os valores são nulos.</h6>", unsafe_allow_html=True)
st.image('valores_nulos.jpg', caption='Trecho de código em que os valores nulos são removidos')

st.markdown("<h6 style='font-family: Roboto Light, sans-serif; text-align:justify;'>Após o tratamento desse e de outros pequenos problemas no conjunto de dados, o modelo de previsão do <a href='https://facebook.github.io/prophet/docs/quick_start.html'>Prophet</a> foi o escolhido para a previsão dos valores dos barris de petróleo brent, sendo reafirmado como um bom modelo para essa finalidade por meio do cálculo das métricas de desenvolvimento.</h6>", unsafe_allow_html=True)
st.image('metricas.jpg', caption='Métricas de desenvolvimento calculadas a partir da eficácia do Prophet na previsão dos dados')

st.markdown("<h6 style='font-family: Roboto Light, sans-serif; text-align:justify;'>Esse site foi construído com o uso do <a href='https://docs.streamlit.io/'>streamlit</a> e as bibliotecas python <a href='https://pandas.pydata.org/'>pandas</a> e <a href='https://matplotlib.org/'>matplotlib</a>.</h6>", unsafe_allow_html=True)
