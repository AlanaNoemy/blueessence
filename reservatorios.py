
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from unicodedata import normalize
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings("ignore")


# =====================================================
# 1. CONFIGURACOES
# =====================================================

st.set_page_config(
    page_title="Blue Essence | Inteligência Hídrica",
    page_icon="💧",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
PASTA_DADOS = BASE_DIR / "dados"

HORIZONTE = 24

ARQUIVOS = {
    "Cantareira": "DADOS_CANTAREIRA.csv",
    "Alto Tietê": "DADOS_ALTO_DO_TIETE.csv",
    "Guarapiranga": "DADOS_GUARAPIRANGA.csv",
    "Rio Claro": "DADOS_RIO_CLARO.csv"
}

CORES = {
    "Cantareira": "#00C8FF",
    "Alto Tietê": "#00D084",
    "Guarapiranga": "#FFB000",
    "Rio Claro": "#FF5470"
}

FUNDO = "#202729"
PAINEL = "#303638"


# =====================================================
# 2. ESTILO
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #202729;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #172124;
}

h1, h2, h3 {
    color: #FFFFFF !important;
}

.metric-card {
    background-color: #303638;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #087FA5;
}

div[data-testid="stMetric"] {
    background-color: #303638;
    padding: 15px;
    border-radius: 10px;
    border-left: 3px solid #00C8FF;
}

.stTabs [data-baseweb="tab"] {
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. FUNCOES AUXILIARES
# =====================================================

def normalizar(texto):
    texto = normalize("NFKD", str(texto))
    texto = "".join(
        c for c in texto
        if not __import__("unicodedata").combining(c)
    )
    return texto.lower().strip()


def encontrar_coluna(df, termos, obrigatoria=False):

    for coluna in df.columns:
        nome = normalizar(coluna)

        if all(normalizar(t) in nome for t in termos):
            return coluna

    if obrigatoria:
        raise ValueError(
            f"Coluna não encontrada. Termos procurados: {termos}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    return None


def ler_csv(caminho):

    ultimo_erro = None

    for encoding in ["utf-8-sig", "latin1", "cp1252"]:

        try:
            return pd.read_csv(
                caminho,
                sep=None,
                engine="python",
                decimal=",",
                encoding=encoding
            )

        except Exception as erro:
            ultimo_erro = erro

    raise ValueError(
        f"Não foi possível ler {caminho}: {ultimo_erro}"
    )


# =====================================================
# 4. CARREGAMENTO DOS RESERVATORIOS
# =====================================================

@st.cache_data
def carregar_dados():

    reservatorios = {}

    for sistema, arquivo in ARQUIVOS.items():

        caminho = PASTA_DADOS / arquivo

        if not caminho.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {caminho}"
            )

        df = ler_csv(caminho)

        coluna_data = encontrar_coluna(
            df, ["data"], obrigatoria=True
        )

        coluna_volume = encontrar_coluna(
            df, ["volume", "util"], obrigatoria=True
        )

        coluna_total = encontrar_coluna(
            df, ["volume", "total"]
        )

        coluna_captada = encontrar_coluna(
            df, ["vazao", "captada"]
        )

        coluna_natural = encontrar_coluna(
            df, ["vazao", "natural"]
        )

        dados = pd.DataFrame()

        dados["Data"] = pd.to_datetime(
            df[coluna_data],
            dayfirst=True,
            errors="coerce"
        )

        dados["Volume Útil (%)"] = pd.to_numeric(
            df[coluna_volume],
            errors="coerce"
        )

        if coluna_total:
            dados["Volume Total (hm³)"] = pd.to_numeric(
                df[coluna_total],
                errors="coerce"
            )

        if coluna_captada:
            dados["Vazão Captada (m³/s)"] = pd.to_numeric(
                df[coluna_captada],
                errors="coerce"
            )

        if coluna_natural:
            dados["Vazão Natural (m³/s)"] = pd.to_numeric(
                df[coluna_natural],
                errors="coerce"
            )

        dados = dados.dropna(
            subset=["Data", "Volume Útil (%)"]
        )

        dados = dados.sort_values("Data")

        dados["Data"] = (
            dados["Data"].dt.to_period("M").dt.to_timestamp()
        )

        dados = dados.groupby("Data").mean(numeric_only=True)

        dados = dados.asfreq("MS")

        dados["Volume Útil (%)"] = (
            dados["Volume Útil (%)"]
            .interpolate(method="time", limit=2)
        )

        if dados["Volume Útil (%)"].isna().any():
            raise ValueError(
                f"Existem meses sem volume útil em {sistema}."
            )

        if not dados["Volume Útil (%)"].between(0, 100).all():
            raise ValueError(
                f"Volume útil fora do intervalo 0 a 100 em {sistema}."
            )

        if len(dados) < 36:
            raise ValueError(
                f"Histórico insuficiente para {sistema}."
            )

        reservatorios[sistema] = dados

    return reservatorios


# =====================================================
# 5. MODELOS PREDITIVOS
# =====================================================

def ajustar_modelo(modelo, serie, horizonte):

    if modelo == "Sazonal":

        valores = np.tile(
            serie.iloc[-12:].values,
            int(np.ceil(horizonte / 12))
        )[:horizonte]

        return valores, None, None

    if modelo == "ETS":

        ajuste = ExponentialSmoothing(
            serie,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated"
        ).fit(optimized=True)

        previsao = np.asarray(
            ajuste.forecast(horizonte)
        )

        residuos = (
            serie - ajuste.fittedvalues
        ).dropna().values

        return previsao, residuos, None

    if modelo == "SARIMA":

        ajuste = SARIMAX(
            serie,
            order=(1, 0, 1),
            seasonal_order=(1, 1, 0, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        resultado = ajuste.get_forecast(
            steps=horizonte
        )

        previsao = np.asarray(
            resultado.predicted_mean
        )

        intervalo = resultado.conf_int(alpha=0.20)

        inferior = np.asarray(intervalo.iloc[:, 0])
        superior = np.asarray(intervalo.iloc[:, 1])

        return previsao, inferior, superior

    raise ValueError("Modelo desconhecido")


# =====================================================
# 6. VALIDACAO TEMPORAL
# =====================================================

def avaliar_modelos(serie):

    modelos = ["Sazonal", "ETS", "SARIMA"]

    resultados = []

    # Teste com os 12 meses mais recentes.
    treino = serie.iloc[:-12]
    teste = serie.iloc[-12:]

    for modelo in modelos:

        try:

            previsao, _, _ = ajustar_modelo(
                modelo,
                treino,
                12
            )

            erros = teste.values - previsao

            mae = np.mean(np.abs(erros))

            rmse = np.sqrt(np.mean(erros ** 2))

            resultados.append({
                "Modelo": modelo,
                "MAE": mae,
                "RMSE": rmse
            })

        except Exception:
            continue

    if not resultados:
        raise RuntimeError(
            "Nenhum modelo conseguiu realizar a previsão."
        )

    return pd.DataFrame(resultados).sort_values(
        ["MAE", "RMSE"]
    ).reset_index(drop=True)


# =====================================================
# 7. PREVISAO DOS PROXIMOS 24 MESES
# =====================================================

def gerar_previsao(serie):

    serie = serie.astype(float)

    avaliacao = avaliar_modelos(serie)

    melhor_modelo = avaliacao.iloc[0]["Modelo"]

    previsao, limite1, limite2 = ajustar_modelo(
        melhor_modelo,
        serie,
        HORIZONTE
    )

    # Intervalo de confiança:
    # SARIMA fornece intervalo estatístico do próprio modelo.
    # ETS e Sazonal usam aproximação baseada nos resíduos.

    if melhor_modelo == "SARIMA":

        inferior = limite1
        superior = limite2

    else:

        if melhor_modelo == "Sazonal":

            residuos = (
                serie.iloc[12:].values -
                serie.iloc[:-12].values
            )

        else:
            residuos = limite1

        residuos = np.asarray(residuos)
        residuos = residuos[np.isfinite(residuos)]

        sigma = np.std(residuos, ddof=1)

        if not np.isfinite(sigma):
            sigma = 0

        horizonte = np.arange(1, HORIZONTE + 1)

        margem = 1.2816 * sigma * np.sqrt(horizonte)

        inferior = previsao - margem
        superior = previsao + margem

    previsao = np.clip(previsao, 0, 100)
    inferior = np.clip(inferior, 0, 100)
    superior = np.clip(superior, 0, 100)

    datas = pd.date_range(
        start=serie.index[-1] + pd.offsets.MonthBegin(1),
        periods=HORIZONTE,
        freq="MS"
    )

    return {
        "historico": serie,
        "datas": datas,
        "previsao": previsao,
        "inferior": inferior,
        "superior": superior,
        "modelo": melhor_modelo,
        "avaliacao": avaliacao
    }


# =====================================================
# 8. GRAFICO HISTORICO + PREDITIVO
# =====================================================

def criar_grafico(previsoes, sistema):

    dados = previsoes[sistema]

    cor = CORES[sistema]

    historico = dados["historico"]

    datas = dados["datas"]

    previsao = dados["previsao"]

    inferior = dados["inferior"]

    superior = dados["superior"]

    fig = go.Figure()

    # Histórico real

    fig.add_trace(go.Scatter(
        x=historico.index,
        y=historico.values,
        mode="lines",
        name="Histórico real",
        line=dict(
            color=cor,
            width=3
        )
    ))

    # Limite superior da previsão

    fig.add_trace(go.Scatter(
        x=datas,
        y=superior,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Limite inferior + preenchimento

    fig.add_trace(go.Scatter(
        x=datas,
        y=inferior,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(0,200,255,0.15)",
        line=dict(width=0),
        name="Intervalo de confiança (80%)",
        hoverinfo="skip"
    ))

    # Linha preditiva

    fig.add_trace(go.Scatter(
        x=[historico.index[-1]] + list(datas),
        y=[historico.iloc[-1]] + list(previsao),
        mode="lines+markers",
        name="Previsão (24 meses)",
        line=dict(
            color="#FFFFFF",
            width=2.5,
            dash="dash"
        ),
        marker=dict(size=5)
    ))

    # Separação entre passado e futuro

    fig.add_vline(
        x=historico.index[-1].timestamp() * 1000,
        line_dash="dot",
        line_color="#FFFFFF",
        annotation_text="Início da previsão",
        annotation_position="top"
    )

    fig.update_layout(
        title=f"{sistema} | Histórico e projeção",
        template="plotly_dark",
        paper_bgcolor=FUNDO,
        plot_bgcolor=PAINEL,
        height=500,
        hovermode="x unified",
        yaxis=dict(
            title="Volume útil (%)",
            range=[0, 100]
        ),
        xaxis=dict(
            title="Período",
            showgrid=True
        ),
        legend=dict(
            orientation="h",
            y=-0.20
        ),
        margin=dict(
            l=30,
            r=30,
            t=70,
            b=80
        )
    )

    return fig


# =====================================================
# 9. GRAFICO COMPARATIVO DOS RESERVATORIOS
# =====================================================

def grafico_comparativo(previsoes):

    fig = go.Figure()

    for sistema, dados in previsoes.items():

        cor = CORES[sistema]

        historico = dados["historico"]

        datas = dados["datas"]

        previsao = dados["previsao"]

        fig.add_trace(go.Scatter(
            x=historico.index,
            y=historico.values,
            mode="lines",
            name=f"{sistema} - Real",
            line=dict(color=cor, width=2)
        ))

        fig.add_trace(go.Scatter(
            x=[historico.index[-1]] + list(datas),
            y=[historico.iloc[-1]] + list(previsao),
            mode="lines",
            name=f"{sistema} - Previsto",
            line=dict(
                color=cor,
                width=2,
                dash="dash"
            )
        ))

    fig.add_vline(
        x=max(
            d["historico"].index[-1].timestamp() * 1000
            for d in previsoes.values()
        ),
        line_dash="dot",
        line_color="white"
    )

    fig.update_layout(
        title="Comparação dos quatro reservatórios",
        template="plotly_dark",
        paper_bgcolor=FUNDO,
        plot_bgcolor=PAINEL,
        height=550,
        yaxis=dict(
            title="Volume útil (%)",
            range=[0, 100]
        ),
        xaxis_title="Período",
        hovermode="x unified",
        legend=dict(orientation="h")
    )

    return fig


# =====================================================
# 10. INTERFACE PRINCIPAL
# =====================================================

def main():

    st.title("💧 BLUE ESSENCE")

    st.caption(
        "PLATAFORMA DE MONITORAMENTO E INTELIGÊNCIA HÍDRICA"
    )

    st.divider()

    try:

        with st.spinner("Carregando dados e treinando modelos..."):

            dados = carregar_dados()

            previsoes = {}

            for sistema, df in dados.items():

                coluna = "Volume Útil (%)"

                previsoes[sistema] = gerar_previsao(
                    df[coluna]
                )

    except Exception as erro:

        st.error(f"Erro ao processar os dados: {erro}")

        st.stop()

    # Menu lateral

    st.sidebar.title("BLUE ESSENCE")

    pagina = st.sidebar.radio(
        "Navegação",
        [
            "Visão geral",
            "Análise preditiva",
            "Desempenho dos modelos"
        ]
    )

    st.sidebar.divider()

    st.sidebar.caption(
        "Inteligência de dados aplicada aos recursos hídricos."
    )

    # ================================================
    # VISAO GERAL
    # ================================================

    if pagina == "Visão geral":

        st.header("Monitoramento dos reservatórios")

        colunas = st.columns(4)

        for i, (sistema, df) in enumerate(dados.items()):

            atual = df["Volume Útil (%)"].iloc[-1]

            anterior = df["Volume Útil (%)"].iloc[-2]

            colunas[i].metric(
                sistema,
                f"{atual:.2f}%",
                delta=f"{atual - anterior:+.2f} p.p."
            )

        st.divider()

        st.subheader("Volume útil histórico")

        fig = go.Figure()

        for sistema, df in dados.items():

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["Volume Útil (%)"],
                mode="lines",
                name=sistema,
                line=dict(
                    color=CORES[sistema],
                    width=2.5
                )
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=FUNDO,
            plot_bgcolor=PAINEL,
            height=500,
            yaxis=dict(
                title="Volume útil (%)",
                range=[0, 100]
            ),
            xaxis_title="Ano",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        st.subheader("Volume total e vazões")

        sistema = st.selectbox(
            "Selecione o reservatório",
            list(dados.keys())
        )

        df = dados[sistema]

        for coluna in [
            "Volume Total (hm³)",
            "Vazão Captada (m³/s)",
            "Vazão Natural (m³/s)"
        ]:

            if coluna in df.columns:

                grafico = go.Figure()

                grafico.add_trace(go.Scatter(
                    x=df.index,
                    y=df[coluna],
                    mode="lines",
                    name=coluna,
                    line=dict(
                        color=CORES[sistema],
                        width=2.5
                    )
                ))

                grafico.update_layout(
                    title=f"{sistema} - {coluna}",
                    template="plotly_dark",
                    paper_bgcolor=FUNDO,
                    plot_bgcolor=PAINEL,
                    height=350
                )

                st.plotly_chart(
                    grafico,
                    use_container_width=True
                )

    # ================================================
    # ANALISE PREDITIVA
    # ================================================

    elif pagina == "Análise preditiva":

        st.header("Inteligência preditiva")

        st.markdown(
            "Projeção do volume útil dos reservatórios "
            "para os próximos **24 meses**."
        )

        sistema = st.selectbox(
            "Escolha o reservatório",
            list(previsoes.keys())
        )

        resultado = previsoes[sistema]

        historico = resultado["historico"]

        previsao = resultado["previsao"]

        atual = historico.iloc[-1]

        final = previsao[-1]

        diferenca = final - atual

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Volume atual",
            f"{atual:.2f}%"
        )

        c2.metric(
            "Previsão em 24 meses",
            f"{final:.2f}%",
            delta=f"{diferenca:+.2f} p.p."
        )

        c3.metric(
            "Modelo selecionado",
            resultado["modelo"]
        )

        st.divider()

        st.plotly_chart(
            criar_grafico(previsoes, sistema),
            use_container_width=True
        )

        st.info(
            "A linha contínua representa os dados observados. "
            "A linha tracejada representa a previsão. "
            "A faixa sombreada representa o intervalo de confiança "
            "de 80% do modelo, ou uma aproximação baseada nos "
            "resíduos para os modelos sem intervalo nativo."
        )

        st.subheader("Tabela de previsão")

        tabela = pd.DataFrame({
            "Data": resultado["datas"],
            "Previsão (%)": resultado["previsao"],
            "Limite inferior (%)": resultado["inferior"],
            "Limite superior (%)": resultado["superior"]
        })

        st.dataframe(
            tabela.style.format({
                "Previsão (%)": "{:.2f}",
                "Limite inferior (%)": "{:.2f}",
                "Limite superior (%)": "{:.2f}"
            }),
            use_container_width=True
        )

        csv = tabela.to_csv(
            index=False,
            sep=";",
            decimal=","
        ).encode("utf-8-sig")

        st.download_button(
            "📥 Baixar previsão CSV",
            data=csv,
            file_name=f"previsao_{sistema}.csv",
            mime="text/csv"
        )

        st.divider()

        st.subheader("Comparação dos quatro sistemas")

        st.plotly_chart(
            grafico_comparativo(previsoes),
            use_container_width=True
        )
        

    # ================================================
    # DESEMPENHO DOS MODELOS
    # ================================================

    elif pagina == "Desempenho dos modelos":

        st.header("Avaliação dos modelos preditivos")

        st.markdown(
            "Os modelos são comparados utilizando os 12 meses "
            "mais recentes como período de teste."
        )

        for sistema, resultado in previsoes.items():

            st.subheader(sistema)

            st.caption(
                f"Modelo selecionado: {resultado['modelo']}"
            )

            st.dataframe(
                resultado["avaliacao"].style.format({
                    "MAE": "{:.3f}",
                    "RMSE": "{:.3f}"
                }),
                use_container_width=True
            )

            st.divider()

    st.caption(
        "BLUE ESSENCE | Análise baseada em séries temporais "
        "e dados históricos dos reservatórios."
    )


if __name__ == "__main__":
    main()
