

# ============================================================
# BLUE ESSENCE — PLATAFORMA DE INTELIGÊNCIA HÍDRICA
# Streamlit + Python + Pandas + Plotly + Statsmodels
# ============================================================

import re
import warnings
import unicodedata
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PASTA_DADOS = BASE_DIR / "dados"

ARQUIVOS = {
    "Cantareira": [
        "DADOS_CANTAREIRA.csv",
        "DADOS_CANTAREIRA(1).csv"
    ],
    "Alto Tietê": [
        "DADOS_ALTO_DO_TIETE.csv",
        "DADOS_ALTO_DO_TIETE(2).csv"
    ],
    "Guarapiranga": [
        "DADOS_GUARAPIRANGA.csv",
        "DADOS_GUARAPIRANGA(2).csv"
    ],
    "Rio Claro": [
        "DADOS_RIO_CLARO.csv",
        "DADOS_RIO_CLARO(1).csv"
    ]
}

CORES = {
    "Cantareira": "#13A8E0",
    "Alto Tietê": "#10B981",
    "Guarapiranga": "#F5A623",
    "Rio Claro": "#F16B8B"
}

AZUL = "#138FC5"
NAVY = "#153B5B"

INDICADORES = {
    "volume_util": {
        "nome": "Volume útil",
        "unidade": "%",
        "limites": (0, 100)
    },
    "volume_total": {
        "nome": "Volume total",
        "unidade": "hm³",
        "limites": (None, None)
    },
    "vazao_captada": {
        "nome": "Vazão captada",
        "unidade": "m³/s",
        "limites": (None, None)
    },
    "vazao_natural": {
        "nome": "Vazão natural",
        "unidade": "m³/s",
        "limites": (None, None)
    }
}

st.set_page_config(
    page_title="Blue Essence | Inteligência Hídrica",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. IDENTIDADE VISUAL
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

.stApp {
    background: #F1F9FE;
    color: #183C5A;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #E2F3FC;
    border-right: 1px solid #D2EAF7;
}

[data-testid="stSidebar"] * {
    color: #17466A;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.hero {
    background: linear-gradient(120deg, #DDF4FF, #F5FCFF);
    border: 1px solid #C9EAF9;
    border-radius: 24px;
    padding: 30px 35px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #153F63;
    font-size: 36px;
    font-weight: 900;
    margin-bottom: 5px;
}

.hero p {
    color: #507D9D;
    font-size: 16px;
}

.metric-card, .info-card {
    background: white;
    border: 1px solid #D9ECF7;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(30,100,140,0.05);
}

.metric-card {
    min-height: 120px;
}

.metric-label {
    color: #7191A8;
    font-size: 13px;
    font-weight: 700;
}

.metric-value {
    color: #17466A;
    font-size: 29px;
    font-weight: 900;
    margin-top: 8px;
}

.metric-caption {
    color: #7A9AB0;
    font-size: 12px;
}

.section-title {
    color: #17466A;
    font-weight: 900;
    font-size: 23px;
    margin-top: 22px;
    margin-bottom: 12px;
}

.info-card {
    margin-bottom: 12px;
    line-height: 1.8;
}

div[data-testid="stPlotlyChart"] {
    background: white;
    border-radius: 16px;
    padding: 8px;
    border: 1px solid #E0EDF5;
}

.stButton button {
    border-radius: 12px;
    background: #168FCB;
    color: white;
    border: none;
    font-weight: 800;
}

.stButton button:hover {
    background: #087AB5;
    color: white;
}

div[class*="st-key-abrir_monitoramento_"] button {
    background: #8ED8F8;
    color: #17466A;
}

div[class*="st-key-abrir_monitoramento_"] button:hover {
    background: #72CCF4;
    color: #17466A;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. FUNÇÕES DE TRATAMENTO DOS DADOS
# ============================================================

def normalizar(texto):
    texto = str(texto).lower().strip()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(
        c for c in texto
        if not unicodedata.combining(c)
    )


def localizar_arquivo(nomes):
    if not PASTA_DADOS.exists():
        raise FileNotFoundError(
            f"A pasta de dados não foi encontrada: {PASTA_DADOS}"
        )

    arquivos = list(PASTA_DADOS.glob("*.csv"))

    for nome in nomes:
        for arquivo in arquivos:
            if normalizar(arquivo.name) == normalizar(nome):
                return arquivo

    raise FileNotFoundError(
        f"Nenhum arquivo encontrado para: {nomes}"
    )


def ler_csv(caminho):
    for encoding in ["utf-8-sig", "latin1", "cp1252"]:
        try:
            return pd.read_csv(
                caminho,
                sep=None,
                engine="python",
                encoding=encoding,
                decimal=","
            )
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    raise ValueError(f"Não foi possível ler {caminho.name}")


def encontrar_coluna(df, termos):
    for coluna in df.columns:
        nome = normalizar(coluna)
        if all(termo in nome for termo in termos):
            return coluna
    return None


def converter_numero(serie):
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")

    def converter(valor):
        if pd.isna(valor):
            return np.nan

        valor = str(valor).strip()
        valor = valor.replace("%", "").replace(" ", "")

        if "," in valor:
            valor = valor.replace(".", "")
            valor = valor.replace(",", ".")

        try:
            return float(valor)
        except ValueError:
            return np.nan

    return serie.map(converter)


@st.cache_data
def carregar_dados():
    sistemas = {}

    for sistema, nomes in ARQUIVOS.items():
        caminho = localizar_arquivo(nomes)
        df = ler_csv(caminho)

        coluna_data = next(
            (
                c for c in df.columns
                if normalizar(c) in ["data", "mes", "date"]
            ),
            None
        )

        if coluna_data is None:
            raise ValueError(
                f"Coluna de data não encontrada em {caminho.name}"
            )

        df[coluna_data] = pd.to_datetime(
            df[coluna_data],
            dayfirst=True,
            errors="coerce"
        )

        df = df.dropna(subset=[coluna_data])
        df = df.sort_values(coluna_data)
        df = df.set_index(coluna_data)
        df.index = df.index.to_period("M").to_timestamp()
        df = df[~df.index.duplicated(keep="last")]

        colunas = {
            "volume_util": encontrar_coluna(df, ["volume util"]),
            "volume_total": encontrar_coluna(df, ["volume total"]),
            "vazao_captada": encontrar_coluna(df, ["vazao captada"]),
            "vazao_natural": encontrar_coluna(df, ["vazao natural"])
        }

        candidatas = [
            c for c in df.columns
            if "volume util" in normalizar(c)
        ]

        percentuais = [
            c for c in candidatas
            if "%" in str(c) or "percent" in normalizar(c)
        ]

        if percentuais:
            colunas["volume_util"] = percentuais[0]

        if colunas["volume_util"] is None:
            raise ValueError(
                f"Volume útil percentual não encontrado em {sistema}."
            )

        resultado = pd.DataFrame(index=df.index)

        for indicador, coluna in colunas.items():
            if coluna is not None:
                resultado[indicador] = converter_numero(df[coluna])

        resultado = resultado.sort_index()
        resultado = resultado[~resultado.index.duplicated(keep="last")]

        # Frequência mensal e preenchimento de lacunas internas.
        resultado = resultado.asfreq("MS")

        for coluna in resultado.columns:
            resultado[coluna] = resultado[coluna].interpolate(
                method="time",
                limit_area="inside"
            )

        resultado["volume_util"] = resultado["volume_util"].clip(0, 100)

        sistemas[sistema] = resultado

    return sistemas


# ============================================================
# 4. CONFIGURAÇÃO DOS GRÁFICOS
# ============================================================

def configurar_grafico(fig, altura=390):
    fig.update_layout(
        template="plotly_white",
        height=altura,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Nunito", color=NAVY, size=12),
        margin=dict(l=20, r=20, t=85, b=30),
        hovermode="x unified",
        title=dict(x=0.01, xanchor="left"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.5,
            font=dict(size=10)
        ),
        xaxis=dict(showgrid=False, linecolor="#E2EDF4"),
        yaxis=dict(gridcolor="#EAF1F6", zeroline=False)
    )
    return fig


def grafico_volume(dados, selecionados=None):
    fig = go.Figure()

    if selecionados is None:
        selecionados = list(dados.keys())

    for sistema in selecionados:
        serie = dados[sistema]["volume_util"].dropna()

        fig.add_trace(go.Scatter(
            x=serie.index,
            y=serie.values,
            name=sistema,
            mode="lines",
            line=dict(color=CORES[sistema], width=3)
        ))

    adicionar_previsoes(
        fig,
        {
            sistema: dados[sistema]["volume_util"]
            for sistema in selecionados
        },
        "volume_util"
    )

    fig.update_layout(
        title="Evolução histórica do volume útil",
        yaxis_title="Volume útil (%)",
        yaxis=dict(range=[0, 100])
    )

    return configurar_grafico(fig)


def grafico_volume_total(dados):
    fig = go.Figure()
    series = {}

    for sistema, df in dados.items():
        if "volume_total" not in df.columns:
            continue

        serie = df["volume_total"].dropna()

        if not serie.empty:
            fig.add_trace(go.Scatter(
                x=serie.index,
                y=serie.values,
                name=sistema,
                line=dict(color=CORES[sistema], width=2.5)
            ))
            series[sistema] = serie

    adicionar_previsoes(fig, series, "volume_total")

    fig.update_layout(
        title="Volume total armazenado",
        yaxis_title="Volume (hm³)"
    )

    return configurar_grafico(fig)


def grafico_vazoes(df, sistema):
    fig = go.Figure()

    for indicador, nome, estilo in [
        ("vazao_captada", "Vazão captada", "solid"),
        ("vazao_natural", "Vazão natural", "dash")
    ]:
        if indicador not in df.columns:
            continue

        serie = df[indicador].dropna()

        if not serie.empty:
            fig.add_trace(go.Scatter(
                x=serie.index,
                y=serie.values,
                name=nome,
                mode="lines",
                line=dict(
                    color=CORES[sistema],
                    width=2.5,
                    dash=estilo
                )
            ))

        adicionar_previsoes(
            fig,
            {sistema: serie},
            indicador,
            nome_indicador=nome
        )

    fig.update_layout(
        title=f"Comportamento das vazões — {sistema}",
        yaxis_title="Vazão (m³/s)"
    )

    return configurar_grafico(fig)


# ============================================================
# 5. MODELOS PREDITIVOS
# ============================================================

def executar_modelo(serie, modelo, horizonte=24):
    serie = serie.astype(float)

    if modelo == "Sazonal":
        if len(serie) < 12:
            return np.repeat(serie.iloc[-1], horizonte)

        return np.tile(
            serie.iloc[-12:].values,
            int(np.ceil(horizonte / 12))
        )[:horizonte]

    if modelo == "ETS":
        sazonalidade = "add" if len(serie) >= 24 else None

        ajuste = ExponentialSmoothing(
            serie,
            trend="add",
            damped_trend=True,
            seasonal=sazonalidade,
            seasonal_periods=12 if sazonalidade else None,
            initialization_method="estimated"
        ).fit(optimized=True)

        return np.asarray(ajuste.forecast(horizonte))

    if modelo == "SARIMA":
        sazonalidade = (0, 1, 1, 12) if len(serie) >= 36 else (0, 0, 0, 0)

        ajuste = SARIMAX(
            serie,
            order=(1, 0, 0),
            seasonal_order=sazonalidade,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        return np.asarray(ajuste.forecast(horizonte))

    raise ValueError("Modelo não reconhecido.")


def avaliar_modelos(serie):
    modelos = ["Sazonal", "ETS", "SARIMA"]
    resultados = []

    # Backtesting temporal: treina no passado e testa nos 12 meses seguintes.
    origens = [
        len(serie) - 36,
        len(serie) - 24,
        len(serie) - 12
    ]

    for modelo in modelos:
        erros = []

        for origem in origens:
            if origem < 24:
                continue

            treino = serie.iloc[:origem]
            teste = serie.iloc[origem:origem + 12]

            if len(teste) != 12:
                continue

            try:
                previsao = executar_modelo(treino, modelo, 12)
                erros.extend(teste.values - previsao)
            except Exception:
                continue

        if erros:
            erros = np.asarray(erros)

            resultados.append({
                "Modelo": modelo,
                "MAE": np.mean(np.abs(erros)),
                "RMSE": np.sqrt(np.mean(erros ** 2)),
                "Viés": np.mean(erros),
                "Observações testadas": len(erros)
            })

    if not resultados:
        return pd.DataFrame([{
            "Modelo": "Sazonal",
            "MAE": np.nan,
            "RMSE": np.nan,
            "Viés": np.nan,
            "Observações testadas": 0
        }])

    return pd.DataFrame(resultados).sort_values(
        ["MAE", "RMSE"]
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def gerar_previsao(valores, indice, indicador):
    serie = pd.Series(
        valores,
        index=pd.to_datetime(indice)
    ).dropna()

    serie = serie.asfreq("MS").interpolate(method="time")

    if len(serie) < 12:
        raise ValueError(
            "São necessários pelo menos 12 meses de dados "
            "para realizar a projeção."
        )

    avaliacao = avaliar_modelos(serie)
    melhor = avaliacao.iloc[0]["Modelo"]

    previsao = executar_modelo(serie, melhor, 24)

    # Resíduos sazonais para estimar uma faixa de incerteza.
    if len(serie) > 12:
        residuos = (
            serie.iloc[12:].values -
            serie.iloc[:-12].values
        )
    else:
        residuos = serie.diff().dropna().values

    sigma = np.std(residuos, ddof=1) if len(residuos) > 1 else 0

    # A incerteza cresce com o horizonte.
    margem = 1.96 * sigma * np.sqrt(
        np.arange(1, 25) / 12
    )

    inferior = previsao - margem
    superior = previsao + margem

    limites = INDICADORES[indicador]["limites"]

    if limites[0] is not None:
        previsao = np.maximum(previsao, limites[0])
        inferior = np.maximum(inferior, limites[0])

    if limites[1] is not None:
        previsao = np.minimum(previsao, limites[1])
        superior = np.minimum(superior, limites[1])

    datas = pd.date_range(
        serie.index[-1] + pd.offsets.MonthBegin(1),
        periods=24,
        freq="MS"
    )

    return (
        serie,
        datas,
        previsao,
        inferior,
        superior,
        melhor,
        avaliacao
    )


def adicionar_previsoes(
    fig, series_por_sistema, indicador, nome_indicador=None,
    diferenciar=False
):
    if not st.session_state.get("exibir_previsoes", False):
        return fig

    nome_indicador = nome_indicador or INDICADORES[indicador]["nome"]
    falhas = []

    for sistema, serie in series_por_sistema.items():
        serie = serie.dropna()

        if len(serie) < 12:
            falhas.append(
                f"{sistema}: são necessários pelo menos 12 registros"
            )
            continue

        try:
            with st.spinner(
                f"Calculando projeção de {nome_indicador.lower()} "
                f"para {sistema}..."
            ):
                (
                    serie_modelo,
                    datas,
                    previsao,
                    _,
                    _,
                    _,
                    _
                ) = gerar_previsao(
                    tuple(serie.values),
                    tuple(serie.index.astype(str)),
                    indicador
                )
        except Exception as erro:
            falhas.append(f"{sistema}: {erro}")
            continue

        if diferenciar:
            valores_observados = serie_modelo.diff().dropna()
            if valores_observados.empty:
                continue
            ultimo_valor = valores_observados.iloc[-1]
            valores_previsao = np.diff(
                np.concatenate(([serie_modelo.iloc[-1]], previsao))
            )
            datas_grafico = [serie_modelo.index[-1], *datas]
            valores_grafico = np.concatenate(
                ([ultimo_valor], valores_previsao)
            )
        else:
            datas_grafico = [serie_modelo.index[-1], *datas]
            valores_grafico = np.concatenate(
                ([serie_modelo.iloc[-1]], previsao)
            )

        exibir_chave_previsao = not any(
            trace.legendgroup == "previsoes_24_meses"
            and trace.showlegend is not False
            for trace in fig.data
        )

        fig.add_trace(go.Scatter(
            x=datas_grafico,
            y=valores_grafico,
            name=(
                "Previsão (24 meses)"
                if exibir_chave_previsao
                else (
                    f"{sistema} — {nome_indicador.lower()} "
                    "— previsão 24 meses"
                )
            ),
            legendgroup="previsoes_24_meses",
            showlegend=exibir_chave_previsao,
            mode="lines",
            line=dict(
                color=CORES[sistema],
                width=2.5,
                dash="dash"
            ),
            hovertemplate=(
                f"{sistema} — {nome_indicador}<br>"
                "%{x|%b %Y}<br>%{y:.2f}<extra></extra>"
            )
        ))

    if falhas:
        st.warning(
            f"Não foi possível gerar a previsão de {nome_indicador.lower()} "
            f"para: {'; '.join(falhas)}."
        )

    return fig


def grafico_previsao(
    serie, datas, previsao, inferior, superior, indicador, sistema
):
    nome = INDICADORES[indicador]["nome"]
    unidade = INDICADORES[indicador]["unidade"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=serie.index,
        y=serie.values,
        name="Histórico observado",
        line=dict(color=AZUL, width=3)
    ))

    fig.add_trace(go.Scatter(
        x=datas,
        y=superior,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=datas,
        y=inferior,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(19,168,224,0.16)",
        name="Faixa aproximada de incerteza"
    ))

    fig.add_trace(go.Scatter(
        x=datas,
        y=previsao,
        name="Projeção",
        line=dict(color="#F29D38", width=3, dash="dash")
    ))

    fig.add_vline(
        x=serie.index[-1],
        line_dash="dot",
        line_color="#7895A9"
    )

    fig.update_layout(
        title=f"{nome} — histórico e projeção de {sistema}",
        yaxis_title=f"{nome} ({unidade})"
    )

    if indicador == "volume_util":
        fig.update_yaxes(range=[0, 100])

    return configurar_grafico(fig, 460)


# ============================================================
# 6. COMPONENTES VISUAIS
# ============================================================

def titulo(texto, subtitulo=None):
    st.markdown(
        f'<div class="section-title">{texto}</div>',
        unsafe_allow_html=True
    )

    if subtitulo:
        st.caption(subtitulo)


def card_metrica(nome, valor, descricao, cor):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{nome}</div>
        <div class="metric-value" style="color:{cor}">
            {valor}
        </div>
        <div class="metric-caption">{descricao}</div>
    </div>
    """, unsafe_allow_html=True)


def texto_analise(texto):
    texto = texto.replace("\n", "<br>")
    texto = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", texto, flags=re.DOTALL)
    st.markdown(
        f'<div class="info-card">{texto}</div>',
        unsafe_allow_html=True
    )


def formatar_valor(valor, unidade):
    if pd.isna(valor):
        return "Não disponível"
    return f"{valor:.2f} {unidade}"


# ============================================================
# 7. ANÁLISES AUTOMÁTICAS
# ============================================================

def analisar_tendencia(serie):
    serie = serie.dropna()

    if len(serie) < 3:
        return "Não há dados suficientes para identificar uma tendência."

    janela = min(12, max(3, len(serie) // 2))
    media_inicial = serie.iloc[:janela].mean()
    media_final = serie.iloc[-janela:].mean()
    diferenca = media_final - media_inicial

    if abs(diferenca) < 0.5:
        return (
            "O indicador apresenta comportamento relativamente estável "
            "quando comparadas as médias dos períodos inicial e final."
        )

    if diferenca > 0:
        direcao = "crescimento"
        efeito = "aumento"
    else:
        direcao = "redução"
        efeito = "queda"

    return (
        f"Observa-se uma tendência de {direcao}: a média dos "
        f"últimos {janela} registros apresenta {efeito} de "
        f"{abs(diferenca):.2f} {INDICADORES['volume_util']['unidade']} "
        f"em relação à média do período inicial comparado."
    )


def analise_reservatorio(df, sistema):
    serie = df["volume_util"].dropna()

    atual = serie.iloc[-1]
    anterior = serie.iloc[-2] if len(serie) > 1 else atual
    variacao = atual - anterior

    minimo = serie.min()
    maximo = serie.max()
    media = serie.mean()
    desvio = serie.std()

    if atual >= 70:
        situacao = "nível elevado de armazenamento"
    elif atual >= 40:
        situacao = "nível intermediário de armazenamento"
    else:
        situacao = "nível reduzido de armazenamento"

    if variacao > 0:
        movimento = "aumento"
    elif variacao < 0:
        movimento = "redução"
    else:
        movimento = "estabilidade"

    return f"""
    **{sistema} — diagnóstico do volume útil**

    O último registro apresenta **{atual:.2f}%** de volume útil,
    caracterizando {situacao}. Em relação ao mês anterior,
    houve {movimento} de **{abs(variacao):.2f} pontos percentuais**.

    Considerando toda a série histórica disponível, a média é
    **{media:.2f}%**, o mínimo observado é **{minimo:.2f}%**
    e o máximo é **{maximo:.2f}%**. O desvio padrão de
    **{desvio:.2f} pontos percentuais** representa a dispersão
    histórica dos níveis de armazenamento.

    {analisar_tendencia(serie)}
    """


def gerar_analise_comparativa(dados):
    valores = {
        sistema: df["volume_util"].dropna().iloc[-1]
        for sistema, df in dados.items()
        if not df["volume_util"].dropna().empty
    }

    if not valores:
        return "Não existem dados suficientes para a comparação."

    maior = max(valores, key=valores.get)
    menor = min(valores, key=valores.get)

    return f"""
    **Comparação entre os sistemas**

    No último registro disponível, **{maior}** apresenta o maior
    volume útil relativo, com **{valores[maior]:.2f}%**.
    Já **{menor}** apresenta o menor percentual, com
    **{valores[menor]:.2f}%**.

    A comparação percentual permite observar a situação relativa
    de armazenamento de cada sistema. Entretanto, os valores em
    hm³ não devem ser interpretados como diretamente equivalentes,
    pois os reservatórios possuem capacidades diferentes.
    """


# ============================================================
# 8. CARREGAMENTO DOS DADOS
# ============================================================

try:
    with st.spinner("Carregando os dados dos reservatórios..."):
        dados = carregar_dados()

except Exception as erro:
    st.error(f"Erro ao carregar os dados: {erro}")

    st.info(
        "Verifique se os quatro arquivos CSV estão dentro da pasta "
        "'dados' e se possuem as colunas de data e volume útil."
    )

    st.stop()


# ============================================================
# 9. CABEÇALHO DO SITE
# ============================================================

st.markdown("""
<div class="hero">
    <div style="font-size:15px;color:#138FC5;font-weight:900;">
        💧 BLUE ESSENCE | INTELIGÊNCIA HÍDRICA
    </div>
    <h1>A essência de tudo começa com água.</h1>
    <p>
        Monitoramento, análise científica e projeção dos principais
        sistemas de reservatórios de São Paulo.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 10. MENU LATERAL
# ============================================================

def abrir_monitoramento(sistema):
    st.session_state["pagina"] = "Monitoramento"
    st.session_state["sistema_monitoramento"] = sistema


with st.sidebar:

    st.markdown("""
    <div style="text-align:center;padding:15px 5px 25px;">
        <div style="font-size:42px;">💧</div>
        <div style="font-size:24px;font-weight:900;color:#147EBA;">
            Blue Essence
        </div>
        <div style="font-size:10px;letter-spacing:2px;">
            INTELIGÊNCIA HÍDRICA
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    pagina = st.radio(
        "NAVEGAÇÃO",
        [
            "Visão geral",
            "Monitoramento",
            "Inteligência preditiva",
            "Análise científica"
        ],
        key="pagina"
    )

    st.divider()
    st.toggle(
        "Exibir preditivos (24 meses)",
        key="exibir_previsoes",
        help=(
            "Sobrepõe projeções de dois anos nos gráficos de séries "
            "temporais. Gráficos estatísticos não são alterados."
        )
    )

    st.divider()
    st.caption("SISTEMAS MONITORADOS")
    st.caption("Clique em um sistema para abrir seu monitoramento.")

    st.button(
        "🌐 Todos os reservatórios",
        key="abrir_monitoramento_todos",
        use_container_width=True,
        on_click=abrir_monitoramento,
        args=("Todos",)
    )

    icones_sistemas = {
        "Cantareira": "🔵",
        "Alto Tietê": "🟢",
        "Guarapiranga": "🟠",
        "Rio Claro": "🔴"
    }

    for sistema in dados:
        st.button(
            f"{icones_sistemas.get(sistema, '🔹')} {sistema}",
            key=f"abrir_monitoramento_{normalizar(sistema)}",
            use_container_width=True,
            on_click=abrir_monitoramento,
            args=(sistema,)
        )

    st.divider()

    if st.button("Atualizar dados", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption("Blue Essence • Ciência a favor da água")


# ============================================================
# 11. VISÃO GERAL
# ============================================================

if pagina == "Visão geral":

    titulo(
        "Visão geral dos reservatórios",
        "Indicadores atualizados conforme o último registro dos arquivos."
    )

    colunas = st.columns(4)

    for coluna, (sistema, df) in zip(colunas, dados.items()):
        serie = df["volume_util"].dropna()

        atual = serie.iloc[-1]
        anterior = serie.iloc[-2] if len(serie) > 1 else atual

        with coluna:
            card_metrica(
                sistema,
                f"{atual:.2f}%",
                f"{atual - anterior:+.2f} p.p. no último mês",
                CORES[sistema]
            )

    titulo("Panorama histórico")

    st.plotly_chart(
        grafico_volume(dados),
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            grafico_volume_total(dados),
            use_container_width=True
        )

    with c2:
        fig = go.Figure()

        for sistema, df in dados.items():
            fig.add_trace(go.Box(
                y=df["volume_util"].dropna(),
                name=sistema,
                marker_color=CORES[sistema]
            ))

        fig.update_layout(
            title="Distribuição histórica do volume útil",
            yaxis_title="Volume útil (%)"
        )

        st.plotly_chart(
            configurar_grafico(fig),
            use_container_width=True
        )

    titulo("Análise comparativa")

    texto_analise(gerar_analise_comparativa(dados))

    titulo("Diagnóstico individual")

    for sistema, df in dados.items():
        with st.expander(f"Análise de {sistema}"):
            texto_analise(analise_reservatorio(df, sistema))


# ============================================================
# 12. MONITORAMENTO
# ============================================================

elif pagina == "Monitoramento":

    titulo(
        "Monitoramento operacional",
        "Explore um reservatório ou compare todos os sistemas."
    )

    sistema = st.selectbox(
        "Selecione o reservatório",
        ["Todos"] + list(dados.keys()),
        key="sistema_monitoramento"
    )

    if sistema == "Todos":
        titulo(
            "Comparativo dos reservatórios",
            "Compare o volume útil atual e o histórico de cada sistema."
        )

        colunas = st.columns(len(dados))
        comparativo = []

        for coluna, (nome_sistema, df_sistema) in zip(
            colunas, dados.items()
        ):
            serie_sistema = df_sistema["volume_util"].dropna()
            atual_sistema = serie_sistema.iloc[-1]
            anterior_sistema = (
                serie_sistema.iloc[-2]
                if len(serie_sistema) > 1
                else atual_sistema
            )

            with coluna:
                card_metrica(
                    nome_sistema,
                    f"{atual_sistema:.2f}%",
                    f"{atual_sistema - anterior_sistema:+.2f} p.p. "
                    "no último mês",
                    CORES[nome_sistema]
                )

            comparativo.append({
                "Reservatório": nome_sistema,
                "Volume útil atual (%)": f"{atual_sistema:.2f}",
                "Média histórica (%)": f"{serie_sistema.mean():.2f}",
                "Mínimo (%)": f"{serie_sistema.min():.2f}",
                "Máximo (%)": f"{serie_sistema.max():.2f}"
            })

        st.dataframe(
            pd.DataFrame(comparativo).set_index("Reservatório"),
            use_container_width=True
        )

        fig = grafico_volume(dados)
        fig.update_layout(title="Comparativo histórico do volume útil")
        st.plotly_chart(fig, use_container_width=True)

        if any(
            "volume_total" in df_sistema.columns
            and df_sistema["volume_total"].notna().any()
            for df_sistema in dados.values()
        ):
            st.plotly_chart(
                grafico_volume_total(dados),
                use_container_width=True
            )
    else:
        df = dados[sistema]
        serie = df["volume_util"].dropna()

        atual = serie.iloc[-1]
        minimo = serie.min()
        maximo = serie.max()
        media = serie.mean()

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            card_metrica(
                "Volume útil atual",
                f"{atual:.2f}%",
                "Último registro",
                CORES[sistema]
            )

        with c2:
            card_metrica(
                "Média histórica",
                f"{media:.2f}%",
                "Média da série",
                AZUL
            )

        with c3:
            card_metrica(
                "Maior volume",
                f"{maximo:.2f}%",
                "Máximo observado",
                "#10B981"
            )

        with c4:
            card_metrica(
                "Menor volume",
                f"{minimo:.2f}%",
                "Mínimo observado",
                "#E77B76"
            )

        titulo("Evolução do volume útil")

        st.plotly_chart(
            grafico_volume(dados, [sistema]),
            use_container_width=True
        )

        if "volume_total" in df.columns:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["volume_total"],
                name="Volume total",
                line=dict(color=CORES[sistema], width=3)
            ))

            adicionar_previsoes(
                fig,
                {sistema: df["volume_total"]},
                "volume_total"
            )

            fig.update_layout(
                title="Volume total armazenado",
                yaxis_title="Volume (hm³)"
            )

            st.plotly_chart(
                configurar_grafico(fig),
                use_container_width=True
            )

        if "vazao_captada" in df.columns or "vazao_natural" in df.columns:
            st.plotly_chart(
                grafico_vazoes(df, sistema),
                use_container_width=True
            )

        titulo("Diagnóstico do sistema")
        texto_analise(analise_reservatorio(df, sistema))

        with st.expander("Consultar dados históricos"):
            tabela = df.copy()
            tabela.index.name = "Data"

            st.dataframe(
                tabela,
                use_container_width=True
            )

            st.download_button(
                "Baixar histórico em CSV",
                tabela.to_csv(
                    sep=";",
                    decimal=",",
                    encoding="utf-8-sig"
                ),
                file_name=f"historico_{normalizar(sistema)}.csv",
                mime="text/csv"
            )


# ============================================================
# 13. INTELIGÊNCIA PREDITIVA
# ============================================================

elif pagina == "Inteligência preditiva":

    if not st.session_state.get("exibir_previsoes", False):
        st.info(
            "Ative “Exibir preditivos (24 meses)” no menu lateral "
            "para visualizar as projeções."
        )
        st.stop()

    titulo(
        "Inteligência preditiva",
        "Projeções estatísticas e análise dos próximos 24 meses."
    )

    sistema = st.selectbox(
        "Selecione o reservatório",
        list(dados.keys()),
        key="sistema_previsao"
    )

    indicadores_disponiveis = [
        indicador
        for indicador in INDICADORES
        if indicador in dados[sistema].columns
        and dados[sistema][indicador].notna().sum() >= 12
    ]

    if not indicadores_disponiveis:
        st.warning(
            "Não existem indicadores com pelo menos 12 registros "
            "mensais para realizar previsões."
        )
        st.stop()

    opcoes_indicadores = ["Tudo"] + indicadores_disponiveis

    indicador_selecionado = st.selectbox(
        "Indicador para previsão",
        opcoes_indicadores,
        format_func=lambda x: (
            "Todos os indicadores disponíveis"
            if x == "Tudo"
            else (
                f"{INDICADORES[x]['nome']} "
                f"({INDICADORES[x]['unidade']})"
            )
        )
    )

    if indicador_selecionado == "Tudo":
        indicadores_exibidos = indicadores_disponiveis
    else:
        indicadores_exibidos = [indicador_selecionado]

    st.markdown("---")
    st.markdown("### Painel de projeções")

    st.caption(
        f"Reservatório: {sistema} | Horizonte: 24 meses | "
        f"Indicadores: {len(indicadores_exibidos)}"
    )

    resultados_gerais = []
    analises_gerais = []

    for indicador in indicadores_exibidos:

        nome = INDICADORES[indicador]["nome"]
        unidade = INDICADORES[indicador]["unidade"]
        serie_original = dados[sistema][indicador].dropna()

        if len(serie_original) < 12:
            st.warning(
                f"Dados insuficientes para projetar {nome.lower()}."
            )
            continue

        try:
            with st.spinner(
                f"Calculando projeção de {nome.lower()}..."
            ):
                (
                    serie,
                    datas,
                    previsao,
                    inferior,
                    superior,
                    melhor,
                    avaliacao
                ) = gerar_previsao(
                    tuple(serie_original.values),
                    tuple(serie_original.index.astype(str)),
                    indicador
                )

        except Exception as erro:
            st.error(
                f"Falha ao projetar {nome.lower()}: {erro}"
            )
            continue

        st.markdown("---")
        st.markdown(f"## {nome}")

        st.success(
            f"Modelo selecionado para {nome.lower()}: {melhor}"
        )

        st.plotly_chart(
            grafico_previsao(
                serie,
                datas,
                previsao,
                inferior,
                superior,
                indicador,
                sistema
            ),
            use_container_width=True,
            key=f"grafico_previsao_{sistema}_{indicador}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            card_metrica(
                "Último valor observado",
                formatar_valor(serie.iloc[-1], unidade),
                f"Referência: {serie.index[-1]:%m/%Y}",
                AZUL
            )

        with c2:
            card_metrica(
                "Projeção em 12 meses",
                formatar_valor(previsao[11], unidade),
                f"Data: {datas[11]:%m/%Y}",
                "#E9A23B"
            )

        with c3:
            card_metrica(
                "Projeção em 24 meses",
                formatar_valor(previsao[23], unidade),
                f"Data: {datas[23]:%m/%Y}",
                "#10A77B"
            )

        variacao_12 = previsao[11] - serie.iloc[-1]
        variacao_24 = previsao[23] - serie.iloc[-1]

        percentual = (
            (variacao_24 / abs(serie.iloc[-1])) * 100
            if serie.iloc[-1] != 0
            else np.nan
        )

        if variacao_24 > 0:
            tendencia = "crescimento"
        elif variacao_24 < 0:
            tendencia = "redução"
        else:
            tendencia = "estabilidade"

        texto = f"""
        **Análise de {nome.lower()} — {sistema}**

        O último registro histórico apresenta
        **{serie.iloc[-1]:.2f} {unidade}**.

        Para os próximos 12 meses, o modelo estima
        **{previsao[11]:.2f} {unidade}**, com variação de
        **{variacao_12:+.2f} {unidade}** em relação ao último
        valor observado.

        No horizonte de 24 meses, a estimativa é de
        **{previsao[23]:.2f} {unidade}**. A variação acumulada
        projetada é de **{variacao_24:+.2f} {unidade}**.

        Isso representa uma tendência de **{tendencia}**.
        A variação percentual estimada no horizonte é
        **{percentual:.2f}%** quando o valor de referência
        é diferente de zero.

        O modelo selecionado foi **{melhor}**, com escolha
        baseada no desempenho obtido durante a validação temporal.
        """

        texto_analise(texto)

        analises_gerais.append({
            "Indicador": nome,
            "Último observado": serie.iloc[-1],
            "Previsão 12 meses": previsao[11],
            "Previsão 24 meses": previsao[23],
            "Variação em 24 meses": variacao_24,
            "Modelo": melhor,
            "Unidade": unidade
        })

        with st.expander(
            f"Ver validação estatística — {nome}"
        ):
            st.dataframe(
                avaliacao.style.format({
                    "MAE": "{:.3f}",
                    "RMSE": "{:.3f}",
                    "Viés": "{:.3f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "MAE representa o erro absoluto médio; RMSE "
                "penaliza mais os erros maiores; viés indica "
                "a tendência de superestimar ou subestimar."
            )

        tabela = pd.DataFrame({
            "Data": datas,
            "Indicador": nome,
            "Previsão": previsao,
            "Limite inferior": inferior,
            "Limite superior": superior,
            "Unidade": unidade,
            "Modelo": melhor
        })

        resultados_gerais.append(tabela)

        with st.expander(
            f"Consultar os 24 meses projetados — {nome}"
        ):
            st.dataframe(
                tabela.style.format({
                    "Previsão": "{:.2f}",
                    "Limite inferior": "{:.2f}",
                    "Limite superior": "{:.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )

    if analises_gerais:

        st.markdown("---")
        titulo("Resumo consolidado das projeções")

        resumo = pd.DataFrame(analises_gerais)

        st.dataframe(
            resumo.style.format({
                "Último observado": "{:.2f}",
                "Previsão 12 meses": "{:.2f}",
                "Previsão 24 meses": "{:.2f}",
                "Variação em 24 meses": "{:+.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Interpretação conjunta")

        textos = []

        for resultado in analises_gerais:

            direcao = (
                "aumento"
                if resultado["Variação em 24 meses"] > 0
                else (
                    "redução"
                    if resultado["Variação em 24 meses"] < 0
                    else "estabilidade"
                )
            )

            textos.append(
                f"- **{resultado['Indicador']}**: "
                f"projeção de {direcao}, saindo de "
                f"{resultado['Último observado']:.2f} "
                f"para {resultado['Previsão 24 meses']:.2f} "
                f"{resultado['Unidade']} em 24 meses."
            )

        st.markdown("\n".join(textos))

        st.markdown("""
        A leitura conjunta permite observar se os indicadores
        apresentam movimentos convergentes ou divergentes.
        Uma elevação da vazão natural, por exemplo, não implica
        automaticamente aumento do volume armazenado, pois o
        armazenamento também depende das retiradas, perdas,
        transferências e decisões operacionais.
        """)

        if resultados_gerais:

            tabela_consolidada = pd.concat(
                resultados_gerais,
                ignore_index=True
            )

            st.download_button(
                "Baixar todas as previsões em CSV",
                tabela_consolidada.to_csv(
                    index=False,
                    sep=";",
                    decimal=",",
                    encoding="utf-8-sig"
                ),
                file_name=(
                    f"blue_essence_previsoes_"
                    f"{normalizar(sistema)}.csv"
                ),
                mime="text/csv",
                use_container_width=True
            )

    st.info(
        "As projeções representam estimativas estatísticas "
        "condicionadas ao histórico. As faixas são aproximações "
        "de incerteza, não garantias. Os modelos não incorporam "
        "previsões futuras de chuva, evaporação, captação, "
        "transferências ou regras operacionais."
    )


# ============================================================
# 14. ANÁLISE CIENTÍFICA
# ============================================================

elif pagina == "Análise científica":

    titulo(
        "Análise científica",
        "Investigação de tendências, sazonalidade, correlações "
        "e comportamento dos indicadores."
    )

    # --------------------------------------------------------
    # Correlação entre reservatórios
    # --------------------------------------------------------

    st.markdown("### 1. Correlação entre os reservatórios")

    matriz = pd.DataFrame({
        sistema: df["volume_util"]
        for sistema, df in dados.items()
    })

    correlacao = matriz.corr()

    fig = px.imshow(
        correlacao,
        text_auto=".2f",
        color_continuous_scale="Blues",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlação de Pearson entre volumes úteis"
    )

    fig.update_layout(
        height=450,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "A correlação mede associação linear entre séries. "
        "Não demonstra causalidade nem influência operacional direta."
    )

    # --------------------------------------------------------
    # Sazonalidade
    # --------------------------------------------------------

    st.markdown("### 2. Comportamento sazonal")

    nomes_meses = [
        "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
        "Jul", "Ago", "Set", "Out", "Nov", "Dez"
    ]

    sazonal = matriz.copy()
    sazonal["Mês"] = sazonal.index.month

    media_mensal = sazonal.groupby("Mês").mean(numeric_only=True)

    fig = go.Figure()

    for sistema in dados:
        if sistema in media_mensal.columns:
            fig.add_trace(go.Scatter(
                x=nomes_meses,
                y=media_mensal[sistema],
                name=sistema,
                mode="lines+markers",
                line=dict(color=CORES[sistema], width=3)
            ))

    fig.update_layout(
        title="Média histórica do volume útil por mês",
        yaxis_title="Volume útil médio (%)"
    )

    st.plotly_chart(
        configurar_grafico(fig),
        use_container_width=True
    )

    st.markdown("""
    A análise sazonal permite observar como os níveis médios de
    armazenamento se distribuem ao longo dos meses do ano.
    Padrões recorrentes podem estar associados ao regime de chuvas,
    à demanda e às decisões operacionais. A série, isoladamente,
    não permite atribuir uma causa específica a essas variações.
    """)

    # --------------------------------------------------------
    # Médias móveis
    # --------------------------------------------------------

    st.markdown("### 3. Tendência e médias móveis")

    fig = go.Figure()
    legenda_media_movel_exibida = False

    for sistema, df in dados.items():
        serie = df["volume_util"].dropna()

        fig.add_trace(go.Scatter(
            x=serie.index,
            y=serie,
            name=sistema,
            line=dict(color=CORES[sistema], width=1.5)
        ))

        if len(serie) >= 12:
            media_movel = serie.rolling(12).mean()

            fig.add_trace(go.Scatter(
                x=media_movel.index,
                y=media_movel,
                name="Média móvel (12 meses)",
                showlegend=not legenda_media_movel_exibida,
                line=dict(color=CORES[sistema], width=3, dash="dash"),
                hovertemplate=(
                    f"{sistema} — média móvel: "
                    "%{y:.2f}%<extra></extra>"
                )
            ))
            legenda_media_movel_exibida = True

    adicionar_previsoes(
        fig,
        {
            sistema: df["volume_util"]
            for sistema, df in dados.items()
        },
        "volume_util"
    )

    fig.update_layout(
        title="Volume útil e média móvel",
        yaxis_title="Volume útil (%)"
    )

    st.plotly_chart(
        configurar_grafico(fig, 470),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Variação mensal
    # --------------------------------------------------------

    st.markdown("### 4. Variação mensal do volume útil")

    variacoes = matriz.diff()
    fig = go.Figure()

    for sistema in dados:
        fig.add_trace(go.Scatter(
            x=variacoes.index,
            y=variacoes[sistema],
            name=sistema,
            line=dict(color=CORES[sistema], width=2)
        ))

    adicionar_previsoes(
        fig,
        {
            sistema: df["volume_util"]
            for sistema, df in dados.items()
        },
        "volume_util",
        nome_indicador="variação mensal do volume útil",
        diferenciar=True
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#8095A5"
    )

    fig.update_layout(
        title="Variação mensal do volume útil",
        yaxis_title="Variação (pontos percentuais)"
    )

    st.plotly_chart(
        configurar_grafico(fig),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Distribuição
    # --------------------------------------------------------

    st.markdown("### 5. Distribuição dos níveis de armazenamento")

    fig = go.Figure()

    for sistema, df in dados.items():
        fig.add_trace(go.Box(
            y=df["volume_util"].dropna(),
            name=sistema,
            marker_color=CORES[sistema],
            boxmean=True
        ))

    fig.update_layout(
        title="Distribuição histórica do volume útil",
        yaxis_title="Volume útil (%)"
    )

    st.plotly_chart(
        configurar_grafico(fig),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Relação entre volume e vazões
    # --------------------------------------------------------

    st.markdown("### 6. Relação entre volume útil e vazões")

    sistema = st.selectbox(
        "Reservatório para análise de vazões",
        list(dados.keys()),
        key="analise_vazao"
    )

    df = dados[sistema]

    for indicador, nome in [
        ("vazao_natural", "Vazão natural"),
        ("vazao_captada", "Vazão captada")
    ]:
        if indicador not in df.columns:
            continue

        base = df[["volume_util", indicador]].dropna()

        if len(base) < 3:
            continue

        correlacao_vazao = base["volume_util"].corr(
            base[indicador]
        )

        fig = px.scatter(
            base,
            x=indicador,
            y="volume_util",
            trendline="ols",
            title=f"Volume útil × {nome} — {sistema}",
            labels={
                indicador: f"{nome} (m³/s)",
                "volume_util": "Volume útil (%)"
            }
        )

        fig.update_traces(
            marker=dict(color=CORES[sistema], size=9)
        )

        st.plotly_chart(
            configurar_grafico(fig),
            use_container_width=True
        )

        if abs(correlacao_vazao) >= 0.7:
            intensidade = "forte"
        elif abs(correlacao_vazao) >= 0.4:
            intensidade = "moderada"
        else:
            intensidade = "fraca"

        sentido = (
            "positiva"
            if correlacao_vazao > 0
            else "negativa"
        )

        st.markdown(
            f"A correlação entre volume útil e {nome.lower()} "
            f"é **{correlacao_vazao:.2f}**, classificada como "
            f"associação linear {intensidade} e {sentido}. "
            "Esse resultado não deve ser interpretado como causalidade."
        )

    # --------------------------------------------------------
    # Estatísticas descritivas
    # --------------------------------------------------------

    st.markdown("### 7. Estatísticas descritivas")

    estatisticas = matriz.describe().T

    estatisticas.columns = [
        "Quantidade",
        "Média",
        "Desvio padrão",
        "Mínimo",
        "25%",
        "Mediana",
        "75%",
        "Máximo"
    ]

    st.dataframe(
        estatisticas.style.format("{:.2f}"),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Texto de interpretação científica
    # --------------------------------------------------------

    st.markdown("### 8. Interpretação dos resultados")

    for sistema, df in dados.items():
        with st.expander(f"Conclusões — {sistema}"):
            texto_analise(analise_reservatorio(df, sistema))

    texto_analise(gerar_analise_comparativa(dados))

    st.markdown("""
    **Limitações metodológicas**

    As conclusões dependem da qualidade e da extensão das séries
    históricas disponíveis. Dados ausentes, mudanças de medição,
    alterações operacionais e eventos climáticos extremos podem
    afetar os resultados.

    Os modelos preditivos são baseados em padrões históricos e não
    substituem modelos hidrológicos completos. Para ampliar a
    capacidade explicativa e preditiva, recomenda-se incorporar
    precipitação, evaporação, afluências, retiradas, transferências
    entre sistemas e regras de operação.
    """)


# ============================================================
# 15. RODAPÉ
# ============================================================

st.markdown("""
<br><br>
<div style="
    background:#E2F3FC;
    border-radius:18px;
    padding:25px;
    text-align:center;
    border:1px solid #CBE9F7;
">
    <div style="font-size:25px;">💧</div>
    <div style="font-size:20px;font-weight:900;color:#147EBA;">
        Blue Essence
    </div>
    <div style="font-size:12px;color:#6286A0;">
        CIÊNCIA A FAVOR DA ÁGUA
    </div>
    <br>
    <div style="font-size:11px;color:#7896AA;">
        Plataforma de monitoramento e inteligência hídrica.
        <br>
        Desenvolvida com Python, Streamlit, Pandas e Plotly.
    </div>
</div>
""", unsafe_allow_html=True)
