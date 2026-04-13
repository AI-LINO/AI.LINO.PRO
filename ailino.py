import streamlit as st
import yfinance as yf
import numpy as np
from hmmlearn import hmm
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI.Lino Pro", page_icon="🤖", layout="wide")

st.markdown("""
<div style='text-align: center; background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); 
     padding: 30px; border-radius: 15px; margin-bottom: 20px;'>
    <h1 style='color: #00ff88; font-size: 3em; font-family: Arial Black;'>🤖 AI.LINO PRO</h1>
    <h3 style='color: #ffffff;'>Motor de Rebote — HMM + RSI + Volumen + Niveles Exactos</h3>
    <p style='color: #aaaaaa;'>Detecta agotamiento de vendedores · Entrada precisa · Stop Loss automatico</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BUSQUEDA
# ─────────────────────────────────────────────
def buscar_sugerencias(texto):
    try:
        resultados = yf.Search(texto, max_results=8)
        sugerencias = []
        for q in resultados.quotes:
            if q.get("quoteType") in ["EQUITY", "ETF", "CRYPTOCURRENCY"]:
                ticker = q.get("symbol", "")
                nombre = q.get("longname") or q.get("shortname", ticker)
                tipo   = q.get("quoteType", "")
                if ticker.endswith(".MX"):
                    pais = "MX"
                elif any(ticker.endswith(x) for x in [".PA",".DE",".AS",".SW",".MC",".MI",".L"]):
                    pais = "EU"
                elif tipo == "CRYPTOCURRENCY":
                    pais = "CRYPTO"
                else:
                    pais = "USA"
                sugerencias.append({
                    "label": f"[{pais}] {nombre} ({ticker})",
                    "ticker": ticker,
                    "nombre": nombre,
                    "pais": pais
                })
        return sugerencias
    except:
        return []

def obtener_precio_realtime(ticker):
    try:
        fi     = yf.Ticker(ticker).fast_info
        precio = fi.last_price
        prev   = fi.previous_close
        if precio and prev:
            return precio, ((precio - prev) / prev) * 100
    except:
        pass
    return None, None

# ─────────────────────────────────────────────
# INDICADORES
# ─────────────────────────────────────────────
def calcular_rsi(precios, periodo=14):
    delta    = precios.diff()
    ganancia = delta.where(delta > 0, 0).rolling(periodo).mean()
    perdida  = (-delta.where(delta < 0, 0)).rolling(periodo).mean()
    rs       = ganancia / perdida
    return 100 - (100 / (1 + rs))

def calcular_stoch_rsi(precios, periodo=14, smooth=3):
    rsi     = calcular_rsi(precios, periodo)
    rsi_min = rsi.rolling(periodo).min()
    rsi_max = rsi.rolling(periodo).max()
    stoch   = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    k       = stoch.rolling(smooth).mean()
    d       = k.rolling(smooth).mean()
    return k, d

def calcular_macd(precios):
    ema12      = precios.ewm(span=12).mean()
    ema26      = precios.ewm(span=26).mean()
    macd       = ema12 - ema26
    signal     = macd.ewm(span=9).mean()
    histograma = macd - signal
    return macd, signal, histograma

def calcular_bollinger(precios, periodo=20):
    media = precios.rolling(periodo).mean()
    std   = precios.rolling(periodo).std()
    return media + 2*std, media, media - 2*std

# ─────────────────────────────────────────────
# ██████████████████████████████████████████████
# MÓDULO NUEVO: DETECTOR DE PISO INTRADÍA
# ██████████████████████████████████████████████
# ─────────────────────────────────────────────
def detectar_piso_intraday(ticker):
    """
    Detecta el piso intradía en tiempo real usando velas de 5 minutos.
    
    Lógica:
    1. Caída de precio + caída de volumen simultánea = vendedores agotándose
    2. Vela de rechazo (mecha inferior larga) en zona de soporte
    3. Stoch RSI en sobreventa extrema en timeframe corto
    
    Retorna: dict con señal, precio_piso, stop, objetivo, confianza
    """
    try:
        t       = yf.Ticker(ticker)
        # Velas de 5 minutos — últimas 2 sesiones
        df_5m   = t.history(period="2d", interval="5m")
        # Velas de 1 hora — última semana para niveles
        df_1h   = t.history(period="5d", interval="1h")

        if df_5m.empty or len(df_5m) < 20:
            return {"error": "Sin datos intradía suficientes"}

        close_5m  = df_5m['Close'].squeeze()
        volume_5m = df_5m['Volume'].squeeze()
        high_5m   = df_5m['High'].squeeze()
        low_5m    = df_5m['Low'].squeeze()
        open_5m   = df_5m['Open'].squeeze()

        precio_actual = float(close_5m.iloc[-1])
        puntos        = 0
        señales       = []
        alertas_id    = []

        # ── CONDICIÓN 1: Caída de precio + caída de volumen ──
        # Buscar secuencia de 3+ velas bajistas con volumen decreciente
        bajistas_consecutivas = 0
        vol_decreciente       = True
        for i in range(-6, -1):
            if float(close_5m.iloc[i]) < float(close_5m.iloc[i-1]):
                bajistas_consecutivas += 1
                if i < -2 and float(volume_5m.iloc[i]) > float(volume_5m.iloc[i-1]):
                    vol_decreciente = False
            else:
                bajistas_consecutivas = 0
                vol_decreciente       = True

        if bajistas_consecutivas >= 3 and vol_decreciente:
            puntos += 3
            señales.append(f"🔴 Caída {bajistas_consecutivas} velas con volumen decreciente — vendedores agotándose")
        elif bajistas_consecutivas >= 2:
            puntos += 1
            señales.append(f"⚠️ Presión bajista ({bajistas_consecutivas} velas) — vigilar volumen")

        # Caída de volumen general en últimas 5 velas vs 10 anteriores
        vol_rec = float(volume_5m.iloc[-5:].mean())
        vol_ant = float(volume_5m.iloc[-15:-5].mean())
        if vol_ant > 0 and vol_rec < vol_ant * 0.70:
            reduccion_vol = ((vol_ant - vol_rec) / vol_ant) * 100
            puntos += 2
            señales.append(f"📉 Volumen cayó {reduccion_vol:.0f}% en últimas 5 velas — vendedores secándose")

        # ── CONDICIÓN 2: Vela de rechazo (mecha inferior larga) ──
        rechazos_5m = 0
        for i in range(-4, 0):
            o = float(open_5m.iloc[i])
            h = float(high_5m.iloc[i])
            l = float(low_5m.iloc[i])
            c = float(close_5m.iloc[i])
            cuerpo    = abs(c - o) + 1e-10
            mecha_inf = min(o, c) - l
            rango     = h - l + 1e-10
            # Hammer o Pin Bar: mecha inferior > 2x cuerpo y ocupa >35% del rango
            if mecha_inf > cuerpo * 1.8 and mecha_inf / rango > 0.35:
                rechazos_5m += 1
            # Doji (indecisión)
            elif cuerpo / rango < 0.15 and rango > 0:
                rechazos_5m += 0.5

        if rechazos_5m >= 2:
            puntos += 3
            señales.append(f"🕯️ {int(rechazos_5m)} velas de rechazo en mínimos — compradores defendiendo")
        elif rechazos_5m >= 1:
            puntos += 2
            señales.append("🕯️ Vela de rechazo detectada — posible piso")

        # ── CONDICIÓN 3: Stoch RSI en sobreventa intradía ──
        if len(close_5m) >= 20:
            stoch_k_5m, stoch_d_5m = calcular_stoch_rsi(close_5m, periodo=14, smooth=3)
            sk_actual = float(stoch_k_5m.iloc[-1])
            sd_actual = float(stoch_d_5m.iloc[-1])

            if sk_actual < 10:
                puntos += 3
                señales.append(f"📊 Stoch RSI intradía extremo ({sk_actual:.1f}) — rebote inminente")
            elif sk_actual < 20:
                puntos += 2
                señales.append(f"📊 Stoch RSI intradía en sobreventa ({sk_actual:.1f}) — zona de rebote")
            elif sk_actual < 30:
                puntos += 1
                señales.append(f"📊 Stoch RSI bajando ({sk_actual:.1f}) — acercándose a zona rebote")

            if sk_actual > sd_actual and sk_actual < 40:
                puntos += 1
                señales.append("📈 Stoch RSI cruzando al alza — impulso iniciando")
        else:
            sk_actual = 50
            sd_actual = 50

        # ── CONDICIÓN 4: Precio cerca de soporte horario ──
        soporte_1h    = float(df_1h['Low'].squeeze().iloc[-20:].min()) if not df_1h.empty else None
        resistencia_1h = float(df_1h['High'].squeeze().iloc[-20:].max()) if not df_1h.empty else None
        min_dia       = float(low_5m.min())
        max_dia       = float(high_5m.max())

        # Precio dentro del 1% del mínimo del día = zona de soporte
        if precio_actual <= min_dia * 1.01:
            puntos += 2
            señales.append(f"🎯 Precio cerca del mínimo del día (${min_dia:.2f}) — soporte intradía")

        # Precio cerca del soporte horario
        if soporte_1h and precio_actual <= soporte_1h * 1.015:
            puntos += 2
            señales.append(f"🎯 Precio en soporte horario (${soporte_1h:.2f}) — nivel clave")

        # ── CONDICIÓN 5: Compresión de rangos ──
        # Velas cada vez más pequeñas = la caída pierde momentum
        rangos_rec = [float(high_5m.iloc[i]) - float(low_5m.iloc[i]) for i in range(-4, 0)]
        rangos_ant = [float(high_5m.iloc[i]) - float(low_5m.iloc[i]) for i in range(-8, -4)]
        if rangos_ant and sum(rangos_ant) > 0:
            rango_rec_avg = sum(rangos_rec) / len(rangos_rec)
            rango_ant_avg = sum(rangos_ant) / len(rangos_ant)
            if rango_rec_avg < rango_ant_avg * 0.75:
                compresion = ((rango_ant_avg - rango_rec_avg) / rango_ant_avg) * 100
                puntos += 2
                señales.append(f"🔇 Rango comprimido {compresion:.0f}% — la caída pierde velocidad")

        # ── NIVELES DE OPERACIÓN INTRADÍA ──
        atr_5m    = float((high_5m - low_5m).rolling(14).mean().iloc[-1])
        entrada   = round(precio_actual, 2)
        stop_loss = round(min_dia - atr_5m * 0.5, 2)
        objetivo1 = round(precio_actual + atr_5m * 2, 2)
        objetivo2 = round(precio_actual + atr_5m * 4, 2)
        riesgo    = round(((precio_actual - stop_loss) / precio_actual) * 100, 2)
        ganancia  = round(((objetivo1 - precio_actual) / precio_actual) * 100, 2)
        rr        = round(ganancia / riesgo, 1) if riesgo > 0 else 0

        # ── ALERTAS ──
        rsi_5m = calcular_rsi(close_5m).iloc[-1]
        if rsi_5m > 60:
            alertas_id.append(f"⚠️ RSI intradía elevado ({rsi_5m:.1f}) — no es zona de compra")
        if bajistas_consecutivas == 0:
            alertas_id.append("ℹ️ No hay tendencia bajista activa — esperar corrección")

        # ── DECISIÓN FINAL ──
        if puntos >= 10:
            nivel     = "FUERTE"
            color     = "#00ff88"
            señal_txt = "🟢 POSIBLE PISO — ENTRAR AHORA"
            confianza = min(95, 60 + puntos * 2)
        elif puntos >= 7:
            nivel     = "MODERADO"
            color     = "#FFD700"
            señal_txt = "🟡 PISO PROBABLE — PREPARARSE"
            confianza = min(75, 45 + puntos * 2)
        elif puntos >= 4:
            nivel     = "DEBIL"
            color     = "#FF8C00"
            señal_txt = "🟠 PRIMERAS SEÑALES — VIGILAR"
            confianza = min(55, 30 + puntos * 2)
        else:
            nivel     = "SIN SEÑAL"
            color     = "#ff4444"
            señal_txt = "🔴 SIN PISO DETECTADO — ESPERAR"
            confianza = max(10, puntos * 5)

        return {
            "señal":         señal_txt,
            "nivel":         nivel,
            "color":         color,
            "puntos":        puntos,
            "confianza":     confianza,
            "precio_actual": precio_actual,
            "min_dia":       min_dia,
            "max_dia":       max_dia,
            "soporte_1h":    soporte_1h,
            "resistencia_1h": resistencia_1h,
            "entrada":       entrada,
            "stop_loss":     stop_loss,
            "objetivo1":     objetivo1,
            "objetivo2":     objetivo2,
            "riesgo_pct":    riesgo,
            "ganancia_pct":  ganancia,
            "rr":            rr,
            "atr_5m":        round(atr_5m, 4),
            "stoch_k_5m":    round(sk_actual, 1),
            "rsi_5m":        round(float(rsi_5m), 1),
            "señales":       señales,
            "alertas":       alertas_id,
            "df_5m":         df_5m,
            "df_1h":         df_1h,
        }

    except Exception as e:
        return {"error": str(e)}


def grafica_intraday(df_5m, df_1h, niveles_id, ticker, nombre):
    """Gráfica intradía de 5 minutos con señales de piso."""
    if df_5m is None or df_5m.empty or len(df_5m) < 10:
        return None

    close_5m           = df_5m['Close'].squeeze()
    stoch_k, stoch_d   = calcular_stoch_rsi(close_5m)
    vol_ma             = df_5m['Volume'].squeeze().rolling(20).mean()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.55, 0.20, 0.25],
        subplot_titles=(
            f"Intradía 5min — {nombre} ({ticker})",
            "Volumen 5min",
            "Stoch RSI 5min — Señal de Piso",
        )
    )

    # Velas 5 min
    fig.add_trace(go.Candlestick(
        x=df_5m.index,
        open=df_5m['Open'], high=df_5m['High'],
        low=df_5m['Low'],   close=df_5m['Close'],
        name="5min",
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444',
        increasing_fillcolor='#00ff88',  decreasing_fillcolor='#ff4444',
    ), row=1, col=1)

    # Niveles horizontales
    px_ini = df_5m.index[max(0, len(df_5m)-50)]
    px_fin = df_5m.index[-1]
    for y_val, color, etq in [
        (niveles_id.get('entrada'),    '#00ff88', f"Entrada ${niveles_id.get('entrada', 0):.2f}"),
        (niveles_id.get('stop_loss'),  '#ff4444', f"Stop ${niveles_id.get('stop_loss', 0):.2f}"),
        (niveles_id.get('objetivo1'),  '#FFD700', f"T1 ${niveles_id.get('objetivo1', 0):.2f}"),
        (niveles_id.get('objetivo2'),  '#00BFFF', f"T2 ${niveles_id.get('objetivo2', 0):.2f}"),
        (niveles_id.get('min_dia'),    '#FF8C00', f"Mín día ${niveles_id.get('min_dia', 0):.2f}"),
    ]:
        if y_val:
            fig.add_shape(type="line",
                x0=px_ini, x1=px_fin, y0=y_val, y1=y_val,
                line=dict(color=color, width=1.5, dash="dash"), row=1, col=1)
            fig.add_annotation(x=px_fin, y=y_val, text=etq,
                showarrow=False, xanchor="right",
                font=dict(color=color, size=10),
                bgcolor="rgba(0,0,0,0.6)", row=1, col=1)

    # Soporte 1h
    if niveles_id.get('soporte_1h'):
        fig.add_shape(type="line",
            x0=px_ini, x1=px_fin,
            y0=niveles_id['soporte_1h'], y1=niveles_id['soporte_1h'],
            line=dict(color='#9B59B6', width=2, dash="dot"), row=1, col=1)
        fig.add_annotation(
            x=px_ini, y=niveles_id['soporte_1h'],
            text=f"Soporte 1h ${niveles_id['soporte_1h']:.2f}",
            showarrow=False, xanchor="left",
            font=dict(color='#9B59B6', size=9),
            bgcolor="rgba(0,0,0,0.5)", row=1, col=1)

    # Volumen
    colores_vol = ['#00ff88' if c >= o else '#ff4444'
                   for c, o in zip(df_5m['Close'], df_5m['Open'])]
    fig.add_trace(go.Bar(x=df_5m.index, y=df_5m['Volume'],
        name="Volumen", marker_color=colores_vol, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_5m.index, y=vol_ma,
        name="Vol MA20", line=dict(color='#888888', width=1, dash='dot')), row=2, col=1)

    # Stoch RSI
    fig.add_trace(go.Scatter(x=df_5m.index, y=stoch_k,
        name="Stoch K", line=dict(color='#00ff88', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_5m.index, y=stoch_d,
        name="Stoch D", line=dict(color='#ff4444', width=1.5, dash='dot')), row=3, col=1)
    fig.add_hrect(y0=0,  y1=20,  fillcolor="#00ff88", opacity=0.10, line_width=0, row=3, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="#ff4444", opacity=0.08, line_width=0, row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="#00ff88", line_width=1, row=3, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="#ff4444", line_width=1, row=3, col=1)

    fig.update_layout(
        height=700,
        paper_bgcolor='#0f0c29', plot_bgcolor='#0f0c29',
        font=dict(color='#ffffff', size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1,
                    bgcolor='rgba(0,0,0,0.4)', font=dict(size=9)),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor='#1a1a2e', zeroline=False,
                         showspikes=True, spikecolor="#00ff88",
                         spikethickness=1, spikemode="across", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='#1a1a2e', zeroline=False,
                         tickfont=dict(size=9), row=i, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig

# ─────────────────────────────────────────────
# MOTOR DE AGOTAMIENTO DE VENDEDORES (5 CAPAS)
# ─────────────────────────────────────────────
def detectar_agotamiento_vendedores(df):
    if len(df) < 15:
        return False, 0, "Sin datos suficientes", []

    close  = df['Close'].squeeze()
    volume = df['Volume'].squeeze()
    high   = df['High'].squeeze()
    low    = df['Low'].squeeze()

    puntos   = 0
    detalles = []

    for v in [3, 5, 8]:
        if len(df) >= v * 2:
            p_rec   = close.iloc[-v:].mean()
            p_ant   = close.iloc[-v*2:-v].mean()
            vol_rec = volume.iloc[-v:].mean()
            vol_ant = volume.iloc[-v*2:-v].mean()
            if p_rec < p_ant and vol_rec < vol_ant:
                reduccion = ((vol_ant - vol_rec) / vol_ant) * 100
                puntos += 1
                detalles.append(f"Volumen cae {reduccion:.0f}% mientras precio baja (ventana {v}d)")

    rechazos = 0
    for i in range(-5, 0):
        o = float(df['Open'].iloc[i])
        h = float(df['High'].iloc[i])
        l = float(df['Low'].iloc[i])
        c = float(df['Close'].iloc[i])
        cuerpo    = abs(c - o) + 1e-10
        mecha_inf = min(o, c) - l
        rango     = h - l + 1e-10
        if mecha_inf > cuerpo * 1.5 and mecha_inf / rango > 0.35:
            rechazos += 1
    if rechazos >= 2:
        puntos += 2
        detalles.append(f"{rechazos} velas con rechazo de minimos — compradores absorbiendo presion")
    elif rechazos == 1:
        puntos += 1
        detalles.append("1 vela con rechazo de minimos detectada")

    vols_bajas = []
    for i in range(-8, 0):
        if float(close.iloc[i]) < float(df['Open'].iloc[i]):
            vols_bajas.append(float(volume.iloc[i]))
    if len(vols_bajas) >= 3:
        mitad        = len(vols_bajas) // 2
        vol_reciente = sum(vols_bajas[mitad:]) / max(len(vols_bajas[mitad:]), 1)
        vol_anterior = sum(vols_bajas[:mitad]) / max(mitad, 1)
        if vol_reciente < vol_anterior * 0.85:
            reduccion = ((vol_anterior - vol_reciente) / vol_anterior) * 100
            puntos += 2
            detalles.append(f"Dias bajistas con volumen {reduccion:.0f}% menor — vendedores agotandose")

    rangos_rec = [float(high.iloc[i]) - float(low.iloc[i]) for i in range(-5, 0)]
    rangos_ant = [float(high.iloc[i]) - float(low.iloc[i]) for i in range(-10, -5)]
    if rangos_ant and sum(rangos_ant) > 0:
        rango_rec = sum(rangos_rec) / len(rangos_rec)
        rango_ant = sum(rangos_ant) / len(rangos_ant)
        if rango_rec < rango_ant * 0.80:
            compresion = ((rango_ant - rango_rec) / rango_ant) * 100
            puntos += 1
            detalles.append(f"Rango diario comprimido {compresion:.0f}% — la caida pierde velocidad")

    min_rec = float(low.iloc[-5:].min())
    min_ant = float(low.iloc[-15:-5].min())
    if min_ant > 0:
        tolerancia = min_ant * 0.015
        if abs(min_rec - min_ant) <= tolerancia:
            puntos += 3
            detalles.append(f"Doble piso en ${min_rec:.2f} — soporte fuerte confirmado en 2 toques")

    if puntos >= 6:
        return True, 3, "AGOTAMIENTO FUERTE — Rebote de alta probabilidad", detalles
    elif puntos >= 4:
        return True, 2, "AGOTAMIENTO MODERADO — Señales positivas acumulandose", detalles
    elif puntos >= 2:
        return True, 1, "AGOTAMIENTO DEBIL — Primeras señales visibles", detalles
    else:
        return False, 0, "Sin agotamiento confirmado aun", detalles


def detectar_vela_rebote(df):
    senales = []
    if len(df) < 2:
        return senales
    for i in range(1, len(df)):
        o = float(df['Open'].iloc[i])
        h = float(df['High'].iloc[i])
        l = float(df['Low'].iloc[i])
        c = float(df['Close'].iloc[i])
        cuerpo      = abs(c - o)
        rango_total = h - l + 1e-10
        mecha_inf   = min(o, c) - l
        mecha_sup   = h - max(o, c)
        if mecha_inf >= 2 * cuerpo and mecha_sup <= cuerpo * 0.5 and cuerpo / rango_total < 0.4:
            senales.append((df.index[i], "Hammer", "#00ff88"))
        elif cuerpo / rango_total < 0.1:
            senales.append((df.index[i], "Doji", "#FFD700"))
        elif i > 0:
            o_prev = float(df['Open'].iloc[i-1])
            c_prev = float(df['Close'].iloc[i-1])
            if c_prev < o_prev and c > o and c > o_prev and o < c_prev:
                senales.append((df.index[i], "Engulfing Alcista", "#00BFFF"))
    return senales

def calcular_niveles_trading(df, precio_actual):
    close = df['Close'].squeeze()
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr              = tr.rolling(14).mean().iloc[-1]
    soporte_reciente = low.iloc[-10:].min()
    resistencia      = high.iloc[-10:].max()
    soporte_fuerte   = low.iloc[-30:].min()
    _, _, bb_low_s   = calcular_bollinger(close)
    entrada_ideal    = soporte_reciente * 1.005
    stop_loss        = soporte_reciente - atr
    objetivo_1       = precio_actual + atr * 1.5
    objetivo_2       = precio_actual + atr * 3.0
    objetivo_3       = resistencia * 0.995
    riesgo_pct       = ((precio_actual - stop_loss) / precio_actual) * 100
    pot_ganancia     = ((objetivo_2 - precio_actual) / precio_actual) * 100
    rr_ratio         = pot_ganancia / riesgo_pct if riesgo_pct > 0 else 0
    return {
        "entrada":        round(entrada_ideal, 4),
        "stop_loss":      round(stop_loss, 4),
        "objetivo_1":     round(objetivo_1, 4),
        "objetivo_2":     round(objetivo_2, 4),
        "objetivo_3":     round(objetivo_3, 4),
        "soporte":        round(soporte_reciente, 4),
        "resistencia":    round(resistencia, 4),
        "soporte_fuerte": round(soporte_fuerte, 4),
        "atr":            round(atr, 4),
        "riesgo_pct":     round(riesgo_pct, 2),
        "pot_ganancia":   round(pot_ganancia, 2),
        "rr_ratio":       round(rr_ratio, 2),
        "bb_low":         round(bb_low_s.iloc[-1], 4),
    }

def calcular_score_rebote(estado_hmm, rsi, stoch_k, stoch_d,
                           macd_val, signal_val, agot_nivel,
                           velas_rebote, precio_vs_bb_low):
    score   = 50
    razones = []
    alertas = []

    if estado_hmm == "ALCISTA":
        score += 15
        razones.append("HMM detecta regimen alcista")
    elif estado_hmm == "BAJISTA":
        score -= 10
        alertas.append("HMM bajista — rebote de corta duracion posible")
    else:
        razones.append("HMM lateral — posible acumulacion")

    if rsi < 25:
        score += 25
        razones.append(f"RSI extremo ({rsi:.1f}) — sobreventa severa")
    elif rsi < 35:
        score += 15
        razones.append(f"RSI en sobreventa ({rsi:.1f}) — zona de rebote")
    elif rsi > 70:
        score -= 20
        alertas.append(f"RSI sobrecomprado ({rsi:.1f}) — no entrar")
    elif rsi > 60:
        score -= 10
        alertas.append(f"RSI elevado ({rsi:.1f}) — cuidado")

    if stoch_k < 20 and stoch_d < 20:
        score += 20
        razones.append(f"Stoch RSI extremo ({stoch_k:.1f}) — rebote inminente")
    elif stoch_k > stoch_d and stoch_k < 40:
        score += 10
        razones.append(f"Stoch RSI cruzando alza ({stoch_k:.1f}) — impulso iniciando")
    elif stoch_k > 80:
        score -= 15
        alertas.append(f"Stoch RSI sobrecomprado ({stoch_k:.1f}) — salir pronto")

    if macd_val > signal_val:
        score += 10
        razones.append("MACD sobre senal — momentum positivo")
    else:
        score -= 5
        alertas.append("MACD bajo senal — presion vendedora activa")

    if agot_nivel == 3:
        score += 25
        razones.append("Agotamiento de vendedores FUERTE confirmado")
    elif agot_nivel == 2:
        score += 15
        razones.append("Agotamiento de vendedores MODERADO confirmado")
    elif agot_nivel == 1:
        score += 8
        razones.append("Primeras señales de agotamiento detectadas")
    else:
        alertas.append("Agotamiento de vendedores sin confirmar")

    if len(velas_rebote) > 0:
        score += 15
        razones.append(f"Patron de vela: {velas_rebote[-1][1]} detectado")

    if precio_vs_bb_low < 0:
        score += 15
        razones.append("Precio bajo BB inferior — zona de reversion estadistica")
    elif precio_vs_bb_low < 2:
        score += 5
        razones.append("Precio rozando BB inferior")

    return max(0, min(100, score)), razones, alertas

# ─────────────────────────────────────────────
# MOTOR HMM
# ─────────────────────────────────────────────
class MaquinaDineroLino:
    def __init__(self):
        self.modelo = hmm.GaussianHMM(
            n_components=3, covariance_type="full",
            n_iter=2000, random_state=42, tol=1e-5
        )

    def analizar(self, ticker):
        try:
            t  = yf.Ticker(ticker)
            df = t.history(period="2y", interval="1d")
            if df.empty or len(df) < 60:
                return None, "No hay suficientes datos historicos"

            close       = df["Close"].squeeze()
            retornos    = np.log(close / close.shift(1)).dropna()
            volatilidad = retornos.rolling(5).std().dropna()
            momentum    = close.pct_change(5).dropna()
            min_len     = min(len(retornos), len(volatilidad), len(momentum))
            X = np.column_stack([
                retornos.values[-min_len:],
                volatilidad.values[-min_len:],
                momentum.values[-min_len:]
            ])
            self.modelo.fit(X)
            _, states     = self.modelo.decode(X, algorithm="viterbi")
            means         = self.modelo.means_[:, 0]
            bull_state    = int(np.argmax(means))
            bear_state    = int(np.argmin(means))
            estado_actual = int(states[-1])
            if estado_actual == bull_state:
                estado_hmm = "ALCISTA"
            elif estado_actual == bear_state:
                estado_hmm = "BAJISTA"
            else:
                estado_hmm = "LATERAL"

            rsi_serie        = calcular_rsi(close)
            rsi              = rsi_serie.iloc[-1]
            stoch_k, stoch_d = calcular_stoch_rsi(close)
            sk, sd           = stoch_k.iloc[-1], stoch_d.iloc[-1]
            macd, signal, _  = calcular_macd(close)
            macd_val         = macd.iloc[-1]
            signal_val       = signal.iloc[-1]

            df_chart = t.history(period="3mo", interval="1d")

            agot_ok, agot_nivel, agot_desc, agot_detalles = detectar_agotamiento_vendedores(df_chart)
            velas_rev   = detectar_vela_rebote(df_chart.tail(10))

            _, _, bb_low_s = calcular_bollinger(close)
            bb_low_val     = bb_low_s.iloc[-1]

            precio_rt, cambio_rt = obtener_precio_realtime(ticker)
            if precio_rt:
                precio_actual = precio_rt
                cambio        = cambio_rt
            else:
                precio_actual = float(close.iloc[-1])
                cambio = ((precio_actual - float(close.iloc[-2])) / float(close.iloc[-2])) * 100

            precio_vs_bb = ((precio_actual - bb_low_val) / bb_low_val) * 100

            score, razones, alertas = calcular_score_rebote(
                estado_hmm, rsi, sk, sd,
                macd_val, signal_val,
                agot_nivel, velas_rev, precio_vs_bb
            )

            niveles = calcular_niveles_trading(df_chart, precio_actual)

            if score >= 70:
                senal = "ENTRAR — REBOTE PROBABLE"
                color = "#00ff88"
            elif score >= 55:
                senal = "PREPARARSE — REBOTE POSIBLE"
                color = "#FFD700"
            elif score <= 30:
                senal = "NO ENTRAR — TENDENCIA BAJISTA"
                color = "#ff4444"
            else:
                senal = "NEUTRO — SIN SENAL CLARA"
                color = "#FF8C00"

            senales_bt = []
            for i in range(30, len(close)):
                r            = calcular_rsi(close.iloc[:i+1]).iloc[-1]
                sk_bt, sd_bt = calcular_stoch_rsi(close.iloc[:i+1])
                sk_v         = sk_bt.iloc[-1]
                sd_v         = sd_bt.iloc[-1]
                m, s, _      = calcular_macd(close.iloc[:i+1])
                mv, sv       = m.iloc[-1], s.iloc[-1]
                sc, _, _     = calcular_score_rebote(
                    estado_hmm, r, sk_v, sd_v, mv, sv, 0, [], 0
                )
                senales_bt.append(sc)

            senales_series = pd.Series(senales_bt)
            compras = (senales_series >= 70).sum()
            ventas  = (senales_series <= 30).sum()
            total   = len(senales_series)

            return {
                "senal": senal, "score": score, "color": color,
                "estado_hmm": estado_hmm,
                "precio": precio_actual, "cambio": cambio,
                "rsi": round(rsi, 1),
                "stoch_k": round(sk, 1), "stoch_d": round(sd, 1),
                "macd": round(macd_val, 4), "signal_line": round(signal_val, 4),
                "agot_ok": agot_ok, "agot_nivel": agot_nivel,
                "agot_desc": agot_desc, "agot_detalles": agot_detalles,
                "velas_rev": velas_rev,
                "razones": razones, "alertas": alertas,
                "niveles": niveles,
                "compras_bt": compras, "ventas_bt": ventas, "total_bt": total,
                "close": close, "df_chart": df_chart, "datos": len(df)
            }, None

        except Exception as e:
            return None, str(e)

# ─────────────────────────────────────────────
# GRAFICA PROFESIONAL (SWING)
# ─────────────────────────────────────────────
def grafica_rebote_profesional(df, ticker, nombre, niveles, velas_rev):
    if df.empty or len(df) < 10:
        return None

    close              = df['Close'].squeeze()
    ema9               = close.ewm(span=9).mean()
    ema21              = close.ewm(span=21).mean()
    stoch_k, stoch_d   = calcular_stoch_rsi(close)
    macd, signal, hist = calcular_macd(close)
    bb_up, _, bb_low   = calcular_bollinger(close)

    colores_velas = ['#00ff88' if c >= o else '#ff4444'
                     for c, o in zip(df['Close'], df['Open'])]
    colores_hist  = ['#00ff88' if v >= 0 else '#ff4444' for v in hist]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.48, 0.14, 0.19, 0.19],
        subplot_titles=(
            f"Velas Diarias — {nombre} ({ticker})",
            "Volumen",
            "Stoch RSI — Senal de Rebote",
            "MACD"
        )
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'],   close=df['Close'],
        name="Precio",
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444',
        increasing_fillcolor='#00ff88',  decreasing_fillcolor='#ff4444',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=bb_up, name="BB Superior",
        line=dict(color='#555577', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_low, name="BB Inferior",
        line=dict(color='#335577', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(0,100,255,0.05)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema9,  name="EMA 9",
        line=dict(color='#FFD700', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema21, name="EMA 21",
        line=dict(color='#00BFFF', width=1.5)), row=1, col=1)

    primer_idx = df.index[max(0, len(df)-30)]
    ultimo_idx = df.index[-1]
    for y_val, color, etiqueta in [
        (niveles['entrada'],    '#00ff88', f"Entrada ${niveles['entrada']:.2f}"),
        (niveles['stop_loss'],  '#ff4444', f"Stop ${niveles['stop_loss']:.2f}"),
        (niveles['objetivo_1'], '#FFD700', f"T1 ${niveles['objetivo_1']:.2f}"),
        (niveles['objetivo_2'], '#00BFFF', f"T2 ${niveles['objetivo_2']:.2f}"),
    ]:
        fig.add_shape(type="line",
            x0=primer_idx, x1=ultimo_idx, y0=y_val, y1=y_val,
            line=dict(color=color, width=1.5, dash="dash"), row=1, col=1)
        fig.add_annotation(x=ultimo_idx, y=y_val, text=etiqueta,
            showarrow=False, xanchor="right",
            font=dict(color=color, size=10),
            bgcolor="rgba(0,0,0,0.5)", row=1, col=1)

    for fecha, tipo_vela, color_vela in velas_rev[-3:]:
        if fecha in df.index:
            precio_vela = float(df.loc[fecha, 'Low']) * 0.998
            fig.add_trace(go.Scatter(
                x=[fecha], y=[precio_vela],
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=14, color=color_vela),
                text=[tipo_vela], textposition="bottom center",
                textfont=dict(size=9, color=color_vela),
                name=tipo_vela, showlegend=False
            ), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
        name="Volumen", marker_color=colores_velas, opacity=0.7), row=2, col=1)
    vol_avg = df['Volume'].rolling(20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=vol_avg, name="Vol MA20",
        line=dict(color='#888888', width=1, dash='dot')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=stoch_k,
        name="Stoch K", line=dict(color='#00ff88', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=stoch_d,
        name="Stoch D", line=dict(color='#ff4444', width=1.5, dash='dot')), row=3, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="#ff4444", opacity=0.08, line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=20,  fillcolor="#00ff88", opacity=0.08, line_width=0, row=3, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="#ff4444", line_width=1, row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="#00ff88", line_width=1, row=3, col=1)

    fig.add_trace(go.Bar(x=df.index, y=hist,
        name="Histograma", marker_color=colores_hist, opacity=0.7), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd,   name="MACD",
        line=dict(color='#00BFFF', width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal, name="Signal",
        line=dict(color='#FF8C00', width=1.5)), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#888888", line_width=1, row=4, col=1)

    fig.update_layout(
        height=820,
        paper_bgcolor='#0f0c29', plot_bgcolor='#0f0c29',
        font=dict(color='#ffffff', size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1,
                    bgcolor='rgba(0,0,0,0.4)', font=dict(size=9)),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor='#1a1a2e', zeroline=False,
                         showspikes=True, spikecolor="#00ff88",
                         spikethickness=1, spikemode="across", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='#1a1a2e', zeroline=False,
                         tickfont=dict(size=9), row=i, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig

# ─────────────────────────────────────────────
# INTERFAZ
# ─────────────────────────────────────────────
if "ticker_sel" not in st.session_state: st.session_state.ticker_sel = ""
if "nombre_sel" not in st.session_state: st.session_state.nombre_sel = ""
if "pais_sel"   not in st.session_state: st.session_state.pais_sel   = ""

col_s, col_b = st.columns([4, 1])
with col_s:
    busqueda = st.text_input("Escribe empresa o ticker:",
        placeholder="Ej: Tesla, Apple, FEMSA, Cemex, Bitcoin...")
with col_b:
    st.write(""); st.write("")
    buscar_btn = st.button("ANALIZAR", use_container_width=True)

if busqueda and len(busqueda) >= 2 and not buscar_btn:
    with st.spinner("Buscando..."):
        sugs = buscar_sugerencias(busqueda)
    if sugs:
        opciones  = [s["label"] for s in sugs]
        seleccion = st.radio("Selecciona el activo:", opciones,
                             key="radio_sug", label_visibility="visible")
        idx = opciones.index(seleccion)
        st.session_state.ticker_sel = sugs[idx]["ticker"]
        st.session_state.nombre_sel = sugs[idx]["nombre"]
        st.session_state.pais_sel   = sugs[idx]["pais"]
        st.info(f"Seleccionado: **{st.session_state.nombre_sel}** "
                f"`{st.session_state.ticker_sel}` [{st.session_state.pais_sel}]")

col_r1, _ = st.columns([1, 5])
with col_r1:
    auto_refresh = st.checkbox("Auto-refresh 15 min")
if auto_refresh:
    import time; time.sleep(900); st.rerun()

ticker_a_usar = st.session_state.ticker_sel or busqueda.upper().strip()
nombre_a_usar = st.session_state.nombre_sel or busqueda

if buscar_btn and ticker_a_usar:
    with st.spinner("AI.Lino analizando..."):
        maquina          = MaquinaDineroLino()
        resultado, error = maquina.analizar(ticker_a_usar)

    if error:
        st.error(f"Error: {error}")
    elif resultado:

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #0f0c29, #302b63); 
             padding: 30px; border-radius: 15px; text-align: center;
             border: 3px solid {resultado["color"]}; margin: 15px 0;'>
            <h1 style='color: {resultado["color"]}; font-size: 2.8em;'>{resultado["senal"]}</h1>
            <h2 style='color: white;'>Score AI.LINO: {resultado["score"]}/100</h2>
            <h3 style='color: #aaaaaa;'>Precio en tiempo real:
            <span style='color:white; font-weight:bold'> ${resultado["precio"]:.2f}</span>
            <span style='color:{"#00ff88" if resultado["cambio"]>0 else "#ff4444"}'>
            ({resultado["cambio"]:+.2f}%)</span></h3>
        </div>
        """, unsafe_allow_html=True)

        # Niveles swing
        niv = resultado["niveles"]
        st.markdown("### Niveles de Trading")
        n1, n2, n3, n4, n5 = st.columns(5)
        n1.metric("Entrada Ideal",      f"${niv['entrada']:.2f}")
        n2.metric("Stop Loss",          f"${niv['stop_loss']:.2f}",
                  delta=f"-{niv['riesgo_pct']:.1f}%", delta_color="inverse")
        n3.metric("Objetivo 1",         f"${niv['objetivo_1']:.2f}")
        n4.metric("Objetivo 2",         f"${niv['objetivo_2']:.2f}",
                  delta=f"+{niv['pot_ganancia']:.1f}%")
        n5.metric("Riesgo / Beneficio", f"1 : {niv['rr_ratio']:.1f}")
        st.caption(f"ATR: ${niv['atr']:.2f} | Soporte: ${niv['soporte']:.2f} | "
                   f"Resistencia: ${niv['resistencia']:.2f} | BB Inf: ${niv['bb_low']:.2f}")

        # ══════════════════════════════════════════════════
        # MÓDULO PISO INTRADÍA — NUEVO
        # ══════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #0a0a1a, #1a1a3e);
             padding: 12px 20px; border-radius: 10px; border-left: 4px solid #FF8C00;
             margin-bottom: 10px;'>
            <h3 style='color: #FF8C00; margin: 0;'>⚡ MÓDULO PISO INTRADÍA — Velas 5 minutos</h3>
            <p style='color: #888; margin: 4px 0 0 0; font-size: 0.85em;'>
            Detecta el piso del día en tiempo real · Ideal para operaciones de rango intradía
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Analizando intradía (velas 5min)..."):
            id_result = detectar_piso_intraday(ticker_a_usar)

        if "error" in id_result:
            st.warning(f"Datos intradía no disponibles: {id_result['error']}")
        else:
            # Señal principal intradía
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #0f0c29, #1a1a3e);
                 padding: 25px; border-radius: 12px; text-align: center;
                 border: 3px solid {id_result["color"]}; margin: 10px 0;'>
                <h2 style='color: {id_result["color"]}; font-size: 2em; margin: 0;'>
                    {id_result["señal"]}
                </h2>
                <p style='color: #aaa; margin: 8px 0 0 0;'>
                    Confianza intradía: 
                    <span style='color: {id_result["color"]}; font-size: 1.3em; font-weight: bold;'>
                        {id_result["confianza"]}%
                    </span>
                    &nbsp;|&nbsp; Puntos detectados: <b style='color:white'>{id_result["puntos"]}</b>/14
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Niveles intradía
            st.markdown("#### 📍 Niveles Intradía (ATR 5min)")
            i1, i2, i3, i4, i5 = st.columns(5)
            i1.metric("Entrada Ahora",   f"${id_result['entrada']:.2f}")
            i2.metric("Stop Loss",       f"${id_result['stop_loss']:.2f}",
                      delta=f"-{id_result['riesgo_pct']:.2f}%", delta_color="inverse")
            i3.metric("Objetivo 1",      f"${id_result['objetivo1']:.2f}",
                      delta=f"+{id_result['ganancia_pct']:.2f}%")
            i4.metric("Objetivo 2",      f"${id_result['objetivo2']:.2f}")
            i5.metric("R/B Intradía",    f"1 : {id_result['rr']:.1f}")

            # Info extra intradía
            st.caption(
                f"Mín día: ${id_result['min_dia']:.2f} | "
                f"Máx día: ${id_result['max_dia']:.2f} | "
                f"ATR 5min: ${id_result['atr_5m']:.3f} | "
                f"Stoch RSI 5m: {id_result['stoch_k_5m']} | "
                f"RSI 5m: {id_result['rsi_5m']} | "
                + (f"Soporte 1h: ${id_result['soporte_1h']:.2f}" if id_result.get('soporte_1h') else "")
            )

            # Señales y alertas intradía
            col_si, col_ai = st.columns(2)
            with col_si:
                st.markdown("**✅ Señales Detectadas**")
                if id_result["señales"]:
                    for s in id_result["señales"]:
                        st.success(s)
                else:
                    st.info("Sin señales de piso activas")
            with col_ai:
                st.markdown("**⚠️ Alertas**")
                if id_result["alertas"]:
                    for a in id_result["alertas"]:
                        st.warning(a)
                else:
                    st.success("Sin alertas críticas")

            # Gráfica intradía
            st.markdown("#### 📈 Gráfica Intradía 5 minutos")
            fig_id = grafica_intraday(
                id_result.get("df_5m"),
                id_result.get("df_1h"),
                id_result,
                ticker_a_usar,
                nombre_a_usar
            )
            if fig_id:
                st.plotly_chart(fig_id, use_container_width=True)

        st.markdown("---")
        # ══════════════════════════════════════════════════
        # FIN MÓDULO PISO INTRADÍA
        # ══════════════════════════════════════════════════

        # Agotamiento de vendedores
        st.markdown("### Agotamiento de Vendedores")
        nivel_colores = {0: "#ff4444", 1: "#FF8C00", 2: "#FFD700", 3: "#00ff88"}
        nivel_iconos  = {0: "Sin confirmar", 1: "Debil", 2: "Moderado", 3: "FUERTE"}
        nv = resultado["agot_nivel"]
        color_agot = nivel_colores[nv]
        st.markdown(f"""
        <div style='background:#1a1a2e; border:2px solid {color_agot};
             border-radius:12px; padding:16px; margin-bottom:12px;'>
            <h3 style='color:{color_agot}; margin:0;'>
                {nivel_iconos[nv]} — {resultado["agot_desc"]}
            </h3>
        </div>
        """, unsafe_allow_html=True)

        if resultado["agot_detalles"]:
            for detalle in resultado["agot_detalles"]:
                st.success(f"✅ {detalle}")
        else:
            st.warning("Ninguna capa de agotamiento confirmada todavia")

        if resultado["velas_rev"]:
            velas_str = " | ".join([v[1] for v in resultado["velas_rev"][-3:]])
            st.success(f"Patrones de vela: {velas_str}")

        st.markdown("### Analisis de Rebote")
        col_r, col_a = st.columns(2)
        with col_r:
            st.markdown("**Senales a Favor**")
            for r in resultado["razones"]:
                st.success(r)
        with col_a:
            st.markdown("**Alertas**")
            for a in resultado["alertas"]:
                st.warning(a)

        st.markdown("### Indicadores")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("HMM Viterbi", resultado["estado_hmm"])
        c2.metric("RSI",         resultado["rsi"],
                  "Sobreventa" if resultado["rsi"] < 35 else
                  "Sobrecompra" if resultado["rsi"] > 70 else "Neutral")
        c3.metric("Stoch RSI K", resultado["stoch_k"],
                  "Zona rebote" if resultado["stoch_k"] < 20 else
                  "Sobrecompra" if resultado["stoch_k"] > 80 else "")
        c4.metric("MACD",        resultado["macd"],
                  f"Signal: {resultado['signal_line']}")
        c5.metric("Datos",       f"{resultado['datos']} dias")

        st.markdown("### Backtesting (2 anos)")
        b1, b2, b3 = st.columns(3)
        b1.metric("Senales Compra",    resultado["compras_bt"])
        b2.metric("Senales No Entrar", resultado["ventas_bt"])
        pct = round((resultado["compras_bt"] + resultado["ventas_bt"]) /
                    resultado["total_bt"] * 100, 1)
        b3.metric("Actividad", f"{pct}%")

        st.markdown("### Grafica con Niveles de Entrada y Salida")
        fig = grafica_rebote_profesional(
            resultado["df_chart"], ticker_a_usar,
            nombre_a_usar, niv, resultado["velas_rev"]
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.caption(f"AI.LINO Pro | HMM + Stoch RSI + Agotamiento 5 Capas + Piso Intradía | {ticker_a_usar}")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555;'>AI.Lino Pro 2026 — Herramienta educativa. No es asesoria financiera.</p>",
    unsafe_allow_html=True
)
