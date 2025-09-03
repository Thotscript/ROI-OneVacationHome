# app.py
# -*- coding: utf-8 -*-
import os, re, logging
from typing import Dict, Any
from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
from unidecode import unidecode  # pip install unidecode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("OVH")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "ovh_dashboard.csv")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# -------------------- HELPERS -----------------------
# Mapas robustos PT/EN para nomes completos e abreviações
MONTH_MAP_FULL = {
    # EN
    "jan": "Jan", "january": "Jan",
    "feb": "Feb", "february": "Feb",
    "mar": "Mar", "march": "Mar",
    "apr": "Apr", "april": "Apr",
    "may": "May",
    "jun": "Jun", "june": "Jun",
    "jul": "Jul", "july": "Jul",
    "aug": "Aug", "august": "Aug",
    "sep": "Sep", "sept": "Sep", "september": "Sep",
    "oct": "Oct", "october": "Oct",
    "nov": "Nov", "november": "Nov",
    "dec": "Dec", "december": "Dec",
    # PT (sem acento via unidecode)
    "jan": "Jan", "janeiro": "Jan",
    "fev": "Feb", "fevereiro": "Feb",
    "mar": "Mar", "marco": "Mar",
    "abr": "Apr", "abril": "Apr",
    "mai": "May", "maio": "May",
    "jun": "Jun", "junho": "Jun",
    "jul": "Jul", "julho": "Jul",
    "ago": "Aug", "agosto": "Aug",
    "set": "Sep", "setembro": "Sep",
    "out": "Oct", "outubro": "Oct",
    "nov": "Nov", "novembro": "Nov",
    "dez": "Dec", "dezembro": "Dec",
}

NUM_TO_ABBR = {
    1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
}

VALID_MONTHS = set(NUM_TO_ABBR.values())

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas: minúsculo, sem acento, underscores"""
    df = df.copy()
    df.columns = (
        df.columns.map(lambda c: unidecode(str(c)))
                  .str.strip()
                  .str.lower()
                  .str.replace(r"\s+", "_", regex=True)
    )
    return df

def _money_to_float(v) -> float:
    if v is None or pd.isna(v): return 0.0
    if isinstance(v, (int,float)): return float(v)
    s = str(v).replace("R$","").replace("$","").strip()
    s = re.sub(r"[^0-9\.,-]", "", s).rstrip(".,")
    s = s.replace(".","").replace(",",".")
    try: return float(s)
    except: return 0.0

def _to_month(v) -> str:
    """Extrai abreviação de mês (Jan..Dec) de strings como '2025-02', '02/2025', 'Fev', 'Feb', '05M', '1.' etc.
    Nunca faz fallback silencioso para 'Jan'; retorna "" em caso de falha."""
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    if not s:
        return ""

    s_norm = unidecode(s).lower()

    # 1) Nome completo ou abreviação PT/EN
    if s_norm in MONTH_MAP_FULL:
        return MONTH_MAP_FULL[s_norm]

    # 2) Tenta pelos 3 primeiros caracteres (PT/EN)
    abbr3 = s_norm[:3]
    if abbr3 in MONTH_MAP_FULL:
        return MONTH_MAP_FULL[abbr3]

    # 3) Extrai um número de mês (1..12) de qualquer formato (ex.: 2025-02, 02/2025, 5M)
    nums = re.findall(r"(1[0-2]|0?[1-9])", s_norm)
    if nums:
        m = int(nums[-1])  # último match parece mais confiável em formatos AAAA-MM
        return NUM_TO_ABBR.get(m, "")

    # 4) Nada encontrado
    return ""

def _UP(x): 
    if x is None or pd.isna(x): 
        return ""
    return str(x).strip().upper()

def _make_key(condominio, tamanho):
    """Cria chave padronizada para união"""
    c = _UP(condominio) if condominio else ""
    t = _UP(tamanho) if tamanho else ""
    return f"{c} {t}".strip()

# -------------------- GLOBAIS -----------------------
_df_costs = pd.DataFrame()
_df_proj  = pd.DataFrame()
_df_occ   = pd.DataFrame()
_df_master = pd.DataFrame()

# -------------------- NORMALIZAÇÕES -----------------

def _normalize_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza planilha de Custos"""
    df = _normalize_columns(df)
    df = df.rename(columns={"dispesa": "despesa"})

    # Normaliza colunas obrigatórias
    required_cols = ["despesa", "condominio", "tamanho", "valor"]
    for col in required_cols:
        if col not in df.columns:
            log.warning(f"Coluna '{col}' não encontrada em Custos. Colunas disponíveis: {list(df.columns)}")
            df[col] = ""

    # Conserta typos comuns em 'despesa'
    if "despesa" in df.columns:
        df["despesa"] = df["despesa"].astype(str).str.strip()
        df["despesa"] = df["despesa"].replace({
            "Propery Management": "Property Management",
            "propery management": "Property Management",
        })

    # Converte valores monetários
    if "valor" in df.columns:
        df["valor"] = df["valor"].apply(_money_to_float)

    # Padroniza identificadores
    for c in ("condominio", "tamanho"):
        if c in df.columns:
            df[c] = df[c].apply(_UP)

    # Cria chave unificada
    df["chave"] = df.apply(lambda r: _make_key(r.get("condominio"), r.get("tamanho")), axis=1)

    log.info(f"Custos normalizados: {len(df)} linhas, chaves únicas: {df['chave'].nunique()}")
    return df


def _normalize_proj(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza planilha de Projeção Mensal"""
    df = _normalize_columns(df)

    # Verifica colunas essenciais
    required_cols = ["goal_actual", "mes_classificado", "tamanho", "condominio"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        log.warning(f"Colunas ausentes em Projeção: {missing_cols}. Colunas disponíveis: {list(df.columns)}")
        for col in missing_cols:
            if col == "goal_actual":
                df[col] = 0.0
            else:
                df[col] = ""

    # Converte valores monetários
    if "goal_actual" in df.columns:
        df["goal_actual"] = df["goal_actual"].apply(_money_to_float)

    # Padroniza identificadores
    for c in ("condominio", "tamanho"):
        if c in df.columns:
            df[c] = df[c].apply(_UP)

    # Normaliza mês
    if "mes_classificado" in df.columns:
        df["mes"] = df["mes_classificado"].apply(_to_month)
        # mantêm apenas meses válidos
        before = len(df)
        df = df[df["mes"].isin(VALID_MONTHS)].copy()
        after = len(df)
        if after < before:
            log.warning(f"Projeção: {before - after} linhas descartadas por mês inválido")
        log.info(f"Meses convertidos: {df['mes'].value_counts().to_dict()}")
    else:
        df["mes"] = ""

    # Cria chave unificada
    df["chave"] = df.apply(lambda r: _make_key(r.get("condominio"), r.get("tamanho")), axis=1)

    log.info(f"Projeção normalizada: {len(df)} linhas, chaves únicas: {df['chave'].nunique()}")
    log.info(f"Meses encontrados: {sorted(df['mes'].unique().tolist())}")
    return df


def _normalize_occ(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza planilha de Ocupação Mensal"""
    df = _normalize_columns(df)

    # Mapeia possíveis nomes de colunas
    col_mapping = {
        "mes": ["mes", "mês", "month"],
        "ocupacao": ["ocupacao", "ocupação", "occupation"],
        "dias": ["dias", "days"],
    }

    # Renomeia colunas para padrão
    for standard_name, possible_names in col_mapping.items():
        for col in df.columns:
            if col in possible_names:
                df = df.rename(columns={col: standard_name})
                break

    # ---- Ocupação (aceita valores como 75, 75%, 0.75) ----
    if "ocupacao" in df.columns:
        def _to_frac(x):
            if x is None or pd.isna(x):
                return 0.0
            s = str(x).strip()
            has_pct = "%" in s
            s_num = s.replace("%", "")
            num = pd.to_numeric(s_num, errors="coerce")
            if pd.isna(num):
                return 0.0
            # Se tinha '%' explícito OU número > 1, tratamos como 0..100
            if has_pct or float(num) > 1.0:
                return float(num) / 100.0
            # Já está em fração 0..1
            return float(num)
        df["ocupacao"] = df["ocupacao"].apply(_to_frac).clip(lower=0.0, upper=1.0)
    else:
        log.warning("Coluna 'ocupação' não encontrada")
        df["ocupacao"] = 0.0

    # ---- Dias ----
    if "dias" in df.columns:
        df["dias"] = pd.to_numeric(df["dias"], errors="coerce").fillna(0).astype(int)
    else:
        log.warning("Coluna 'dias' não encontrada")
        df["dias"] = 30

    # ---- Mês ----
    if "mes" in df.columns:
        df["mes"] = df["mes"].apply(_to_month)
        df = df[df["mes"].isin(VALID_MONTHS)].copy()
        # Um registro de ocupação por mês
        before = len(df)
        df = df.drop_duplicates(subset=["mes"], keep="last")
        after = len(df)
        if after < before:
            log.info(f"Ocupação: removidas {before - after} duplicatas por mês")
        log.info(f"Ocupação - meses convertidos: {df['mes'].value_counts().to_dict()}")
    else:
        log.warning("Coluna 'mes' não encontrada")
        df["mes"] = "Jan"

    log.info(f"Ocupação normalizada: {len(df)} linhas")
    log.info(f"Meses ocupação: {sorted(df['mes'].unique().tolist())}")
    return df

# -------------------- BUILD MASTER ------------------

def _compute_block(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas derivadas"""
    d = df.copy()

    # Garante que temos as colunas numéricas
    numeric_cols = ["receita_projetada", "ocupacao", "dias"]
    for col in numeric_cols:
        if col not in d.columns:
            d[col] = 0.0
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    d["dias_ocupados"] = d["dias"] * d["ocupacao"]

    d["adr"] = d.apply(
        lambda r: (r["receita_projetada"]/r["dias_ocupados"]) if r["dias_ocupados"] > 0 else 0.0, axis=1
    )
    d["revpar"] = d.apply(
        lambda r: (r["receita_projetada"]/r["dias"]) if r["dias"] > 0 else 0.0, axis=1
    )

    # Calcula receita anual por chave
    anual_rev = d.groupby(["chave"], dropna=False)["receita_projetada"].transform("sum")
    d["commission_fee_20"] = anual_rev * 0.20

    # Identifica colunas de despesas (exclui campos conhecidos)
    base_cols = {
        "condominio", "tamanho", "chave", "mes", "dias", "unit_id", "name",
        "receita_projetada", "receita_mensal_real", "ocupacao", "adr", "revpar",
        "dias_ocupados", "commission_fee_20", "despesas_outras_anuais", 
        "despesas_anuais", "lucro_mensal", "goal_actual", "mes_classificado"
    }

    expense_cols = [c for c in d.columns if c not in base_cols]

    # Converte colunas de despesas para numérico
    for c in expense_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    d["despesas_outras_anuais"] = d[expense_cols].sum(axis=1) if expense_cols else 0.0
    d["despesas_anuais"] = d["despesas_outras_anuais"] + d["commission_fee_20"]
    d["lucro_mensal"] = d["receita_projetada"] - (d["despesas_anuais"]/12.0)

    return d


def _build_master_df() -> pd.DataFrame:
    """Constrói DataFrame master com todos os dados unificados"""
    global _df_costs, _df_proj, _df_occ

    if _df_proj.empty:
        log.warning("DataFrame de projeção está vazio")
        return pd.DataFrame()

    # ---- Base: Projeção ----
    df = _df_proj.copy()

    # Garante que temos receita projetada
    df["receita_projetada"] = df["goal_actual"] if "goal_actual" in df.columns else 0.0
    df["receita_mensal_real"] = df["receita_projetada"]

    log.info(f"Base Projeção: {len(df)} linhas")

    # ---- União com Ocupação ----
    if not _df_occ.empty:
        occ = _df_occ[["mes", "ocupacao", "dias"]].drop_duplicates("mes", keep="last").copy()
        log.info(f"Fazendo merge com ocupação por 'mes'")

        before_merge = len(df)
        # validate='m:1' garante que não haja multiplicação de linhas por mês duplicado em 'occ'
        df = pd.merge(df, occ, on="mes", how="left", validate="m:1")
        after_merge = len(df)

        log.info(f"Merge ocupação: {before_merge} -> {after_merge} linhas")

        # Preenche valores ausentes
        df["ocupacao"] = df["ocupacao"].fillna(0.0)
        df["dias"] = df["dias"].fillna(30)
    else:
        log.warning("DataFrame de ocupação está vazio, usando valores padrão")
        df["ocupacao"] = 0.0
        df["dias"] = 30

    # ---- União com Custos ----
    if not _df_costs.empty:
        costs = _df_costs.copy()

        # Pivot das despesas por chave
        pivot = (
            costs
            .groupby(["chave", "despesa"], dropna=False)["valor"]
            .sum()
            .unstack("despesa", fill_value=0)
            .reset_index()
        )

        log.info(f"Pivot custos: {len(pivot)} linhas, colunas: {list(pivot.columns)}")

        # Merge com dados principais
        before_merge = len(df)
        df = pd.merge(df, pivot, on="chave", how="left")
        after_merge = len(df)

        log.info(f"Merge custos: {before_merge} -> {after_merge} linhas")

        # Preenche despesas ausentes com 0
        expense_cols = [c for c in pivot.columns if c != "chave"]
        for col in expense_cols:
            df[col] = df[col].fillna(0.0)

    # ---- Logs de debug ----
    log.info(f"Chaves únicas no master: {df['chave'].nunique()}")
    log.info(f"Condominios únicos: {sorted(df['condominio'].dropna().unique().tolist())}")
    log.info(f"Tamanhos únicos: {sorted(df['tamanho'].dropna().unique().tolist())}")

    if "mes" in df.columns:
        log.info(f"Meses no master: {sorted(df['mes'].dropna().unique().tolist())}")

    # ---- Computa métricas finais ----
    master = _compute_block(df).reset_index(drop=True)

    log.info(f"Master final: {len(master)} linhas")
    return master

# -------------------- PERSIST -----------------------

def _save_master(df: pd.DataFrame):
    global _df_master
    _df_master = df
    try:
        df.to_csv(DATA_CSV, index=False)
        log.info(f"Master salvo em CSV: {len(df)} linhas")
    except Exception as e:
        log.warning(f"Erro ao salvar CSV: {e}")

def _load_master():
    global _df_master
    if os.path.exists(DATA_CSV):
        try:
            _df_master = pd.read_csv(DATA_CSV)
            log.info(f"Master carregado do CSV: {len(_df_master)} linhas")
        except Exception as e:
            log.warning(f"Erro ao carregar CSV: {e}")
            _df_master = pd.DataFrame()
    else:
        _df_master = pd.DataFrame()

# -------------------- ROTAS -------------------------
@app.get("/")
def index():
    return send_from_directory("/var/www/html/roi", "index.html")

@app.get("/api/options")
def api_options():
    _load_master()

    # Usa projeção como fonte principal (tem dados mensais)
    if not _df_proj.empty:
        src = _df_proj.copy()
    elif not _df_master.empty:
        src = _df_master.copy()
    else:
        return jsonify({"condominios": [], "tamanhos": []})

    # Padroniza identificadores
    src["condominio"] = src["condominio"].apply(_UP)
    src["tamanho"] = src["tamanho"].apply(_UP)

    # Remove valores vazios
    src = src[src["condominio"].str.len() > 0]
    src = src[src["tamanho"].str.len() > 0]

    cond = request.args.get("condominio")
    if cond:
        # Retorna tamanhos para o condomínio selecionado
        tam = sorted(src.loc[src["condominio"]==_UP(cond), "tamanho"].dropna().unique().tolist())
        tam = [t for t in tam if t.strip()]  # Remove vazios
        log.info(f"[/api/options] tamanhos de {cond} -> {tam}")
        return jsonify({"tamanhos": tam})

    # Retorna todos os condomínios
    condos = sorted(src["condominio"].dropna().unique().tolist())
    condos = [c for c in condos if c.strip()]  # Remove vazios
    log.info(f"[/api/options] condominios -> {condos}")
    return jsonify({"condominios": condos})

@app.get("/api/data")
def api_data():
    _load_master()
    cond = request.args.get("condominio")
    tam  = request.args.get("tamanho")

    if not cond or not tam:
        return jsonify({"error":"Informe condominio e tamanho."}), 400

    df = _df_master.copy()
    if df.empty:
        return jsonify({"data": [], "totals": {}, "expenses": {}})

    # Padroniza filtros
    df["condominio"] = df["condominio"].apply(_UP)
    df["tamanho"] = df["tamanho"].apply(_UP)

    sel = df[(df["condominio"]==_UP(cond)) & (df["tamanho"]==_UP(tam))]

    log.info(f"[/api/data] filtro cond={cond} tam={tam} -> linhas={len(sel)}")

    if sel.empty:
        return jsonify({"data": [], "totals": {}, "expenses": {}})

    # ---- Dados mensais ----
    cols = ["mes","dias","receita_projetada","ocupacao","adr","revpar","receita_mensal_real","lucro_mensal"]
    rows = sel[cols].copy()

    # Converte para tipos corretos
    rows["dias"] = pd.to_numeric(rows["dias"], errors="coerce").fillna(0).astype(int)
    for c in cols:
        if c not in ["mes","dias"]:
            rows[c] = pd.to_numeric(rows[c], errors="coerce").fillna(0.0)

    # Renomeia para o frontend
    rows = rows.rename(columns={
        "mes": "Mês",
        "dias": "Dias", 
        "receita_projetada": "Receita Projetada",
        "ocupacao": "Ocupação",
        "adr": "ADR",
        "revpar": "RevPar",
        "receita_mensal_real": "Receita Mensal Real",
        "lucro_mensal": "Lucro Mensal"
    })

    # ---- Totais ----
    total_rev = float(pd.to_numeric(sel["receita_projetada"], errors="coerce").fillna(0).sum())
    total_days = float(pd.to_numeric(sel["dias"], errors="coerce").fillna(0).sum())
    total_days_occ = float((pd.to_numeric(sel["dias"], errors="coerce").fillna(0) *
                            pd.to_numeric(sel["ocupacao"], errors="coerce").fillna(0)).sum())

    # Pega despesas anuais de uma linha representativa (remove duplicatas por chave)
    annual_expenses = float(sel.drop_duplicates("chave")["despesas_anuais"].sum())

    adr = (total_rev/total_days_occ) if total_days_occ > 0 else 0.0
    revpar = (total_rev/total_days) if total_days > 0 else 0.0
    occ = (total_days_occ/total_days) if total_days > 0 else 0.0
    annual_profit = total_rev - annual_expenses
    monthly_profit = annual_profit/12.0

    totals = {
        "revenue": total_rev,
        "occupation": occ,
        "adr": adr,
        "revpar": revpar,
        "annual_revenue": total_rev,
        "annual_expenses": annual_expenses,
        "annual_profit": annual_profit,
        "monthly_profit": monthly_profit
    }

    # ---- Despesas ----
    expenses: Dict[str, float] = {}
    uniq = sel.drop_duplicates(subset=["chave"])

    # Identifica colunas de despesas
    base_cols = {
        "condominio", "tamanho", "chave", "mes", "dias", "unit_id", "name",
        "receita_projetada", "receita_mensal_real", "ocupacao", "adr", "revpar",
        "dias_ocupados", "commission_fee_20", "despesas_outras_anuais",
        "despesas_anuais", "lucro_mensal", "goal_actual", "mes_classificado"
    }

    for c in uniq.columns:
        if c not in base_cols:
            v = float(pd.to_numeric(uniq[c], errors="coerce").fillna(0).sum())
            if v > 0:
                # Formata nome da despesa
                label = c.replace("_"," ").title()
                if c.upper() == "HOA":
                    label = "HOA"
                elif "propery" in c.lower() or "property" in c.lower():
                    label = "Property Management"
                expenses[label] = v

    # Adiciona taxa de comissão
    commission = float(pd.to_numeric(uniq["commission_fee_20"], errors="coerce").fillna(0).sum())
    if commission > 0:
        expenses["Commission Fee (20%)"] = commission

    log.info(f"[/api/data] totals={totals}")
    log.info(f"[/api/data] expenses={expenses}")

    return jsonify({
        "data": rows.to_dict(orient="records"), 
        "totals": totals, 
        "expenses": expenses
    })

# -------------------- WEBHOOK ------------------------
@app.route("/webhooks/sheets-recv", methods=["POST"])
def sheets_recv():
    global _df_costs, _df_proj, _df_occ

    payload = request.get_json(silent=True) or {}
    sheet = payload.get("sheet")
    rows = payload.get("rows", [])

    if not sheet or not rows:
        return jsonify({"ok": False, "error": "sem sheet/rows"}), 400

    df = pd.DataFrame(rows)

    # Log das colunas recebidas
    log.info(f"[webhook] sheet={sheet} colunas recebidas: {list(df.columns)}")

    try:
        if sheet == "Custos":
            new_costs = _normalize_costs(df)

            # Se já temos dados de custos, fazemos append em vez de substituir
            if not _df_costs.empty:
                # Remove duplicatas por chave+despesa para evitar dados duplicados
                combined = pd.concat([_df_costs, new_costs], ignore_index=True)
                _df_costs = combined.drop_duplicates(subset=['chave', 'despesa'], keep='last')
            else:
                _df_costs = new_costs

            log.info(f"[webhook] Custos processados: {len(new_costs)} novas linhas, total: {len(_df_costs)} linhas")

        elif sheet == "Projecao-Mensal":
            new_proj = _normalize_proj(df)

            # Se já temos dados de projeção, fazemos append
            if not _df_proj.empty:
                # Remove duplicatas por chave+mes para evitar dados duplicados
                combined = pd.concat([_df_proj, new_proj], ignore_index=True)
                _df_proj = combined.drop_duplicates(subset=['chave', 'mes'], keep='last')
            else:
                _df_proj = new_proj

            log.info(f"[webhook] Projeção processada: {len(new_proj)} novas linhas, total: {len(_df_proj)} linhas")

        elif sheet == "Ocupacao-Mensal":
            new_occ = _normalize_occ(df)

            # Para ocupação, como são dados globais por mês, substituímos
            _df_occ = new_occ
            log.info(f"[webhook] Ocupação processada: {len(_df_occ)} linhas")

        else:
            return jsonify({"ok": False, "error": f"sheet desconhecida: {sheet}"}), 400

        # Reconstrói o master apenas se temos projeção
        master = _build_master_df()
        _save_master(master)

        # Log de resumo dos dados finais
        if not _df_proj.empty:
            condos = sorted(_df_proj['condominio'].dropna().unique().tolist())
            log.info(f"[webhook] Condomínios disponíveis: {len(condos)} - {condos[:10]}")

        log.info(f"[webhook] {sheet} processado com sucesso. Master: {len(master)} linhas")
        return jsonify({"ok": True, "sheet": sheet, "rows": len(df), "master_rows": len(master)})

    except Exception as e:
        log.error(f"[webhook] Erro processando {sheet}: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

# -------------------- MAIN ---------------------------
if __name__ == "__main__":
    # Carrega dados existentes na inicialização
    _load_master()
    app.run(host="0.0.0.0", port=3000, debug=False)
