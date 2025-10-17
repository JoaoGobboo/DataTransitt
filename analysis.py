import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import kagglehub  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("kagglehub must be installed to download the dataset") from exc

DATASET_HANDLE = "sobhanmoosavi/us-accidents"
OUTPUT_ROOT = Path("outputs")
PLOTS_DIR = OUTPUT_ROOT / "plots"
INTERACTIVE_DIR = OUTPUT_ROOT / "interactive"
TABLES_DIR = OUTPUT_ROOT / "tables"

USECOLS = [
    "ID",
    "Severity",
    "Start_Time",
    "End_Time",
    "Start_Lat",
    "Start_Lng",
    "Distance(mi)",
    "City",
    "County",
    "State",
    "Timezone",
    "Weather_Timestamp",
    "Weather_Condition",
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Temperature(F)",
    "Wind_Speed(mph)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Direction",
    "Precipitation(in)",
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal",
    "Turning_Loop",
]

DTYPE_MAP = {
    "Severity": "Int8",
    "Start_Lat": "float32",
    "Start_Lng": "float32",
    "Distance(mi)": "float32",
    "Temperature(F)": "float32",
    "Wind_Speed(mph)": "float32",
    "Humidity(%)": "float32",
    "Pressure(in)": "float32",
    "Visibility(mi)": "float32",
    "Precipitation(in)": "float32",
    "Amenity": "boolean",
    "Bump": "boolean",
    "Crossing": "boolean",
    "Give_Way": "boolean",
    "Junction": "boolean",
    "No_Exit": "boolean",
    "Railway": "boolean",
    "Roundabout": "boolean",
    "Station": "boolean",
    "Stop": "boolean",
    "Traffic_Calming": "boolean",
    "Traffic_Signal": "boolean",
    "Turning_Loop": "boolean",
}

REGION_MAP = {
    "CT": "Northeast",
    "ME": "Northeast",
    "MA": "Northeast",
    "NH": "Northeast",
    "RI": "Northeast",
    "VT": "Northeast",
    "NJ": "Northeast",
    "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest",
    "IN": "Midwest",
    "MI": "Midwest",
    "OH": "Midwest",
    "WI": "Midwest",
    "IA": "Midwest",
    "KS": "Midwest",
    "MN": "Midwest",
    "MO": "Midwest",
    "NE": "Midwest",
    "ND": "Midwest",
    "SD": "Midwest",
    "AL": "South",
    "AR": "South",
    "DE": "South",
    "DC": "South",
    "FL": "South",
    "GA": "South",
    "KY": "South",
    "LA": "South",
    "MD": "South",
    "MS": "South",
    "NC": "South",
    "OK": "South",
    "SC": "South",
    "TN": "South",
    "TX": "South",
    "VA": "South",
    "WV": "South",
    "AZ": "West",
    "CO": "West",
    "ID": "West",
    "MT": "West",
    "NV": "West",
    "NM": "West",
    "UT": "West",
    "WY": "West",
    "AK": "West",
    "HI": "West",
    "CA": "West",
    "OR": "West",
    "WA": "West",
    "PR": "Territories",
}

ROAD_FEATURES = [
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal",
    "Turning_Loop",
]

NUMERIC_COLUMNS = [
    "Severity",
    "Distance(mi)",
    "Temperature(F)",
    "Wind_Speed(mph)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Precipitation(in)",
]

DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

plt.switch_backend("Agg")
sns.set_theme(style="whitegrid")


def resolve_dataset_path() -> Path:
    # Baixa ou reutiliza cache local e retorna o CSV bruto
    dataset_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
    csv_path = dataset_dir / "US_Accidents_March23.csv"
    if not csv_path.exists():  # pragma: no cover
        raise FileNotFoundError(f"Dataset file not found at {csv_path}")
    return csv_path


def simplify_weather(condition: str) -> str:
    if not isinstance(condition, str) or not condition:
        return "Unknown"
    text = condition.lower()
    if any(token in text for token in ("snow", "sleet", "freez", "ice", "hail", "blizzard")):
        return "Snow / Ice"
    if any(token in text for token in ("rain", "storm", "drizzle", "thunder", "t-storm", "shower")):
        return "Rain / Storm"
    if any(token in text for token in ("fog", "mist", "haze", "smoke")):
        return "Fog / Mist"
    if any(token in text for token in ("cloud", "overcast")):
        return "Cloudy"
    if any(token in text for token in ("clear", "fair")):
        return "Clear"
    if any(token in text for token in ("wind", "gust")):
        return "Windy"
    if any(token in text for token in ("dust", "sand", "ash")):
        return "Dust / Sand"
    return "Other"


def assign_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def ensure_output_dirs() -> None:
    for folder in (OUTPUT_ROOT, PLOTS_DIR, INTERACTIVE_DIR, TABLES_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    print("Loading dataset...")
    # Executa limpeza e enriquecimento em etapas sequenciais
    # Carrega subconjunto de colunas aplicando dtypes otimizados
    df = pd.read_csv(
        csv_path,
        usecols=USECOLS,
        dtype=DTYPE_MAP,
        low_memory=False,
    )
    original_rows = len(df)
    # Remove duplicidades por identificador
    df = df.drop_duplicates(subset="ID")
    print(f"Removed {original_rows - len(df):,} duplicate rows")

    # Converte campos temporais relevantes e descarta linhas sem posicao basica
    for col in ("Start_Time", "End_Time", "Weather_Timestamp"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["Start_Time", "Start_Lat", "Start_Lng"])

    # Preenche dados categoricos chave antes de agregacoes
    df["Weather_Condition"] = df["Weather_Condition"].fillna("Unknown")
    df["Timezone"] = df["Timezone"].fillna("Unknown")
    df["Sunrise_Sunset"] = df["Sunrise_Sunset"].fillna("Unknown")
    df["Civil_Twilight"] = df["Civil_Twilight"].fillna("Unknown")

    # Imputa numericos com mediana ou zero conforme aplicavel
    for col in NUMERIC_COLUMNS:
        if col not in df:
            continue
        if col == "Precipitation(in)":
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = df[col].fillna(df[col].median())

    # Garante indicadores booleanos sem nulos
    for feature in ROAD_FEATURES:
        df[feature] = df[feature].fillna(False)

    # Deriva atributos de clima, tempo e severidade auxiliar
    df["Weather_Category"] = df["Weather_Condition"].map(simplify_weather)
    df["Start_Hour"] = df["Start_Time"].dt.hour
    df["Day_Of_Week"] = df["Start_Time"].dt.day_name()
    df["Month"] = df["Start_Time"].dt.month
    df["Year"] = df["Start_Time"].dt.year
    df["Season"] = df["Month"].map(assign_season)

    df["US_Region"] = df["State"].map(REGION_MAP).fillna("Other")
    # Marca acidentes severos e resume infraestrutura presente
    df["Is_Severe"] = df["Severity"] >= 3
    df["Infra_Count"] = df[ROAD_FEATURES].sum(axis=1)

    return df


def save_tables(df: pd.DataFrame) -> None:
    print("Saving summary tables...")
    # Sintetiza metricas agregadas para exploracao rapida
    accident_summary = (
        df.groupby("US_Region")
        .agg(
            accidents=("ID", "count"),
            avg_severity=("Severity", "mean"),
            severe_rate=("Is_Severe", "mean"),
        )
        .reset_index()
        .sort_values("accidents", ascending=False)
    )
    accident_summary.to_csv(TABLES_DIR / "region_summary.csv", index=False)

    # Resume volume e severidade por categoria de clima simplificada
    # Analisa distribuicao e severidade por categoria de clima
    weather_summary = (
        df.groupby("Weather_Category")
        .agg(
            accidents=("ID", "count"),
            avg_severity=("Severity", "mean"),
            severe_rate=("Is_Severe", "mean"),
        )
        .reset_index()
        .sort_values("accidents", ascending=False)
    )
    weather_summary.to_csv(TABLES_DIR / "weather_summary.csv", index=False)

    # Percorre atributos viarios booleanos e calcula indicadores
    road_rows = []
    for feature in ROAD_FEATURES:
        feature_df = df[df[feature]]
        if feature_df.empty:
            continue
        road_rows.append(
            {
                "feature": feature,
                "accidents_with_feature": int(feature_df.shape[0]),
                "share_of_total": feature_df.shape[0] / df.shape[0],
                "avg_severity_with_feature": feature_df["Severity"].mean(),
                "severe_rate_with_feature": feature_df["Is_Severe"].mean(),
            }
        )
    road_summary = pd.DataFrame(road_rows).sort_values(
        "severe_rate_with_feature", ascending=False
    )
    road_summary.to_csv(TABLES_DIR / "road_feature_summary.csv", index=False)

    # Gera serie para heatmap temporal
    hourly_counts = (
        df.groupby(["Day_Of_Week", "Start_Hour"])["ID"].count().reset_index()
    )
    hourly_counts.to_csv(TABLES_DIR / "hourly_heatmap_counts.csv", index=False)


def create_seaborn_visuals(df: pd.DataFrame) -> None:
    print("Creating static charts...")
    # Correla matriz numerica para destacar relacoes globais
    numeric_df = df[NUMERIC_COLUMNS].copy()
    corr = numeric_df.corr(method="spearman")
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Spearman correlation among key numeric features")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()

    # Analisa distribuicao e severidade por categoria de clima
    weather_summary = (
        df.groupby("Weather_Category")
        .agg(avg_severity=("Severity", "mean"), accidents=("ID", "count"))
        .reset_index()
        .sort_values("accidents", ascending=False)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=weather_summary,
        x="accidents",
        y="Weather_Category",
        palette="viridis",
        order=weather_summary["Weather_Category"].tolist(),
    )
    plt.title("Accident counts by weather category")
    plt.xlabel("Accidents")
    plt.ylabel("Weather category")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "weather_category_counts.png", dpi=300)
    plt.close()

    severity_sorted = weather_summary.sort_values("avg_severity", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=severity_sorted,
        x="avg_severity",
        y="Weather_Category",
        palette="rocket",
        order=severity_sorted["Weather_Category"].tolist(),
    )
    plt.title("Average severity by weather category")
    plt.xlabel("Average severity")
    plt.ylabel("Weather category")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "weather_category_severity.png", dpi=300)
    plt.close()

    # Compara taxa de severidade entre carateristicas de infraestrutura
    road_summary = pd.read_csv(TABLES_DIR / "road_feature_summary.csv")
    if not road_summary.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=road_summary,
            x="severe_rate_with_feature",
            y="feature",
            palette="magma",
        )
        plt.title("Severe accident share when feature is present")
        plt.xlabel("Severe accident share")
        plt.ylabel("Road feature")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "road_feature_severe_rate.png", dpi=300)
        plt.close()

    # Visualiza severidade media por macro-regiao
    region_summary = pd.read_csv(TABLES_DIR / "region_summary.csv")
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=region_summary,
        x="US_Region",
        y="avg_severity",
        palette="crest",
    )
    plt.title("Average severity by US region")
    plt.xlabel("Region")
    plt.ylabel("Average severity")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "region_avg_severity.png", dpi=300)
    plt.close()

    # Prepara tabela hora x dia para heatmap
    pivot = (
        df.pivot_table(
            index="Day_Of_Week",
            columns="Start_Hour",
            values="ID",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(DAY_ORDER)
        .reindex(columns=range(24))
    )
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="YlOrRd")
    plt.title("Accident counts by day of week and hour")
    plt.xlabel("Hour of day")
    plt.ylabel("Day of week")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "temporal_heatmap.png", dpi=300)
    plt.close()


def create_plotly_visuals(df: pd.DataFrame) -> None:
    print("Creating interactive visuals...")
    # Agrega estatisticas por estado para mapas
    state_summary = (
        df.groupby("State")
        .agg(
            avg_severity=("Severity", "mean"),
            severe_rate=("Is_Severe", "mean"),
            accidents=("ID", "count"),
        )
        .reset_index()
    )
    state_summary["severe_rate"] = state_summary["severe_rate"] * 100
    # Choropleth mostra severidade media em todo o territorio
    choropleth = px.choropleth(
        state_summary,
        locations="State",
        locationmode="USA-states",
        color="avg_severity",
        scope="usa",
        color_continuous_scale="Reds",
        hover_data={"accidents": ":,", "severe_rate": ":.2f"},
        title="Average accident severity by state",
    )
    choropleth.write_html(INTERACTIVE_DIR / "state_severity_map.html", include_plotlyjs="cdn")

    severe_df = df[df["Is_Severe"]]
    sample_size = 15000
    # Limita amostra para manter interatividade responsiva
    if severe_df.shape[0] > sample_size:
        severe_sample = severe_df.sample(sample_size, random_state=42)
    else:
        severe_sample = severe_df
    # Mapa interativo destaca amostra de acidentes severos
    scatter_map = px.scatter_mapbox(
        severe_sample,
        lat="Start_Lat",
        lon="Start_Lng",
        color="Severity",
        size="Distance(mi)",
        hover_data={
            "City": True,
            "County": True,
            "State": True,
            "Weather_Category": True,
            "Start_Time": True,
        },
        zoom=3,
        height=700,
        title="Sample of severe accidents (severity >= 3)",
    )
    scatter_map.update_layout(mapbox_style="open-street-map")
    scatter_map.write_html(
        INTERACTIVE_DIR / "severe_accidents_map.html",
        include_plotlyjs="cdn",
    )

    # Heatmap interativo com padrao temporal
    temporal_pivot = (
        df.groupby(["Day_Of_Week", "Start_Hour"])["ID"].count().unstack(fill_value=0)
    )
    temporal_pivot = temporal_pivot.reindex(DAY_ORDER).reindex(columns=range(24))
    heatmap = px.imshow(
        temporal_pivot,
        labels=dict(x="Hour of day", y="Day of week", color="Accidents"),
        color_continuous_scale="Turbo",
        title="Temporal heatmap of accidents",
    )
    heatmap.update_yaxes(categoryorder="array", categoryarray=DAY_ORDER)
    heatmap.write_html(INTERACTIVE_DIR / "temporal_heatmap.html", include_plotlyjs="cdn")

    # Serie temporal por categoria de clima
    weather_timeline = (
        df.groupby(["Year", "Weather_Category"])["ID"].count().reset_index()
    )
    weather_line = px.line(
        weather_timeline,
        x="Year",
        y="ID",
        color="Weather_Category",
        markers=True,
        title="Trend of accidents by weather category",
    )
    weather_line.write_html(
        INTERACTIVE_DIR / "weather_trend.html",
        include_plotlyjs="cdn",
    )


def main() -> None:
    ensure_output_dirs()
    # Orquestra pipeline: download, limpeza, sumarizacao e visualizacoes
    csv_path = resolve_dataset_path()
    df = load_and_clean(csv_path)
    save_tables(df)
    create_seaborn_visuals(df)
    create_plotly_visuals(df)
    print("Analysis complete. Outputs saved under the 'outputs' directory.")


if __name__ == "__main__":
    main()
