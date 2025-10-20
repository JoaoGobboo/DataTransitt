---
marp: true
paginate: true
---

<!-- _class: lead -->
# DataTransitt
## Análise de Acidentes de Trânsito nos EUA

- Kaggle US Accidents Dataset
- Download, limpeza e visualizações automatizadas
- Foco em fatores climáticos, regionais e de infraestrutura

---

## Objetivos

- Identificar padrões de gravidade em diferentes regiões
- Investigar influência de clima e características viárias
- Criar visualizações estáticas e interativas reproducíveis

---

## Pipeline

1. `analysis.py` baixa e atualiza o dataset
2. Limpeza e engenharia de variáveis chave (clima, região, infraestrutura)
3. Geração de tabelas `outputs/tables/`
4. Plotagens em `outputs/plots/` e visualizações interativas em `outputs/interactive/`

---

## Principais Insights

- Midwest com maior severidade média (~2,34) e 31% de acidentes graves
- Condições Rain / Storm e Cloudy elevam gravidade frente a tempo Clear
- Terças e quartas entre 7h-8h concentram mais ocorrências
- Eventos em `Junction` apresentam ~29% de severidade alta

---

## Severidade por Região

<img src="outputs/plots/region_avg_severity.png" alt="Severidade média de acidentes por região dos EUA" width="80%" />

---

## Clima e Gravidade

<img src="outputs/plots/weather_category_counts.png" alt="Comparação das categorias de clima por número de acidentes" width="80%" />

---

## Clima e Severidade


<img src="outputs/plots/weather_category_severity.png" alt="Severidade média por categoria de clima" width="80%" />

---

## Pontos Críticos na Infraestrutura

<img src="outputs/plots/road_feature_severe_rate.png" alt="Taxa de acidentes severos em diferentes características viárias" width="80%" />

---

## Ritmo Temporal

<img src="outputs/plots/temporal_heatmap.png" alt="Mapa de calor de acidentes por dia da semana e horário" width="80%" />

---

## Variáveis Correlacionadas

<img src="outputs/plots/correlation_heatmap.png" alt="Matriz de correlação das principais variáveis" width="80%" />

---

## Requisitos e Execução

- Python 3.10+, `kagglehub`, `pandas`, `seaborn`, `matplotlib`, `plotly`
- Instalação: `pip install -r requirements.txt`
- Execução rápida: `python analysis.py`
- Resultados completos em `outputs/`

---

## Próximos Passos

- Integrar alertas para novas versões do dataset
- Explorar modelos preditivos de severidade
- Publicar painel interativo com atualizações automáticas
