# Analise de Acidentes de Transito nos EUA

## Visao Geral
Projeto de exploracao dos dados publicos **US Accidents** (Kaggle) para identificar fatores climaticos, regionais e de infraestrutura associados a ocorrencia e a gravidade dos acidentes de transito nos Estados Unidos. O pipeline automatiza o download, a limpeza, a engenharia de variaveis e a producao de visualizacoes estaticas e interativas.

## Pre-requisitos
- Python 3.10+
- Bibliotecas: kagglehub, pandas, seaborn, matplotlib, plotly

Instalacao sugerida:
```bash
pip install -r requirements.txt  # ou
pip install kagglehub pandas seaborn matplotlib plotly
```

## Como executar
```bash
python analysis.py
```
O script baixa a versao mais recente do dataset via kagglehub, realiza limpeza e prepara todas as saidas dentro da pasta `outputs/`.

## Estrutura de Saida
```
outputs/
|- tables/
|  |- region_summary.csv
|  |- weather_summary.csv
|  |- road_feature_summary.csv
|  |- hourly_heatmap_counts.csv
|- plots/
|  |- correlation_heatmap.png
|  |- weather_category_counts.png
|  |- weather_category_severity.png
|  |- road_feature_severe_rate.png
|  |- region_avg_severity.png
|  |- temporal_heatmap.png
`- interactive/
   |- state_severity_map.html
   |- severe_accidents_map.html
   |- temporal_heatmap.html
   |- weather_trend.html
```

## Principais Insights
- **Regioes:** Midwest lidera a severidade media (~2,34) e a taxa de acidentes graves (31%), enquanto o Sul concentra o maior volume de ocorrencias.
- **Clima:** Condicoes de Rain / Storm e Cloudy elevam a gravidade frente a tempo Clear.
- **Temporalidade:** Tercas e quartas-feiras entre 7h e 8h acumulam os maiores volumes de acidentes.
- **Infraestrutura:** Acidentes em Junction apresentam ~29% de severidade alta, significativamente acima de cruzamentos com sinalizacao (~9%).

## Proximos Passos Sugeridos
1. Trocar `px.scatter_mapbox` por `px.scatter_map` para evitar avisos de depreciacao do Plotly.
2. Salvar um recorte limpo em formato Parquet para acelerar analises adicionais ou treinamentos de modelos.

## Observacoes
- O arquivo CSV ultrapassa 3 GB; garanta espaco em disco e tempo de processamento adequados (execucao completa leva ~5 minutos em CPU comum).
- Para reutilizar o dataset sem novo download, mantenha o cache padrao criado em `~/.cache/kagglehub/`.
