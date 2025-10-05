-- Consulta a base dados do INMET pela url: https://console.cloud.google.com/bigquery?p=basedosdados&d=br_inmet_bdmep&t=microdados&page=table&project=teste-controle-de-processos

WITH 
-- Passo 1: Mapa de Município para UF (Aqui são utilizados dados do MapBiomas para correlação entre UF e Município).
municipio_para_uf AS (
  SELECT DISTINCT
    id_municipio,
    sigla_uf
  FROM
    `basedosdados.br_mapbiomas_estatisticas.cobertura_municipio_classe`
),

-- Passo 2: Juntar os microdados com as estações, filtrando por ano a partir de 2010.
clima_com_municipio AS (
  SELECT
    microdados.ano,
    microdados.mes,
    estacao.id_municipio,
    microdados.precipitacao_total,
    microdados.pressao_atm_max,
    microdados.pressao_atm_min,
    microdados.temperatura_max,
    microdados.temperatura_min,
    microdados.umidade_rel_max,
    microdados.umidade_rel_min
  FROM
    `basedosdados.br_inmet_bdmep.microdados` AS microdados
  JOIN
    `basedosdados.br_inmet_bdmep.estacao` AS estacao
  ON
    microdados.id_estacao = estacao.id_estacao
  WHERE
    microdados.ano >= 2010
),

-- Passo 3: Agregação das informações das estações por Município.

clima_agregado_por_municipio AS (
  SELECT
    ano,
    mes,
    id_municipio,
    SUM(precipitacao_total) AS precip_total_mensal_municipio,
    AVG(temperatura_max) AS temp_max_media_mensal_municipio,
    AVG(temperatura_min) AS temp_min_media_mensal_municipio,
    AVG(umidade_rel_max) AS umidade_max_media_mensal_municipio,
    AVG(umidade_rel_min) AS umidade_min_media_mensal_municipio,
    AVG(pressao_atm_max) AS pressao_max_media_mensal_municipio,
    AVG(pressao_atm_min) AS pressao_min_media_mensal_municipio
  FROM
    clima_com_municipio
  GROUP BY
    ano,
    mes,
    id_municipio
)

-- Passo 4: Agregação Final para o Estado (UF), calculando a média das variáveis mensais dos municípios por UF.

SELECT
  clima_municipio.ano,
  clima_municipio.mes,
  uf_map.sigla_uf,
  

  AVG(clima_municipio.precip_total_mensal_municipio) AS precipitacao_media_mensal_uf,
  AVG(clima_municipio.temp_max_media_mensal_municipio) AS temp_max_media_mensal_uf,
  AVG(clima_municipio.temp_min_media_mensal_municipio) AS temp_min_media_mensal_uf,
  AVG(clima_municipio.umidade_max_media_mensal_municipio) AS umidade_max_media_mensal_uf,
  AVG(clima_municipio.umidade_min_media_mensal_municipio) AS umidade_min_media_mensal_uf,
  AVG(clima_municipio.pressao_max_media_mensal_municipio) AS pressao_max_media_mensal_uf,
  AVG(clima_municipio.pressao_min_media_mensal_municipio) AS pressao_min_media_mensal_uf

FROM
  clima_agregado_por_municipio AS clima_municipio
JOIN
  municipio_para_uf AS uf_map
ON
  clima_municipio.id_municipio = uf_map.id_municipio

-- Agrupamento por UF, ano e mês para se alinhar com dados SINAN
GROUP BY
  uf_map.sigla_uf,
  clima_municipio.ano,
  clima_municipio.mes

ORDER BY
  clima_municipio.ano,
  clima_municipio.mes;