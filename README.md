# APIs de energia renovável e aprendizado de máquina

Projeto de avaliação que consulta duas APIs públicas, organiza os dados e resolve **duas tarefas**, comparando **três algoritmos** em cada uma:

1. **Classificação (ANEEL):** identificar a fonte de um empreendimento de geração (Solar, Eólica ou Hidráulica) a partir da potência e da localização.
2. **Regressão (Open-Meteo):** estimar a radiação solar horária em Petrolina (PE) a partir de variáveis meteorológicas e da hora.

## Fontes e período dos dados

| Tarefa        | Fonte | Período / recorte | Arquivo gerado |
|---------------|-------|-------------------|----------------|
| Classificação | [ANEEL — SIGA](https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel) (API CKAN/DataStore) | Cadastro atual; até 1.200 registros por sigla (`UFV`, `EOL`, `UHE`, `PCH`, `CGH`) | `aneel_classificacao_orange.csv` |
| Regressão     | [Open-Meteo — histórico](https://open-meteo.com/en/docs/historical-weather-api) (lat. -9,39, long. -40,50) | 01/04/2025 a 30/06/2025, horas de 7h a 17h, horário local (`America/Recife`) | `meteo_regressao_orange.csv` |

Ambas as consultas são públicas e **não exigem login, token ou chave de API**. Os dados do Open-Meteo são estimados por modelos/reanálise, não leituras de um sensor específico.

## Como executar

1. Instale as dependências: `pip install pandas numpy scikit-learn matplotlib seaborn plotly`.
2. Abra `Aula_APIs_Energia_Renovavel_ML.ipynb` e execute as células **na ordem**. As primeiras células consultam as APIs e geram os dois CSVs (é necessário acesso à internet). Os CSVs gerados também estão neste repositório.
3. O notebook lê os CSVs pelo caminho relativo `.\aneel_classificacao_orange.csv`. Em Linux/macOS, troque por `./aneel_classificacao_orange.csv`.

## Resumo geral

| Tarefa        | Melhor modelo     | Desempenho no teste                          |
|---------------|-------------------|----------------------------------------------|
| Classificação | Gaussian Process  | Acurácia 0,830 · F1 macro 0,823              |
| Regressão     | Gradient Boosting (empatado com Random Forest) | MAE 64,22 W/m² · R² 0,845 |

Em ambas as tarefas o limite está mais nos dados do que no algoritmo: na classificação, potência e localização não separam bem solar de eólica; na regressão, a radiação não equivale à energia gerada.

---

## 1. Classificação: fonte de empreendimentos de geração (`fonte`)

**Pergunta:** a partir da potência e da localização de um empreendimento, é possível classificar sua fonte como **Solar, Eólica ou Hidráulica**?

**Dados:** registros do SIGA (Sistema de Informações de Geração da ANEEL), consultados pela API pública CKAN/DataStore, sem token. Foram consultadas as siglas `UFV` (solar), `EOL` (eólica), `UHE`, `PCH` e `CGH` (as três últimas agrupadas na classe *Hidráulica*). Cada linha do CSV (`aneel_classificacao_orange.csv`) é um empreendimento.
**Entradas (X):** `potencia_kw`, `latitude`, `longitude`.
**Alvo (y):** `fonte` (Solar, Eólica, Hidráulica). A coluna `SigTipoGeracao` originou o alvo e **não entra em X**; nomes, CEG e campos de combustível também não foram usados, pois revelariam a classe.

### Base de dados

- 3.876 empreendimentos válidos, sem valores ausentes nas quatro colunas.
- A consulta limitou cada sigla a 1.200 linhas, então as proporções abaixo **não representam a matriz elétrica brasileira**.

| Classe     | Registros | Proporção |
|------------|----------:|----------:|
| Hidráulica | 1.476     | 38,1%     |
| Solar      | 1.200     | 31,0%     |
| Eólica     | 1.200     | 31,0%     |

As classes são razoavelmente equilibradas, então a acurácia não é enganosa aqui, mas as métricas foram calculadas com média **macro**.

**Divisão:** treino/teste com `train_test_split`, semente fixa (`random_state=42`), **75% treino (2.907) e 25% teste (969)**. Não foi usado `stratify`, então as proporções das classes no teste podem diferir levemente das da base (no teste: 288 eólicas, 363 hidráulicas e 318 solares). A `StandardScaler` foi colocada dentro de um `Pipeline`, portanto seu ajuste usa apenas o treino.

**O que os boxplots (por classe, no treino) mostram:**
- **Potência (escala log):** eólicas concentradas em uma faixa estreita (mediana ~30 MW). Hidráulicas espalhadas de 1 kW a mais de 10 GW. Solares com distribuição muito aberta: uma parte de usinas de ~1 kW (provavelmente geração distribuída) e outra de grandes usinas, na mesma faixa das eólicas.
- **Latitude:** solares e eólicas ficam mais ao norte (Nordeste), hidráulicas mais ao sul.
- **Longitude:** eólicas mais a leste (~-41°), hidráulicas e solares mais a oeste.
- **Coordenadas (0, 0):** há empreendimentos com latitude e longitude iguais a 0, provavelmente coordenada não preenchida na origem. Eles passaram pelo filtro de valores ausentes e aparecem como outliers nos boxplots.

### Modelos comparados (conjunto de teste)

Métricas de precisão, recall e F1 com média **macro** (média simples entre as três classes).

| Modelo                    | Accuracy | Precision | Recall | F1    |
|---------------------------|---------:|----------:|-------:|------:|
| Gaussian Process          | 0,830    | 0,841     | 0,830  | 0,823 |
| Ridge Classifier          | 0,776    | 0,782     | 0,772  | 0,771 |
| Bernoulli Naive Bayes     | 0,764    | 0,779     | 0,758  | 0,756 |

Os três algoritmos são de famílias diferentes: um processo gaussiano (não paramétrico, fronteiras flexíveis), um classificador linear regularizado e um modelo probabilístico bayesiano ingênuo.

![Matrizes de confusão](figuras/matrizes_confusao.png)

### Matriz de confusão do melhor modelo (Gaussian Process)

| Real \ Previsto | Eólica | Hidráulica | Solar | Recall da classe |
|-----------------|-------:|-----------:|------:|-----------------:|
| **Eólica**      | 272    | 16         | 0     | 0,94             |
| **Hidráulica**  | 18     | 326        | 19    | 0,90             |
| **Solar**       | 84     | 28         | 206   | 0,65             |

### Resumo dos resultados

- **Melhor modelo:** o Gaussian Process foi o melhor em todas as métricas (acurácia de ~83% e F1 macro de ~0,82), cerca de 5 a 7 pontos percentuais acima dos outros dois. A escolha é razoável, mas a diferença foi medida em uma única divisão de teste, sem validação cruzada, então deve ser lida com cautela.
- **Onde erra:** a classe **Solar** é a mais difícil (recall de 0,65). Dos 318 empreendimentos solares do teste, 84 foram classificados como eólicos e 28 como hidráulicos. Isso é coerente com os boxplots: grandes usinas solares e eólicas têm potências parecidas e ficam na mesma região (Nordeste), então potência e coordenadas quase não as distinguem. Eólica e Hidráulica são as melhores: a eólica quase nunca é confundida com solar (0 erros no Gaussian Process).
- **Padrão de erro igual nos três modelos:** todos confundem Solar com Eólica em proporção parecida (81 a 91 casos), o que indica que o limite está nas **variáveis**, não no algoritmo.
- **Modelos lineares e Bernoulli NB:** o Ridge é linear e o Bernoulli NB binariza as entradas (após a padronização, cada variável vira "acima/abaixo da média"), o que descarta muita informação. Isso ajuda a explicar o desempenho inferior, e o Naive Bayes ainda erra mais eólicas classificadas como hidráulicas (62 casos).
- **Observação:** as coordenadas (0, 0) são ruído de cadastro e podem prejudicar principalmente os modelos mais simples. Tratá-las (removendo ou marcando como ausentes) é um próximo passo natural.

### Potência e localização não bastam para uma aplicação real

O modelo classifica a **categoria do cadastro**, não a energia gerada. Potência outorgada é a capacidade nominal autorizada, não a produção, e a localização aproximada não separa fontes que ocupam as mesmas regiões. Na prática, uma fonte depende de fatores que não estão nas três entradas: recurso natural disponível (radiação, vento, vazão dos rios), tecnologia, porte do projeto (usina de grande porte × geração distribuída), fase do empreendimento e topografia. Por isso, o resultado serve como exercício de classificação, e não como ferramenta para identificar a fonte de um empreendimento real. Além disso, o recorte de 1.200 registros por sigla não reflete a distribuição real de fontes no país.

---

## 2. Regressão: estimativa da radiação solar (`radiacao_w_m2`)

**Dados:** medições horárias (das 7h às 17h) de abril a junho de 2025, com 1.001 horas válidas.
**Entradas (X):** `temperatura_c`, `umidade_pct`, `nuvens_pct`, `vento_kmh`, `hora`.
**Alvo (y):** `radiacao_w_m2`.
**Divisão:** cronológica, sem embaralhar. As primeiras ~80% das horas foram para treino (até 12/06) e as últimas ~20% para teste (13/06 a 30/06).

### Modelos comparados (conjunto de teste)

| Modelo            | MAE (W/m²) | RMSE (W/m²) | R²    |
|-------------------|-----------:|------------:|------:|
| Random Forest     | 66,25      | 85,15       | 0,845 |
| Gradient Boosting | 64,22      | 85,21       | 0,845 |
| Regressão Linear  | 145,21     | 173,30      | 0,360 |

![Comparação das métricas](figuras/metricas.png)
![Real vs. previsto](figuras/real_vs_previsto.png)

### Resumo dos resultados

- **Melhores modelos:** Gradient Boosting e Random Forest tiveram desempenho quase igual, com erro médio de ~65 W/m² e R² próximo de 0,85.
- **Pior modelo:** a Regressão Linear errou mais que o dobro (MAE 145 W/m²) e explicou só 36% da variação. A radiação ao longo do dia tem formato de sino, e uma reta não representa isso bem.
- **Peso da hora:** a `hora` foi a variável mais importante (queda de ~0,86 no R² ao ser embaralhada), seguida de `temperatura_c`. `umidade_pct` e `nuvens_pct` tiveram peso menor, e `vento_kmh` foi praticamente irrelevante. Embora a correlação linear da hora com a radiação seja baixa (0,12), a relação é forte e não linear, e por isso os modelos de árvores a capturam.
- **Erros:** o modelo erra mais nas horas de subida e descida da radiação (10h e 14h, ~90 W/m²) e menos no início e no fim do dia (7h e 17h), quando a radiação é baixa. O gráfico de real × previsto mostra que o modelo acompanha bem o padrão diário, mas suaviza os picos e as quedas causadas por nuvens.
- **Observação:** `temperatura_c` e `umidade_pct` são muito correlacionadas entre si (-0,93), o que pode prejudicar principalmente a regressão linear.

### Radiação prevista não é energia produzida

O modelo estima a radiação que chega ao local (potência em W/m²), não a energia elétrica gerada. Um sistema fotovoltaico depende também da eficiência dos painéis, da temperatura dos módulos, da inclinação e orientação, de sombras e sujeira, das perdas no inversor e cabos e do tamanho do sistema. Além disso, é preciso integrar a potência ao longo do tempo para obter energia (kWh).
