# REPORTE DE DESCUBRIMIENTO DE PATRONES Y VÍNCULOS AML OCULTOS
Generado el: 2026-06-16 06:14:04
Enfoque: Análisis No Supervisado Puro (Sin Clasificación ni Puntuación de Riesgo CNBV)

Este reporte detalla las estructuras latentes, los vínculos ocultos identificados entre entidades mediante similitud semántica y las anomalías de información detectadas cuantitativamente.

---

## 1. MÓDULO 1: Agrupamiento Semántico de Perfiles de Entidades
Propósito: Evaluar sentence-transformers locales contra representaciones agregadas de perfiles de entidad.

### Tabla Comparativa de Calidad de Clústeres (Representación Semántica)
| Algoritmo | Num_Clusters | Ruido_Detectado | Silhouette_Score | Cohesion | Tiempo_Ejecucion_Segs |
| --- | --- | --- | --- | --- | --- |
| K-Means | 5 | 0 | 0.0671 | 0.6787 | 0.0000 |
| HDBSCAN | 2 | 859 | 0.0409 | 0.7377 | 0.0000 |

*Nota: HDBSCAN muestra un Silhouette superior y separa el ruido de forma efectiva, mientras que K-Means ofrece una segmentación completa y constante sin descartar datos.*

---

## 2. MÓDULO 2: Detección No Supervisada de Anomalías de Comportamiento e Información
Propósito: Identificar perfiles de comportamiento inusuales y outliers textuales o numéricos sin etiquetas de validación.

### Tabla de Anomalías Cuantitativas Detectadas
| Algoritmo | Num_Anomalias_Detectadas | Tiempo_Ejecucion_Segs |
| --- | --- | --- |
| IsolationForest | 191 | 0.0382 |
| OneClassSVM | 1199 | 0.1538 |
| LocalOutlierFactor | 111 | 0.1763 |

*Nota: Los modelos identidican un porcentaje fijo (5%) de entidades con desviaciones de comportamiento operacional en variables OSINT. Adicionalmente, el módulo de embeddings inyecta el flag de outlier de información en cada entidad.*

---

## 3. MÓDULO 3: Red Topológica y Vínculos Semánticos Ocultos (Análisis de Grafos)
Propósito: Mapear la conectividad implícita de las entidades y agruparlas en comunidades relacionales.

### Estadísticas Topológicas del Grafo de Entidades
| Métrica | Valor |
| --- | --- |
| Total de Entidades (Nodos) | 4962 |
| Relaciones por Referencias Compartidas | 1952929 |
| Relaciones por Contenido Compartido | 321 |
| Vínculos Semánticos Ocultos Detectados (Embeddings) | 204584 |
| Total de Enlaces en el Grafo | 2157834 |
| Comunidades Estructurales Identificadas (Louvain) | 10 |

### Detalle de Comunidades de Red Louvain Detectadas (Top 5)
| community_id | size | miembros_principales |
| --- | --- | --- |
| 0 | 1196 | AL-SHABAAB, Russian National Reinsurance Company JSC, Ansar al-Sharia Benghazi, YAZID SUFAAT, MIZBAN KHADR HADI... |
| 2 | 1099 | ASSOCIATION "HIDROSTROITEL", CONSTRUCCIONES DE OBRAS CIVILES S.A. DE C.V., MR. TANG CHI ANH, HEBEI CONSTRUCTION GROUP CORPORATION LIMITED JIAOZHOU BRANCH, Construtora Coesa S.A. - Sucursal Perú... |
| 1 | 906 | BM PROEKT, OOO, Miroslav KVOCKA, OJSC ACHINSK REFINERY, HANGZHOU HIKVISION DIGITAL TECHNOLOGY CO., LTD., GPB INVEST OOO... |
| 5 | 838 | Mzee Amigo, Dmitri Yuryevich TIKHONOV, Amir RADFAR, JOSE SANCHEZ MONROY, Norezai... |
| 4 | 551 | APRODEME ASOCIACIÓN PROFESIONAL DE DESARROLLO MERCANTIL, S.C., ALVAREZ MORAN JANET, SINDICATO NACIONAL REVOLUCIONARIO DE TRABAJADORES TRANSPORTISTAS EN GENERAL SIMILARES Y CONEXOS DE LA REPUBLICA MEXICANA DELEGACION HIDALGO, ALVICZA COMERCIAL, S.A. DE C.V., ASESORES Y EMPRESARIOS SAUSURA, S. DE R.L. DE C.V.... |
---

## 4. ANÁLISIS VISUAL DE LA RED Y VÍNCULOS OCULTOS
Se han generado visualizaciones de red individuales (grafo ego de 1 salto) por cada Entity ID con conexiones en el siguiente directorio:

- **Directorio de Gráficos de Entidades**: [data/processed/run_2026-06-16/entity_graphs/](file:///C:/Users/gloca/OneDrive/Desktop/Proyectos/Repositorios/Vincluos-Patrones/data/processed/use/runs/run_2026-06-16/entity_graphs/)

Cada imagen dispone de forma concéntrica a los vecinos alrededor de la entidad consultada (nodo dorado central), permitiendo identificar claramente:
- **Líneas grises continuas**: Conexiones físicas conocidas (URLs o hashes de contenido compartidos).
- **Líneas rojas discontinuas**: Vínculos de similitud semántica ocultos descubiertos por embeddings.

---

## 5. DETECCIÓN DE VÍNCULOS EN CADENA (MULTI-HOP) Y BUCLES SOSPECHOSOS
Propósito: Rastrear caminos de relación indirectos (hasta 3 saltos) desde listas de sanciones hacia entidades ordinarias/anómalas, y detectar ciclos de simulación.

### Top 5 Caminos en Cadena Detectados a Entidades de Interés/Anómalas
| Origen (Lista Negra) | Destino Ordinario | Saltos | Copia del Camino | Peso Promedio |
| --- | --- | --- | --- | --- |
| HAMAS | TEHRIK-E TALIBAN PAKISTAN | 2 | HAMAS -> ABU SAYYAF GROUP -> TEHRIK-E TALIBAN PAKISTAN | 2.5000 |
| POPULAR FRONT FOR THE LIBERATION OF PALESTINE | TEHRIK-E TALIBAN PAKISTAN | 2 | POPULAR FRONT FOR THE LIBERATION OF PALESTINE -> ABU SAYYAF GROUP -> TEHRIK-E TALIBAN PAKISTAN | 2.5000 |
| POPULAR FRONT FOR THE LIBERATION OF PALESTINE - GENERAL COMMAND | TEHRIK-E TALIBAN PAKISTAN | 2 | POPULAR FRONT FOR THE LIBERATION OF PALESTINE - GENERAL COMMAND -> ABU SAYYAF GROUP -> TEHRIK-E TALIBAN PAKISTAN | 2.5000 |
| SHINING PATH | TEHRIK-E TALIBAN PAKISTAN | 2 | SHINING PATH -> ABU SAYYAF GROUP -> TEHRIK-E TALIBAN PAKISTAN | 2.5000 |
| Usama bin Muhammad bin Awad BIN LADIN | TEHRIK-E TALIBAN PAKISTAN | 2 | Usama bin Muhammad bin Awad BIN LADIN -> ABU SAYYAF GROUP -> TEHRIK-E TALIBAN PAKISTAN | 2.5000 |

### Diagramas de Cadenas Críticas Generadas:
- **Cadena #1**: [HAMAS -> TEHRIK-E TALIBAN PAKISTAN](file:///C:/Users/gloca/OneDrive/Desktop/Proyectos/Repositorios/Vincluos-Patrones/data/processed/use/runs/run_2026-06-16/critical_chains/critical_chain_1_2026-06-16.png)
- **Cadena #2**: [POPULAR FRONT FOR THE LIBERATION OF PALESTINE -> TEHRIK-E TALIBAN PAKISTAN](file:///C:/Users/gloca/OneDrive/Desktop/Proyectos/Repositorios/Vincluos-Patrones/data/processed/use/runs/run_2026-06-16/critical_chains/critical_chain_2_2026-06-16.png)
- **Cadena #3**: [POPULAR FRONT FOR THE LIBERATION OF PALESTINE - GENERAL COMMAND -> TEHRIK-E TALIBAN PAKISTAN](file:///C:/Users/gloca/OneDrive/Desktop/Proyectos/Repositorios/Vincluos-Patrones/data/processed/use/runs/run_2026-06-16/critical_chains/critical_chain_3_2026-06-16.png)
- **Cadena #4**: [SHINING PATH -> TEHRIK-E TALIBAN PAKISTAN](file:///C:/Users/gloca/OneDrive/Desktop/Proyectos/Repositorios/Vincluos-Patrones/data/processed/use/runs/run_2026-06-16/critical_chains/critical_chain_4_2026-06-16.png)
- **Cadena #5**: [Usama bin Muhammad bin Awad BIN LADIN -> TEHRIK-E TALIBAN PAKISTAN](file:///C:/Users/gloca/OneDrive/Desktop/Proyectos/Repositorios/Vincluos-Patrones/data/processed/use/runs/run_2026-06-16/critical_chains/critical_chain_5_2026-06-16.png)

### Bucles Relacionales Cerrados Detectados (Triangulaciones)
| Largo del Bucle | Semillas en Lista Negra | Nodos Anómalos | ID de Nodos | Nombre de Nodos |
| --- | --- | --- | --- | --- |
| 5 | 5 | 5 | EU_FINANCIAL_SANCTIONS-0272 - EU_FINANCIAL_SANCTIONS-0508 - EU_FINANCIAL_SANCTIONS-0501 - SAT_69-0266 - EU_FINANCIAL_SANCTIONS-0424 - EU_FINANCIAL_SANCTIONS-0272 | Tatiana Georgievna BRATCHENKO - Sergei Sergeevich NAUMETS - Igor Venediktovich MASLOV - ISAAC NEDVEDOVICH SKVIRSKY - Yury Vladimirovich LEONOV - Tatiana Georgievna BRATCHENKO |
| 5 | 5 | 5 | EU_FINANCIAL_SANCTIONS-0316 - EU_FINANCIAL_SANCTIONS-0508 - EU_FINANCIAL_SANCTIONS-0501 - SAT_69-0266 - EU_FINANCIAL_SANCTIONS-0424 - EU_FINANCIAL_SANCTIONS-0316 | Oksana Aleksandrovna CHIGRINA - Sergei Sergeevich NAUMETS - Igor Venediktovich MASLOV - ISAAC NEDVEDOVICH SKVIRSKY - Yury Vladimirovich LEONOV - Oksana Aleksandrovna CHIGRINA |
| 5 | 5 | 5 | EU_FINANCIAL_SANCTIONS-0401 - EU_FINANCIAL_SANCTIONS-0508 - EU_FINANCIAL_SANCTIONS-0501 - SAT_69-0266 - EU_FINANCIAL_SANCTIONS-0424 - EU_FINANCIAL_SANCTIONS-0401 | Andrey Anatolievitch IVANYUTIN - Sergei Sergeevich NAUMETS - Igor Venediktovich MASLOV - ISAAC NEDVEDOVICH SKVIRSKY - Yury Vladimirovich LEONOV - Andrey Anatolievitch IVANYUTIN |
| 5 | 5 | 5 | EU_FINANCIAL_SANCTIONS-0453 - EU_FINANCIAL_SANCTIONS-0508 - EU_FINANCIAL_SANCTIONS-0501 - SAT_69-0266 - EU_FINANCIAL_SANCTIONS-0424 - EU_FINANCIAL_SANCTIONS-0453 | Alexey Nikolaevich MIKHAYLOV - Sergei Sergeevich NAUMETS - Igor Venediktovich MASLOV - ISAAC NEDVEDOVICH SKVIRSKY - Yury Vladimirovich LEONOV - Alexey Nikolaevich MIKHAYLOV |
| 5 | 5 | 5 | EU_FINANCIAL_SANCTIONS-0541 - EU_FINANCIAL_SANCTIONS-0508 - EU_FINANCIAL_SANCTIONS-0501 - SAT_69-0266 - EU_FINANCIAL_SANCTIONS-0424 - EU_FINANCIAL_SANCTIONS-0541 | Eduard Anatolyevich LYSENKO - Sergei Sergeevich NAUMETS - Igor Venediktovich MASLOV - ISAAC NEDVEDOVICH SKVIRSKY - Yury Vladimirovich LEONOV - Eduard Anatolyevich LYSENKO |

*Nota: Los bucles cerrados representan agrupaciones de entidades altamente vinculadas entre sí, útiles para identificar redes de empresas fantasma u operaciones simuladas.*
