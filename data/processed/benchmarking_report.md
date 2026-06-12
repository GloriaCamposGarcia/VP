# REPORTE DE DESCUBRIMIENTO DE PATRONES Y VÍNCULOS AML OCULTOS
Generado el: 2026-06-11 17:27:55
Enfoque: Análisis No Supervisado Puro (Sin Clasificación ni Puntuación de Riesgo CNBV)

Este reporte detalla las estructuras latentes, los vínculos ocultos identificados entre entidades mediante similitud semántica y las anomalías de información detectadas cuantitativamente.

---

## 1. MÓDULO 1: Agrupamiento Semántico de Perfiles de Entidades
Propósito: Evaluar sentence-transformers locales contra representaciones agregadas de perfiles de entidad.

### Tabla Comparativa de Calidad de Clústeres (Representación Semántica)
| Algoritmo | Num_Clusters | Ruido_Detectado | Silhouette_Score | Cohesion | Tiempo_Ejecucion_Segs |
| --- | --- | --- | --- | --- | --- |
| K-Means | 5 | 0 | 0.0705 | 0.6808 | 3.5539 |
| HDBSCAN | 25 | 3579 | 0.2214 | 0.4788 | 19.4857 |

*Nota: HDBSCAN muestra un Silhouette superior y separa el ruido de forma efectiva, mientras que K-Means ofrece una segmentación completa y constante sin descartar datos.*

---

## 2. MÓDULO 2: Detección No Supervisada de Anomalías de Comportamiento e Información
Propósito: Identificar perfiles de comportamiento inusuales y outliers textuales o numéricos sin etiquetas de validación.

### Tabla de Anomalías Cuantitativas Detectadas
| Algoritmo | Num_Anomalias_Detectadas | Tiempo_Ejecucion_Segs |
| --- | --- | --- |
| IsolationForest | 243 | 0.2220 |
| OneClassSVM | 1871 | 0.1781 |
| LocalOutlierFactor | 249 | 0.1288 |

*Nota: Los modelos identidican un porcentaje fijo (5%) de entidades con desviaciones de comportamiento operacional en variables OSINT. Adicionalmente, el módulo de embeddings inyecta el flag de outlier de información en cada entidad.*

---

## 3. MÓDULO 3: Red Topológica y Vínculos Semánticos Ocultos (Análisis de Grafos)
Propósito: Mapear la conectividad implícita de las entidades y agruparlas en comunidades relacionales.

### Estadísticas Topológicas del Grafo de Entidades
| Métrica | Valor |
| --- | --- |
| Total de Entidades (Nodos) | 4962 |
| Relaciones por Referencias Compartidas | 2135218 |
| Relaciones por Contenido Compartido | 578590 |
| Vínculos Semánticos Ocultos Detectados (Embeddings) | 192532 |
| Total de Enlaces en el Grafo | 2906340 |
| Comunidades Estructurales Identificadas (Louvain) | 10 |

### Detalle de Comunidades de Red Louvain Detectadas (Top 5)
| community_id | size | miembros_principales |
| --- | --- | --- |
| 0 | 1969 | Ahmed Khalfan GHAILANI, POPULAR MOVEMENT OF KOSOVO, GASTON IYAMUREMYE, Abu ABBAS, Alcides RAMON MAGANA... |
| 4 | 1092 | GOLDEN HALL SERVICES CO., LTD., Lao- Asie Consultant Group, SINO-KENYA ENGINEERING GROUP COMNPANY LIMITED, FEDDERS ELECTRIC & ENGINEERING LTD., Pewee Flomo... |
| 2 | 676 | Hamid Reza SHARIFI TEHRANI, JOSE SANCHEZ MONROY, MARICELA DEL CONSUELO PEDROZA ARELLANO, RODRIGO BERNABE GARCIA SANCHEZ, ACME TECHNICAL SERVICES LLC... |
| 8 | 549 | ASFALTOS, CONSTRUCCIONES Y PROYECTOS YAHILL, S.A DE C.V., ABC CONSULTORES CONTABLES Y JURÍDICOS, S.A. DE C.V., ACHTECK, S.A. DE C.V., ACCESS ENTERTAINMENT, S. DE R.L. DE C.V., ABOAFF,  S.A. DE C.V.... |
| 5 | 368 | CL CONSTRUCTORA SA DE CV, ARRENDAMIENTO Y EDIFICACIONES LAZHER SA DE CV, FORMULARIOS E IMPRESOS SA DE CV, CHEMICALS AND MANUFACTURING DE MEXICO SA DE CV, GRUPO EMPRESARIAL ADDARO SA DE CV... |
---

## 4. ANÁLISIS VISUAL DE LA RED Y VÍNCULOS OCULTOS
Se han generado visualizaciones de red individuales (grafo ego de 1 salto) por cada Entity ID con conexiones en el siguiente directorio:

- **Directorio de Gráficos de Entidades**: [data/processed/entity_graphs/](file:///C:/Users/gloca/OneDrive/Desktop/Proyectos/Repositorios/Vincluos-Patrones/data/processed/entity_graphs/)

Cada imagen dispone de forma concéntrica a los vecinos alrededor de la entidad consultada (nodo dorado central), permitiendo identificar claramente:
- **Líneas grises continuas**: Conexiones físicas conocidas (URLs o hashes de contenido compartidos).
- **Líneas rojas discontinuas**: Vínculos de similitud semántica ocultos descubiertos por embeddings.

---

## 5. DETECCIÓN DE VÍNCULOS EN CADENA (MULTI-HOP) Y BUCLES SOSPECHOSOS
Propósito: Rastrear caminos de relación indirectos (hasta 3 saltos) desde listas de sanciones hacia entidades ordinarias/anómalas, y detectar ciclos de simulación.

### Top 5 Caminos en Cadena Detectados a Entidades de Interés/Anómalas
| Origen (Lista Negra) | Destino Ordinario | Saltos | Copia del Camino | Peso Promedio |
| --- | --- | --- | --- | --- |
| HAMAS | JEMAAH ISLAMIYAH | 2 | HAMAS -> ABU SAYYAF GROUP -> JEMAAH ISLAMIYAH | 2.5000 |
| HAMAS | TEHRIK-E TALIBAN PAKISTAN | 2 | HAMAS -> ABU SAYYAF GROUP -> TEHRIK-E TALIBAN PAKISTAN | 2.5000 |
| SHINING PATH | JEMAAH ISLAMIYAH | 2 | SHINING PATH -> ABU SAYYAF GROUP -> JEMAAH ISLAMIYAH | 2.5000 |
| SHINING PATH | TEHRIK-E TALIBAN PAKISTAN | 2 | SHINING PATH -> ABU SAYYAF GROUP -> TEHRIK-E TALIBAN PAKISTAN | 2.5000 |
| Usama bin Muhammad bin Awad BIN LADIN | JEMAAH ISLAMIYAH | 2 | Usama bin Muhammad bin Awad BIN LADIN -> ABU SAYYAF GROUP -> JEMAAH ISLAMIYAH | 2.5000 |

### Diagramas de Cadenas Críticas Generadas:
- **Cadena #1**: [HAMAS -> JEMAAH ISLAMIYAH](file:///C:/Users/gloca/OneDrive/Desktop/Proyectos/Repositorios/Vincluos-Patrones/data/processed/critical_chains/critical_chain_1.png)
- **Cadena #2**: [HAMAS -> TEHRIK-E TALIBAN PAKISTAN](file:///C:/Users/gloca/OneDrive/Desktop/Proyectos/Repositorios/Vincluos-Patrones/data/processed/critical_chains/critical_chain_2.png)
- **Cadena #3**: [SHINING PATH -> JEMAAH ISLAMIYAH](file:///C:/Users/gloca/OneDrive/Desktop/Proyectos/Repositorios/Vincluos-Patrones/data/processed/critical_chains/critical_chain_3.png)
- **Cadena #4**: [SHINING PATH -> TEHRIK-E TALIBAN PAKISTAN](file:///C:/Users/gloca/OneDrive/Desktop/Proyectos/Repositorios/Vincluos-Patrones/data/processed/critical_chains/critical_chain_4.png)
- **Cadena #5**: [Usama bin Muhammad bin Awad BIN LADIN -> JEMAAH ISLAMIYAH](file:///C:/Users/gloca/OneDrive/Desktop/Proyectos/Repositorios/Vincluos-Patrones/data/processed/critical_chains/critical_chain_5.png)

### Bucles Relacionales Cerrados Detectados (Triangulaciones)
| Largo del Bucle | Semillas en Lista Negra | Nodos Anómalos | ID de Nodos | Nombre de Nodos |
| --- | --- | --- | --- | --- |
| 5 | 5 | 5 | SAT_69-0119 - SAT_69-0469 - SAT_69-0499 - SAT_69-0513 - SAT_69-0517 - SAT_69-0119 | JUAN JOSE ESCANDON PAZ - ROSA MARIA VEGA LOPEZ - GUADALUPE ADELINA AMARILLAS ROMO - IGNACIO MONTES AHUMADA - PEDRO FEDERICO LAFUENTE DE ANDA - JUAN JOSE ESCANDON PAZ |
| 5 | 5 | 5 | SAT_69-0480 - SAT_69-0469 - SAT_69-0499 - SAT_69-0513 - SAT_69-0517 - SAT_69-0480 | MARIA ANTONIETA RODRIGUEZ MARTINEZ - ROSA MARIA VEGA LOPEZ - GUADALUPE ADELINA AMARILLAS ROMO - IGNACIO MONTES AHUMADA - PEDRO FEDERICO LAFUENTE DE ANDA - MARIA ANTONIETA RODRIGUEZ MARTINEZ |
| 5 | 5 | 5 | SAT_69-0226 - SAT_69-0177 - SAT_69-0469 - SAT_69-0499 - SAT_69-0513 - SAT_69-0226 | EDER LEON ACOSTA - LOUISE MICHELLE GUTIERREZ LARRAÑAGA - ROSA MARIA VEGA LOPEZ - GUADALUPE ADELINA AMARILLAS ROMO - IGNACIO MONTES AHUMADA - EDER LEON ACOSTA |
| 5 | 5 | 5 | EU_FINANCIAL_SANCTIONS-0106 - EU_FINANCIAL_SANCTIONS-0428 - EU_FINANCIAL_SANCTIONS-0259 - EU_FINANCIAL_SANCTIONS-0493 - EU_FINANCIAL_SANCTIONS-0461 - EU_FINANCIAL_SANCTIONS-0106 | Amjad ABBAS - Станислав Петрович ШЕВЧУК - Андрей Юрьевич ПАВЛЮЧЕНКО - Andrei Andreevich PRAKAPUK - Aleksandr Grigorievitch MALOLETKO - Amjad ABBAS |
| 5 | 5 | 5 | EU_FINANCIAL_SANCTIONS-0341 - EU_FINANCIAL_SANCTIONS-0428 - EU_FINANCIAL_SANCTIONS-0259 - EU_FINANCIAL_SANCTIONS-0493 - EU_FINANCIAL_SANCTIONS-0461 - EU_FINANCIAL_SANCTIONS-0341 | ساجي درويش - Станислав Петрович ШЕВЧУК - Андрей Юрьевич ПАВЛЮЧЕНКО - Andrei Andreevich PRAKAPUK - Aleksandr Grigorievitch MALOLETKO - ساجي درويش |

*Nota: Los bucles cerrados representan agrupaciones de entidades altamente vinculadas entre sí, útiles para identificar redes de empresas fantasma u operaciones simuladas.*
