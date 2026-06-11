# REPORTE DE DESCUBRIMIENTO DE PATRONES Y VÍNCULOS AML OCULTOS
Generado el: 2026-06-10 19:50:58
Enfoque: Análisis No Supervisado Puro (Sin Clasificación ni Puntuación de Riesgo CNBV)

Este reporte detalla las estructuras latentes, los vínculos ocultos identificados entre entidades mediante similitud semántica y las anomalías de información detectadas cuantitativamente.

---

## 1. MÓDULO 1: Agrupamiento Semántico de Perfiles de Entidades
Propósito: Evaluar sentence-transformers locales contra representaciones agregadas de perfiles de entidad.

### Tabla Comparativa de Calidad de Clústeres (Representación Semántica)
| Algoritmo | Num_Clusters | Ruido_Detectado | Silhouette_Score | Cohesion | Tiempo_Ejecucion_Segs |
| --- | --- | --- | --- | --- | --- |
| K-Means | 5 | 0 | 0.0705 | 0.6808 | 3.4711 |
| HDBSCAN | 25 | 3579 | 0.2214 | 0.4788 | 18.7787 |

*Nota: HDBSCAN muestra un Silhouette superior y separa el ruido de forma efectiva, mientras que K-Means ofrece una segmentación completa y constante sin descartar datos.*

---

## 2. MÓDULO 2: Detección No Supervisada de Anomalías de Comportamiento e Información
Propósito: Identificar perfiles de comportamiento inusuales y outliers textuales o numéricos sin etiquetas de validación.

### Tabla de Anomalías Cuantitativas Detectadas
| Algoritmo | Num_Anomalias_Detectadas | Tiempo_Ejecucion_Segs |
| --- | --- | --- |
| IsolationForest | 243 | 0.4246 |
| OneClassSVM | 1871 | 0.3523 |
| LocalOutlierFactor | 249 | 0.1257 |

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
| 0 | 1969 | PUMA SECURITY COMPANY, Latif Nusayyif Jasim AL-DULAYMI, Movement of Islamic Holy War, Dragoljub KUNARAC, Qadir... |
| 4 | 1092 | Metal Engineering Eood, Nabaraj Basnet, Canes Charles, MEDNIZA GLOBAL MERCHANTS LIMITED, Mactebac Contractors Limited... |
| 2 | 676 | Ko Ko Maung, FRANCISCO GARCIA GONZALEZ, Фаик Самеддин оглы МАМЕДОВ, Svetlana Vladimirovna DMITROVA, ARMANDO PERALES GANDARA... |
| 8 | 549 | ADMINISTRACION Y ASESORIA SKIMONO, S.A. DE C.V., ASESORÍA EMPRESARIAL SAN MIGUEL, S.A., ANGELES URIBE MARGARITO, ANVORT CONSULTORIA Y PROYECTOS, S.A. DE C.V., ADCODEC, S.C.... |
| 5 | 368 | DERIVADOS DE FRUTAS SA DE CV, PEGASO EXPRESS SA DE CV, MW INDUSTRIAL SA DE CV, CIA CONSTRUCTORA WIDISA SA DE CV, ESTRUCTURAS CORPORATIVAS DE OCCIDENTE SA DE CV... |