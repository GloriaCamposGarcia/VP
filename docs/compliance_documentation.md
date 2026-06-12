# DOCUMENTACIÓN DE CUMPLIMIENTO PLD/AML (CNBV & UIF MÉXICO)

Este documento detalla la justificación técnica y el fundamento  del sistema de monitoreo transaccional y OSINT.

---

## 1. MARCO REGULATORIO Y ALINEACIÓN CON LA CNBV
De acuerdo con las **Disposiciones de Carácter General (DCG)** aplicables a las Instituciones de Tecnología Financiera (ITF) en México, emitidas por la **Comisión Nacional Bancaria y de Valores (CNBV)** y la **Unidad de Inteligencia Financiera (UIF)** de la Secretaría de Hacienda y Crédito Público (SHCP), las entidades financieras deben contar con metodologías de evaluación de riesgos y sistemas automatizados capaces de detectar y reportar operaciones inusuales y preocupantes.

### Umbrales de Reporte y Alerta
El sistema permite parametrizar e integrar reglas duras que alertan de forma automática operaciones que cruzan los umbrales normativos:
- **Operaciones en Efectivo / Límites Locales:** Alertas automáticas para depósitos en efectivo equivalentes o superiores a **$50,000 MXN** (o límites de riesgo acumulado mensual de acuerdo con el nivel de cuenta del cliente).
- **Transferencias Internacionales (Moneda Extranjera):** Vigilancia y reporte automático de transferencias electrónicas de fondos (órdenes de pago internacionales) que igualen o superen los **$10,000 USD** (o su equivalente en moneda nacional).
- **Monitoreo Transaccional Continuo:** Detección de fraccionamiento de operaciones (pitufeo o *smurfing*) diseñadas para eludir los umbrales de reporte individuales.

---

## 2. EXPLICABILIDAD DE INTELIGENCIA ARTIFICIAL (EXPLAINABLE AI)
Una de las mayores objeciones regulatorias al uso de modelos de Machine Learning en cumplimiento es la naturaleza de "caja negra". El sistema resuelve esta limitante aplicando técnicas de explicabilidad:

- **Importancia de Características (Feature Importance):** El modelo supervisado RandomForest proporciona una métrica de la relevancia de cada variable en la decisión de auditoría. Factores como `max_identity_score` (score de identidad de coincidencia en listas) y `evidence_items` (volumen de pruebas acumulado) representan el núcleo del árbol de decisiones.
- **Transparencia en Decisiones del Analista:** El modelo supervisado aprende a imitar el flujo lógico del analista humano diferenciando homónimos de coincidencias reales. Si el sistema emite una alerta, el oficial de cumplimiento puede auditar qué variable (coincidencia de lista de sanciones frente a PEP, o el número de fuentes) pesó más en el veredicto del modelo.

---

## 3. SOLUCIÓN AL PROBLEMA DE HOMÓNIMOS (FALSOS POSITIVOS)
El enfoque tradicional puramente basado en coincidencia de nombres genera un **95% de falsos positivos**, saturando a las áreas de cumplimiento. El sistema reduce este margen combinando:

1. **Módulo de Embeddings Semánticos (Procesamiento de Lenguaje Natural):** Los embeddings generados a partir de los snippets de noticias OSINT permiten agrupar los hallazgos según su contexto y no su nombre. Por ejemplo, se distingue si el texto refiere a una "sanción por lavado de dinero" o simplemente a un "evento corporativo ordinario" que involucra un nombre homónimo.
2. **Detección de Anomalías No Supervisada:** Isolation Forest y Local Outlier Factor analizan la distribución conjunta de variables numéricas cuantitativas. Si un nombre coincide con una lista de control pero el score de coincidencia de identidad (`max_identity_score`) es bajo y no hay evidencias adicionales (`evidence_items` = 0), el modelo lo separa de las anomalías reales.

---

## 4. DETECCIÓN DE PATRONES RELACIONALES (VÍNCULOS OCULTOS)
Las redes de lavado de dinero operan mediante esquemas de fragmentación transaccional o compartición de estructuras. El análisis topológico y la propagación de grafos implementados en el sistema revelan estos patrones:

- **Detección de Comunidades (Algoritmo Louvain):** Agrupa entidades que, a pesar de estar aparentemente desconectadas, comparten elementos comunes en el dataset OSINT (mismos hashes de contenido, URLs o identificadores).
- **PageRank y Convolución GNN (PyTorch Geometric):**
  - **PageRank** mide la centralidad y relevancia del nodo en la red relacional de alertas. Los nodos con PageRank alto representan puentes de riesgo o entidades altamente conectadas con otras listas de control.
  - La **convolución GNN** (GCN) propaga las características cuantitativas de los nodos adyacentes. Si una entidad limpia comparte múltiples enlaces con entidades que presentan hallazgos adversos, el suavizado de la GCN altera sus embeddings relacionales, reflejando el incremento del riesgo transmitido por vecindad estructural sin necesidad de clasificaciones manuales.

---

## 5. GENERACIÓN AUTOMATIZADA DE NARRATIVAS DE ROI
El **Reporte de Operación Inusual (ROI)** exige una narrativa pormenorizada de los hechos y las razones de inusualidad. El módulo `narrative_generator` automatiza este proceso generando una descripción estructurada en español que detalla:
- El perfil general del cliente (Persona Física / Moral y geografía).
- Las razones cuantitativas de alerta (número de evidencias y nivel de coincidencia en listas).
- El contexto relacional detectado en los grafos (ID de comunidad relacional y PageRank).

Este reporte automatizado reduce drásticamente el tiempo operativo de redacción del oficial de cumplimiento, permitiendo que la Fintech responda con celeridad dentro de los plazos legales establecidos por la UIF.
