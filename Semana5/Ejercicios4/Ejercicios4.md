### Ejercicios 4 CC0C2

#### Sección 1: Fundamentos teóricos
Estos ejercicios se centran en explicar los conceptos básicos y unirlos sin código.

1. **Hipótesis distribucional y embeddings básicos**: Explica cómo la hipótesis distribucional (de los notebooks Representaciones_texto.ipynb y Semantica_vectorial_embeddings.ipynb) justifica el uso de embeddings. Contrasta embeddings densos (vectores continuos que capturan similitudes semánticas) con representaciones básicas como one-hot encoding (del notebook Representaciones_texto.ipynb).
   Usa el ejemplo de palabras como "gato" y "perro" del corpus_mini_es.txt: ¿Por qué one-hot no captura que ambas son animales, pero un embedding denso sí?

2. **Embeddings de Mikolov y Word2Vec**: Describe los embeddings de Mikolov como el núcleo de Word2Vec (de Semantica_vectorial_embeddings.ipynb), un modelo que
   aprende vectores densos (típicamente 50-300 dimensiones) de palabras basados en co-ocurrencias contextuales.
   Explica cómo une la hipótesis distribucional con redes neuronales: Word2Vec predice palabras en contextos para aprender vectores donde palabras similares (por ejemplo, "rey" y "reina" de analogias_es.tsv) estén cercanas en el espacio vectorial.

3. **Diferencias entre CBOW y Skip-gram**: Compara CBOW (predice la palabra objetivo a partir de su contexto, eficiente para palabras frecuentes) y Skip-gram (predice el contexto a partir de la palabra objetivo, mejor para palabras raras) de Semantica_vectorial_embeddings.ipynb. Usa el ejemplo "el gato duerme en el sofá" del corpus_mini_es.txt: En CBOW, el contexto ["el", "duerme", "en", "el", "sofá"] predice "gato"; en Skip-gram, "gato" predice el contexto. ¿Cuándo elegirías cada uno para un corpus pequeño como corpus_mini_es.txt?

4. **Negative sampling y optimización**: Explica negative sampling (de Semantica_vectorial_embeddings.ipynb) como una técnica en Word2Vec para reducir el costo computacional al muestrear "negativos" (palabras no contextuales) en lugar de softmax completo. Une esto a los embeddings de Mikolov: ¿Cómo acelera el entrenamiento de CBOW/Skip-gram sin perder calidad semántica? Usa analogias_es.tsv: ¿Por qué ayuda a capturar relaciones como "paris : francia :: madrid : españa"?

5. **Polisemia y limitaciones Context-Free**: Discute la polisemia (palabras con múltiples sentidos, por ejemplo, "banco" como asiento o institución) de Semantica_vectorial_embeddings.ipynb. Explica por qué los embeddings de Mikolov (Word2Vec) son "context-free" (un vector único por palabra, promediando sentidos) vs. modelos contextuales como BERT. Usa "gallo" (macho) y "gallina" (hembra) de analogias_es.tsv: ¿Cómo Word2Vec captura su similitud, pero falla en contextos ambiguos?

#### Sección 2: Análisis teórico-práctico

Estos ejercicios involucran cálculos manuales o análisis en papel, usando datos de los documentos.

6. **Construcción de vocabulario y bolsa de palabras**: Del notebook Representaciones_texto.ipynb, construye manualmente un vocabulario del corpus_mini_es.txt (por ejemplo, palabras únicas como "gato", "perro", "duerme"). Crea una matriz de bolsa de palabras (BoW) para las primeras 5 oraciones. Une a embeddings: ¿Por qué BoW es dispersa y no captura semántica, mientras que Word2Vec (Mikolov) la densifica? Calcula similitud coseno manual entre dos vectores BoW (por ejemplo, "la gata duerme en el sofá" vs. "el perro corre en el parque").

7. **Analogías manuales con hipótesis distribucional**: Usa analogias_es.tsv para listar 5 pares (por ejemplo, "rey : hombre :: reina : mujer"). Explica cómo la hipótesis distribucional predice que en embeddings de Mikolov, el vector de "reina" ≈ "rey" - "hombre" + "mujer". Analiza si corpus_mini_es.txt (con temas como animales y ciudades) apoyaría analogías como "gato : animal :: perro : animal". ¿CBOW o Skip-gram sería mejor para analogías en corpus pequeños?

8. **Matriz de co-ocurrencia y PPMI**: Del código en Semantica_vectorial_embeddings.ipynb, calcula manualmente una matriz de co-ocurrencia para un subconjunto de corpus_mini_es.txt (por ejemplo, ventana de 2 palabras para "el gato duerme en el sofá"). Aplica PPMI (Positive Pointwise Mutual Information) aproximado. Une a Word2Vec: ¿Cómo esta matriz estática se relaciona con los embeddings dinámicos de Mikolov, y por qué Skip-gram captura mejor asociaciones raras?

9. **Evaluación intrínseca vs. extrínseca**: De Semantica_vectorial_embeddings.ipynb, diferencia evaluación intrínseca (por ejemplo, similitud coseno en analogías) y extrínseca (por ejemplo, clasificación downstream). Para analogias_es.tsv, diseña una prueba intrínseca manual: Calcula si "paris - francia + españa ≈ madrid" conceptualmente, asumiendo vectores. ¿Cómo une esto CBOW/Skip-gram a embeddings prácticos?

10. **Sesgos en embeddings**: Analiza sesgos (de Semantica_vectorial_embeddings.ipynb) en embeddings de Mikolov, como género en "doctor : hombre :: enfermera : mujer". Usa analogias_es.tsv: ¿Qué sesgos podrían aparecer en un corpus como corpus_dom_devops_es.txt (técnico, posiblemente sesgado hacia términos masculinos)? Explica cómo negative sampling en Skip-gram podría amplificarlos.

#### Sección 3: Implementaciones prácticas
Estos ejercicios requieren código Python. Usa Gensim para Word2Vec, y los corpus proporcionados.

11. **Entrenamiento básico de Word2Vec con CBOW**: Instala Gensim y NLTK. Preprocesa corpus_mini_es.txt (tokeniza, minusculas, elimina stopwords). Entrena un modelo Word2Vec con CBOW (sg=0, vector_size=50, window=5, min_count=1). Imprime vectores para "gato" y "perro". Calcula similitud coseno entre ellos. Une conceptos: ¿Cómo CBOW (predicción de contexto) captura que son animales similares?

12. **Comparación CBOW vs. Skip-gram**: Repite el ejercicio 11 con Skip-gram (sg=1). Compara similitudes para palabras raras (por ejemplo, "machupicchu" del corpus). Grafica (con matplotlib) similitudes para 5 pares de analogias_es.tsv. Discute: ¿Por qué Skip-gram (embeddings de Mikolov para raros) outperforms CBOW en corpus pequeños como corpus_dom_devops_es.txt?

13. **Analogías con Word2Vec**: Usa el modelo entrenado en 11 o 12. Implementa analogías: model.wv.most_similar(positive=['rey', 'mujer'], negative=['hombre']). Prueba 5 de analogias_es.tsv (por ejemplo, "paris : francia :: ? : españa"). Evalúa precisión. Une: Explica cómo negative sampling optimiza esto en Skip-gram para capturar relaciones semánticas.

14. **Ajuste de hiperparámetros**: Entrena múltiples modelos en corpus_mini_es.txt variando window (2 vs. 10), vector_size (50 vs. 300), y epochs (5 vs. 20). Compara similitudes y analogías. Incluye submuestreo (sample=1e-3). Discute impactos (de Semantica_vectorial_embeddings.ipynb): ¿Cómo afecta window a CBOW/Skip-gram en embeddings densos?

15. **Adaptación a dominio específico**: Combina corpus_mini_es.txt con corpus_dom_devops_es.txt. Entrena Word2Vec (elige CBOW o Skip-gram). Compara vectores para términos generales ("servidor" en DevOps vs. general). Detecta desplazamiento semántico (de Semantica_vectorial_embeddings.ipynb). Une: ¿Cómo Word2Vec de Mikolov maneja polisemia en dominios mixtos?

#### Sección 4: Aplicaciones avanzadas 
Estos unen todo en proyectos más complejos.

16. **Clasificación con embeddings**: Usa embeddings de un modelo entrenado (ej. 12) como features. Clasifica oraciones de corpus_mini_es.txt en categorías (por ejemplo, animales vs. ciudades) con scikit-learn (SVM). Compara CBOW vs. Skip-gram. Reflexiona: ¿Cómo embeddings de Mikolov mejoran sobre BoW (Representaciones_texto.ipynb)?

17. **Visualización y hubness**: Reduce dimensionalidad de vectores (PCA o t-SNE) y visualiza palabras de corpus_dom_devops_es.txt. Identifica hubness (palabras centrales en alta dimensión, de Semantica_vectorial_embeddings.ipynb). Une: ¿Por qué ocurre en Skip-gram y cómo afecta analogías?

18. **Manejo de OOV y sesgos**: Para palabras OOV (fuera de vocabulario), usa subpalabras (inspirado en FastText de Representaciones_texto.ipynb, extensión de Mikolov). Prueba en analogias_es.tsv. Mitiga sesgos calculando analogías de género y ajustando. Reflexiona: Limitaciones context-free de Word2Vec.

19. **Comparación con modelos clásicos**: Implementa PPMI+SVD (código de Semantica_vectorial_embeddings.ipynb) en corpus_mini_es.txt. Compara similitudes con Word2Vec. Une: ¿Por qué embeddings de Mikolov (CBOW/Skip-gram) capturan mejor semántica distribucional que métodos estáticos?

20. **Proyecto final: Sistema de búsqueda semántica**: Construye un buscador: Entrena Word2Vec en el corpus combinado. Dada una query (por ejemplo, "servidor web"), encuentra oraciones similares via similitud coseno. Evalúa con analogías. Reflexiona: Cómo CBOW/Skip-gram unen hipótesis distribucional a aplicaciones reales, superando limitaciones de embeddings no-Mikolov.

