#!/usr/bin/env python3
"""
Updated RAG Pipeline using database-backed retrieval.
"""

from typing import List, Dict, Any, Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langdetect import detect, detect_langs, LangDetectException
import os
import hashlib
import logging

from .retrieval_db import DatabaseRetriever

logger = logging.getLogger(__name__)

class RAGPipelineDB:
    """
    Retrieval-Augmented Generation pipeline using database-backed retrieval
    and Ollama LLM for answer generation.
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", llm_model: str = "llama2", cache_enabled: Optional[bool] = None, cache_settings: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG pipeline.

        Args:
            model_name (str): Embedding model for retrieval
            llm_model (str): Ollama model for generation
            cache_enabled (bool): Whether to enable caching (overrides env var)
            cache_settings (dict): Cache configuration settings
        """
        self.retriever = DatabaseRetriever(model_name)
        # Start batch processing for improved performance
        self.retriever.start_batch_processing()
        self.llm = OllamaLLM(model=llm_model)

        # Initialize cache if enabled
        if cache_enabled is None:
            cache_enabled = os.getenv('CACHE_ENABLED', 'false').lower() == 'true'

        if cache_enabled:
            try:
                from .cache.redis_cache import RedisCache
                cache_config = cache_settings or {}
                self.cache = RedisCache(
                    host=cache_config.get('host', os.getenv('REDIS_HOST', 'localhost')),
                    port=int(cache_config.get('port', os.getenv('REDIS_PORT', 6379))),
                    password=cache_config.get('password', os.getenv('REDIS_PASSWORD')) or None,
                    db=int(cache_config.get('db', os.getenv('REDIS_DB', 0))),
                    ttl_hours=int(cache_config.get('ttl_hours', os.getenv('CACHE_TTL_HOURS', 24)))
                )
                self.cache_enabled = True
                logger.info("LLM caching enabled with Redis")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                self.cache = None
                self.cache_enabled = False
        else:
            self.cache = None
            self.cache_enabled = False

        # Multilingual prompt templates
        self.prompt_templates = {
            'en': PromptTemplate(
                input_variables=["context", "question"],
                template="""
You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Please provide a comprehensive and accurate answer based on the context above. When referencing information from specific sources, include the source reference in brackets like [Source 1: filename.pdf]. If the context doesn't contain enough information to answer the question, say so.

Answer:"""
            ),
            'de': PromptTemplate(
                input_variables=["context", "question"],
                template="""
Sie sind ein hilfreicher Assistent, der Fragen basierend auf dem bereitgestellten Kontext beantwortet.

Kontext:
{context}

Frage: {question}

KRITISCH WICHTIG: Sie MÜSSEN diese Frage AUSSCHLIESSLICH auf DEUTSCH beantworten. Es ist VERBOTEN, englische Wörter oder Sätze zu verwenden. Antworten Sie nur auf Deutsch!

Bitte geben Sie eine umfassende und genaue Antwort basierend auf dem obigen Kontext. Wenn Sie Informationen aus bestimmten Quellen referenzieren, fügen Sie den Quellverweis in Klammern ein, z.B. [Quelle 1: dateiname.pdf]. Wenn der Kontext nicht genügend Informationen enthält, um die Frage zu beantworten, sagen Sie das.

Antwort auf Deutsch:"""
            ),
            'fr': PromptTemplate(
                input_variables=["context", "question"],
                template="""
Vous êtes un assistant utile qui répond aux questions basées sur le contexte fourni.

Contexte:
{context}

Question: {question}

CRITIQUEMENT IMPORTANT: Vous DEVEZ répondre à cette question UNIQUEMENT en FRANÇAIS. Il est INTERDIT d'utiliser des mots ou des phrases anglais. Répondez seulement en français!

Veuillez fournir une réponse complète et précise basée sur le contexte ci-dessus. Lorsque vous référencez des informations provenant de sources spécifiques, incluez la référence de source entre crochets comme [Source 1: nomfichier.pdf]. Si le contexte ne contient pas suffisamment d'informations pour répondre à la question, dites-le.

Réponse en français:"""
            ),
            'es': PromptTemplate(
                input_variables=["context", "question"],
                template="""
Eres un asistente útil que responde preguntas basadas en el contexto proporcionado.

Contexto:
{context}

Pregunta: {question}

Por favor proporciona una respuesta completa y precisa basada en el contexto anterior. Cuando hagas referencia a información de fuentes específicas, incluye la referencia de fuente entre corchetes como [Fuente 1: nombrearchivo.pdf]. Si el contexto no contiene suficiente información para responder la pregunta, dilo.

Respuesta:"""
            ),
            'it': PromptTemplate(
                input_variables=["context", "question"],
                template="""
Sei un assistente utile che risponde alle domande basate sul contesto fornito.

Contesto:
{context}

Domanda: {question}

Fornisci una risposta completa e accurata basata sul contesto sopra. Quando fai riferimento a informazioni da fonti specifiche, includi il riferimento alla fonte tra parentesi quadre come [Fonte 1: nomefile.pdf]. Se il contesto non contiene informazioni sufficienti per rispondere alla domanda, dillo.

Risposta:"""
            ),
            'pt': PromptTemplate(
                input_variables=["context", "question"],
                template="""
Você é um assistente útil que responde perguntas com base no contexto fornecido.

Contexto:
{context}

Pergunta: {question}

Forneça uma resposta completa e precisa baseada no contexto acima. Quando fizer referência a informações de fontes específicas, inclua a referência da fonte entre colchetes como [Fonte 1: nomearquivo.pdf]. Se o contexto não contiver informações suficientes para responder à pergunta, diga isso.

Resposta:"""
            ),
            'nl': PromptTemplate(
                input_variables=["context", "question"],
                template="""
Je bent een behulpzame assistent die vragen beantwoordt op basis van de verstrekte context.

Context:
{context}

Vraag: {question}

Geef een uitgebreide en nauwkeurige antwoord gebaseerd op de bovenstaande context. Wanneer je verwijst naar informatie uit specifieke bronnen, neem dan de bronverwijzing op tussen vierkante haken zoals [Bron 1: bestandsnaam.pdf]. Als de context niet genoeg informatie bevat om de vraag te beantwoorden, zeg dat dan.

Antwoord:"""
            ),
            'sv': PromptTemplate(
                input_variables=["context", "question"],
                template="""
Du är en hjälpsam assistent som svarar på frågor baserat på den tillhandahållna kontexten.

Kontext:
{context}

Fråga: {question}

Vänligen ge ett omfattande och korrekt svar baserat på kontexten ovan. När du refererar till information från specifika källor, inkludera källreferensen inom hakparenteser som [Källa 1: filnamn.pdf]. Om kontexten inte innehåller tillräcklig information för att besvara frågan, säg det.

Svar:"""
            ),
            'pl': PromptTemplate(
                input_variables=["context", "question"],
                template="""
Jesteś pomocnym asystentem, który odpowiada na pytania w oparciu o dostarczony kontekst.

Kontekst:
{context}

Pytanie: {question}

Proszę podać wyczerpującą i dokładną odpowiedź w oparciu o powyższy kontekst. Jeśli kontekst nie zawiera wystarczających informacji, aby odpowiedzieć na pytanie, powiedz to. Gdy odwołujesz się do informacji z konkretnych źródeł, uwzględnij odniesienie do źródła w nawiasach kwadratowych, np. [Źródło 1: nazwapliku.pdf].

Odpowiedź:"""
            ),
            'zh': PromptTemplate(
                input_variables=["context", "question"],
                template="""
您是一个有用的助手，根据提供的上下文回答问题。

上下文：
{context}

问题：{question}

请根据上述上下文提供全面准确的答案。当引用特定来源的信息时，请在方括号中包含来源引用，如[来源1：文件名.pdf]。如果上下文不包含足够的信息来回答问题，请说明。

答案："""
            ),
            'ja': PromptTemplate(
                input_variables=["context", "question"],
                template="""
あなたは提供されたコンテキストに基づいて質問に答える役立つアシスタントです。

コンテキスト：
{context}

質問：{question}

上記のコンテキストに基づいて、包括的で正確な回答を提供してください。具体的なソースからの情報を参照する場合、[ソース1：ファイル名.pdf]のように角括弧内にソース参照を含めてください。コンテキストに質問に答えるのに十分な情報が含まれていない場合は、そう言ってください。

回答："""
            ),
            'ko': PromptTemplate(
                input_variables=["context", "question"],
                template="""
제공된 컨텍스트를 기반으로 질문에 답하는 유용한 어시스턴트입니다.

컨텍스트：
{context}

질문：{question}

위의 컨텍스트를 기반으로 포괄적이고 정확한 답변을 제공하십시오. 특정 소스의 정보를 참조할 때는 [소스 1: 파일명.pdf]처럼 대괄호 안에 소스 참조를 포함하십시오. 컨텍스트에 질문에 답할 충분한 정보가 포함되어 있지 않으면 그렇게 말하십시오.

답변："""
            )
        }

        # Default fallback template
        self.default_template = self.prompt_templates['en']

    def detect_query_language(self, query: str) -> str:
        """
        Detect the language of the user's query with improved accuracy.

        Args:
            query (str): The user's question

        Returns:
            str: ISO 639-1 language code or 'en' as fallback
        """
        if not query or len(query.strip()) < 3:
            return 'en'

        query = query.strip()

        try:
            # Get language probabilities
            lang_probs = detect_langs(query)
            if not lang_probs:
                return 'en'

            # Get the most probable language and its confidence
            top_lang = lang_probs[0]
            detected_lang = top_lang.lang
            confidence = top_lang.prob

            # For short queries or low confidence, use additional heuristics
            if len(query.split()) < 6 or confidence < 0.8:
                # Check for German-specific words and patterns
                german_verbs = ['bedeutet', 'heißt', 'funktioniert', 'arbeitet', 'macht', 'tut', 'wird', 'wurde', 'kann', 'soll', 'muss', 'darf', 'will', 'weiß', 'kennt', 'findet', 'gibt', 'sieht', 'hört', 'riecht', 'schmeckt', 'fühlt', 'denkt', 'glaubt', 'hofft', 'fürchtet', 'liebt', 'hasst']
                german_question_words = ['was', 'wie', 'wo', 'wann', 'warum', 'wieso', 'weshalb', 'wer', 'wen', 'wem', 'welche', 'welcher', 'welches', 'welchen']
                german_articles = ['der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer', 'einem', 'einen']
                german_prepositions = ['auf', 'unter', 'über', 'vor', 'hinter', 'neben', 'zwischen', 'bei', 'von', 'zu', 'aus', 'nach', 'gegen', 'ohne', 'durch', 'um', 'bis']
                german_indicators = ['ß', 'ä', 'ö', 'ü', 'sch', 'tz', 'pf', 'qu', 'ck']

                query_lower = query.lower()
                german_verbs_count = sum(1 for word in german_verbs if word in query_lower)
                german_question_words_count = sum(1 for word in german_question_words if word in query_lower.split())
                german_articles_count = sum(1 for word in german_articles if word in query_lower.split())
                german_prepositions_count = sum(1 for word in german_prepositions if word in query_lower.split())
                german_char_count = sum(1 for indicator in german_indicators if indicator in query_lower)

                total_german_score = german_verbs_count + german_question_words_count + german_articles_count + german_prepositions_count + german_char_count

                # Check German probability
                german_prob = next((prob.prob for prob in lang_probs if prob.lang == 'de'), 0)

                # Only override if we have clear German evidence and German has some probability
                if total_german_score >= 2 and german_prob > 0.05:
                    detected_lang = 'de'

            # Map to supported languages, fallback to English
            return detected_lang if detected_lang in self.prompt_templates else 'en'

        except LangDetectException:
            return 'en'

    def generate_cache_key(self, query: str, context_docs: List[Dict], query_lang: str, model: str, temp: float, max_tokens: int) -> str:
        """Generate deterministic cache key"""
        # Normalize query
        normalized_query = query.lower().strip()

        # Create document fingerprint (sort by ID for consistency)
        doc_ids = sorted([str(doc.get('id', '')) for doc in context_docs])
        docs_hash = hashlib.md5(','.join(doc_ids).encode()).hexdigest()[:12]

        # Include generation parameters
        params_str = f"{model}:{temp}:{max_tokens}:{query_lang}"
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        # Combine components
        key_components = [normalized_query, docs_hash, params_hash]
        full_hash = hashlib.md5(':'.join(key_components).encode()).hexdigest()

        return f"llm:{full_hash}"

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query (str): Search query
            top_k (int): Number of results to retrieve

        Returns:
            list: Retrieved document chunks with content and metadata
        """
        return self.retriever.retrieve(query, top_k)

    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using retrieved context and LLM with caching.

        Args:
            query (str): Original question
            context_docs (list): Retrieved document chunks

        Returns:
            str: Generated answer text
        """
        # Detect query language
        query_lang = self.detect_query_language(query)

        # Check cache first
        cache_key = None
        if self.cache_enabled and self.cache:
            cache_key = self.generate_cache_key(
                query, context_docs, query_lang,
                self.llm.model if hasattr(self.llm, 'model') else 'unknown',
                0.7, 500  # Default parameters - should come from settings
            )

            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.info(f"Cache hit for query: {query[:50]}...")
                # Increment hit counter
                cached_response['cache_hits'] = cached_response.get('cache_hits', 0) + 1
                self.cache.set(cache_key, cached_response)
                return cached_response['answer']

        # Combine context from retrieved documents with source information
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            doc_info = doc.get('document', {})
            filename = doc_info.get('filename', f'Document {i}')
            content = doc['content']
            source_ref = f"[Source {i}: {filename}]"
            context_parts.append(f"{source_ref}\n{content}")

        context = "\n\n".join(context_parts)

        # Get appropriate prompt template for the detected language
        prompt_template = self.prompt_templates.get(query_lang, self.default_template)

        # Create prompt
        prompt = prompt_template.format(context=context, question=query)

        # Generate answer
        try:
            answer = self.llm.invoke(prompt)

            # Cache the response
            if self.cache_enabled and self.cache:
                cache_entry = {
                    'answer': answer,
                    'query_language': query_lang,
                    'model_used': self.llm.model if hasattr(self.llm, 'model') else 'unknown',
                    'num_docs': len(context_docs),
                    'cache_hits': 0
                }
                self.cache.set(cache_key, cache_entry)

            return answer
        except Exception as e:
            return f"Error generating answer: {e}"

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute full RAG pipeline: retrieve relevant documents and generate answer.

        Args:
            question (str): Question to answer
            top_k (int): Number of documents to retrieve

        Returns:
            dict: Response containing question, answer, retrieved docs, and metadata
        """
        # Detect query language
        query_lang = self.detect_query_language(question)

        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, top_k)

        # Generate answer
        answer = self.generate_answer(question, retrieved_docs)

        return {
            'question': question,
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'num_docs': len(retrieved_docs),
            'query_language': query_lang
        }

    def invalidate_document_cache(self, document_id: str) -> int:
        """Invalidate all cache entries for a specific document"""
        if not self.cache_enabled or not self.cache:
            return 0

        # Clear cache entries that might contain this document
        # This is a broad invalidation - could be optimized with better tracking
        pattern = f"llm:*"
        return self.cache.clear_pattern(pattern)

    def invalidate_query_cache(self, query_pattern: str) -> int:
        """Invalidate cache entries matching query pattern"""
        if not self.cache_enabled or not self.cache:
            return 0

        return self.cache.clear_pattern(f"llm:*{query_pattern}*")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        if not self.cache_enabled or not self.cache:
            return {'cache_enabled': False}

        stats = self.cache.get_stats()
        stats['cache_enabled'] = True
        return stats

def format_results_db(results: List[Dict[str, Any]]) -> str:
    """Format retrieval results for display."""
    if not results:
        return "No relevant documents found."

    formatted = []
    for i, result in enumerate(results, 1):
        doc_info = result.get('document', {})
        formatted.append(f"""
**Document {i}:** {doc_info.get('filename', 'Unknown')}
**Relevance Score:** {result.get('score', 0):.3f}
**Content:** {result['content'][:200]}...
""")

    return "\n---\n".join(formatted)

def format_answer_db(answer: str) -> str:
    """Format generated answer for display."""
    return answer.strip()