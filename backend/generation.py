import os
from groq import Groq

# To allow relative imports if run as a script or module
try:
    from backend.config import settings
    from backend.knowledge_router import KnowledgeRouter
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.config import settings
    from backend.knowledge_router import KnowledgeRouter

client = Groq(api_key=settings.groq_api_key)


class GenerationManager:
    @staticmethod
    def generate_hyde_response(query: str) -> str:
        prompt = (
            "You are an educational counselor. "
            "Given the user's question, generate a brief, realistic, hypothetical paragraph "
            "that would answer this question. Do not provide a lengthy explanation, just the core facts "
            f"as they would appear in a document.\n\nQuestion: {query}"
        )
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.hyde_model,
                temperature=0.3,
                max_tokens=settings.hyde_max_tokens,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"HyDE generation failed: {e}")
            return query  # fallback

    @staticmethod
    def contextualize_query(query: str, chat_history: list) -> str:
        if not chat_history:
            return query

        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
        prompt = (
            "Given the following conversation history and the latest user query, "
            "which might reference earlier context (e.g., 'this course', 'it', 'there'), "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question, just return the standalone rewritten question.\n\n"
            f"Chat History:\n{history_text}\n\n"
            f"Latest Query: {query}\n\n"
            "Standalone Question:"
        )

        try:
            r = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.groq_model,
                temperature=0.1,
                max_tokens=settings.contextualize_max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"Contextualize query failed: {e}")
            return query

    @staticmethod
    def build_system_prompt(university_name: str, university_location: str) -> str:
        return (
            "You are an expert admission counselor assistant. You help students understand "
            f"university admissions across India. You are currently assisting with queries "
            f"about {university_name} ({university_location}).\n"
            "Always be accurate, warm, and student-friendly.\n"
            "Never fabricate specific numbers (fees, seats, ranks, deadlines) — only state "
            "what is confirmed in the provided documents.\n"
            "For general educational guidance, you may use your knowledge but label it clearly."
        )

    @staticmethod
    def construct_messages(
        query: str,
        mode: str,
        system_prompt: str,
        context_text: str,
        university_name: str,
        chat_history: list = None,
    ):
        messages = [{"role": "system", "content": system_prompt}]

        if chat_history:
            for msg in chat_history:
                messages.append(msg)

        ref_instruction = (
            "Answer the question naturally in plain prose or bullet points with no inline labels.\n"
            "If any documents from the Context were used to answer the question, you MUST append a References section at the very end of your response exactly like this:\n\n"
            "---\n"
            "**References**\n"
            "1. <Filename> — Page <Page Number>\n"
            "2. <Filename> — Page <Page Number>\n\n"
            "Only list document sources from the Context that were actually used in the answer. If no documents were used, omit the References section entirely."
        )

        if mode == KnowledgeRouter.MODE_RAG_ONLY:
            user_prompt = (
                f"{ref_instruction}\n"
                "Answer using ONLY the document context below.\n"
                "If the answer is not in the context, say so and provide the admissions contact.\n"
                f"Context: {context_text}\n"
                f"Question: {query}"
            )
        elif mode == KnowledgeRouter.MODE_LLM_ONLY:
            user_prompt = (
                f"The student is asking about {university_name}. No specific documents were found "
                "for this question. Answer the question naturally using your general knowledge about Indian higher education.\n"
                "Do not include a References section since no documents are provided.\n"
                f"Question: {query}"
            )
        else:  # BLEND
            user_prompt = (
                f"{ref_instruction}\n"
                f"Answer this question about {university_name}. Use the document context below for "
                "specific facts, and your general knowledge for explanatory or career-related parts.\n"
                f"Context: {context_text}\n"
                f"Question: {query}"
            )

        messages.append({"role": "user", "content": user_prompt})
        return messages

    @staticmethod
    def generate_response(
        query: str,
        mode: str,
        context_text: str,
        university_meta: dict,
        stream: bool = True,
        chat_history: list = None,
    ):
        university_name = university_meta.get("name", "the university")
        university_location = university_meta.get("location", "")

        system_prompt = GenerationManager.build_system_prompt(university_name, university_location)
        messages = GenerationManager.construct_messages(query, mode, system_prompt, context_text, university_name, chat_history)

        temperature = 0.2
        if mode == KnowledgeRouter.MODE_LLM_ONLY:
            temperature = 0.5
        elif mode == KnowledgeRouter.MODE_BLEND:
            temperature = 0.3

        try:
            return client.chat.completions.create(
                messages=messages,
                model=settings.groq_model,
                temperature=temperature,
                stream=stream,
            )
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            raise e
