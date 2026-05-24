import os
from google import genai
from google.genai import types

# To allow relative imports if run as a script or module
try:
    from backend.config import settings
    from backend.knowledge_router import KnowledgeRouter
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.config import settings
    from backend.knowledge_router import KnowledgeRouter

# Shared client — created once at module load time
_client = genai.Client(api_key=settings.gemini_api_key)


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
            response = _client.models.generate_content(
                model=settings.gemini_lite_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=settings.hyde_max_tokens,
                ),
            )
            return response.text
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
            response = _client.models.generate_content(
                model=settings.gemini_lite_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=settings.contextualize_max_tokens,
                ),
            )
            return response.text.strip()
        except Exception as e:
            print(f"Contextualize query failed: {e}")
            return query

    @staticmethod
    def build_system_prompt(university_name: str, university_location: str) -> str:
        return (
            "You are a concise, accurate university admission counselor for Indian higher education, "
            f"currently helping with queries about {university_name}"
            + (f" ({university_location})" if university_location else "") + ".\n"
            "Answer style rules — follow these strictly:\n"
            "- Keep answers to 2–5 sentences. Be direct and precise, like a knowledgeable counselor speaking to a student.\n"
            "- NO greetings (do not start with 'Hello', 'Hi', 'Great question', etc.).\n"
            "- NO markdown headers (##, ###). No numbered section titles like '1. Overview'.\n"
            "- Plain prose only. You may use a short bullet list (3–5 items max) only when listing multiple distinct items (e.g., colleges, subjects).\n"
            "- State the key fact first, then add one supporting detail or caveat if needed.\n"
            "- If a specific number (fee, seat count) is from documents, cite it. If from general knowledge, give the figure confidently without hedging."
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
        # System prompt is passed via config.system_instruction — not part of contents
        messages = []

        if chat_history:
            for msg in chat_history:
                role = "model" if msg["role"] == "assistant" else "user"
                messages.append({"role": role, "parts": [{"text": msg["content"]}]})

        ref_instruction = (
            "If you used a specific document from the Context, append at the very end:\n"
            "---\n**References**\n1. <Filename> — Page <N>\n"
            "Only include sources actually used. Omit the section entirely if none were used."
        )

        base_style = (
            "Answer in 2–5 sentences. Start directly with the answer — no greetings or preamble. "
            "Plain prose only, no markdown headers. A short bullet list is allowed only when listing 3+ distinct items."
        )

        if mode == KnowledgeRouter.MODE_RAG_ONLY:
            user_prompt = (
                f"{base_style}\n{ref_instruction}\n"
                "Use the document Context for specific facts first. "
                "If the fact is not in the Context, answer from your general knowledge about Indian admissions and CUET — never say it is unavailable.\n"
                f"Context: {context_text}\n"
                f"Question: {query}"
            )
        elif mode == KnowledgeRouter.MODE_LLM_ONLY:
            user_prompt = (
                f"{base_style}\n"
                "Answer using your knowledge of Indian universities, CUET, DU, and admissions. No References section.\n"
                f"Question: {query}"
            )
        else:  # BLEND
            user_prompt = (
                f"{base_style}\n{ref_instruction}\n"
                "Use the document Context for specific facts; use your general knowledge for anything not covered.\n"
                f"Context: {context_text}\n"
                f"Question: {query}"
            )

        messages.append({"role": "user", "parts": [{"text": user_prompt}]})
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
        contents = GenerationManager.construct_messages(
            query, mode, system_prompt, context_text, university_name, chat_history
        )

        temperature = 0.2
        if mode == KnowledgeRouter.MODE_LLM_ONLY:
            temperature = 0.5
        elif mode == KnowledgeRouter.MODE_BLEND:
            temperature = 0.3

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        )

        try:
            if stream:
                return _client.models.generate_content_stream(
                    model=settings.gemini_model,
                    contents=contents,
                    config=config,
                )
            else:
                return _client.models.generate_content(
                    model=settings.gemini_model,
                    contents=contents,
                    config=config,
                )
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise e
