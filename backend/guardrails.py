import re

class Guardrails:
    OFF_TOPIC_KEYWORDS = [
        "movie", "movies", "politics", "election", "cooking", "recipe", 
        "weather", "sports", "cricket", "football", "celebrity"
    ]
    
    PII_PATTERNS = [
        # Basic emails
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
        # Basic Phone numbers (India/Generic)
        r'\+?91[\-\s]?\d{10}',
        r'\b\d{10}\b'
    ]

    @staticmethod
    def is_off_topic(query: str) -> bool:
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in Guardrails.OFF_TOPIC_KEYWORDS):
            return True
        return False
        
    @staticmethod
    def construct_off_topic_response(university_name: str) -> str:
        return (f"I am an educational counselor assistant for {university_name}. "
                "I cannot answer questions about movies, politics, or other unrelated topics. "
                "Please ask me about admissions, courses, fees, or campus life!")

    @staticmethod
    def strip_pii(query: str) -> str:
        sanitized = query
        for pattern in Guardrails.PII_PATTERNS:
            sanitized = re.sub(pattern, "[REDACTED]", sanitized)
        return sanitized

    @staticmethod
    def verify_numbers_in_rag(response_text: str, context_text: str) -> str:
        """
        Extracts numbers from response and verifies they exist in the context.
        If a number is found in response but not in context, append a warning.
        """
        response_numbers = set(re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', response_text))
        if not response_numbers:
            return response_text
            
        # Strip commas for pure numeric checks conceptually, but simple existence check first
        for num in response_numbers:
            # Simple substring check (can be improved in production)
            if num not in context_text and num.replace(",", "") not in context_text.replace(",", ""):
                return response_text + "\n\n> **Note:** The exact numbers mentioned above might not be explicitly present in my current document database. Please verify these figures directly with the university."
                
        return response_text

    @staticmethod
    def append_contact_info(response_text: str, email: str, phone: str, website: str) -> str:
        contact_str = f"\n\n---\n**Admissions Contact:**\nEmail: {email or 'N/A'} | Phone: {phone or 'N/A'}\nWebsite: {website or 'N/A'}"
        if "Admissions Contact:" not in response_text:
            return response_text + contact_str
        return response_text
