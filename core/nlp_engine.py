import spacy
from spacy.matcher import PhraseMatcher
'''
Spacy Fine Tuning Layer
'''
class NLP_Analysis():
    def __init__(self):

        # Load once — small model, no GPU needed
        self.nlp = spacy.load("en_core_web_sm")
        
        # ── Domain maps — extend these as your classification vocabulary grows ──
        self.STATE_KEYWORDS = {
            # Braking
            "understeer":        "Trail_Braking",
            "braking":           "Hard_Braking",
            "brake":             "Hard_Braking",
            "trail":             "Trail_Braking",
            "oversteer":         "WOT",            # often power-related
            "push":              "Trail_Braking",  # driver vernacular
            # Throttle
            "throttle":          "Partial_Throttle",
            "wot":               "WOT",
            "full throttle":     "WOT",
            "traction":          "Partial_Throttle",
            "acceleration":      "WOT",
            "accelerating":      "WOT",
            # Coasting / transitions
            "coast":             "Coasting",
            "coasting":          "Coasting",
            "lift":              "Coasting",
            "transition":        "Aggressive_Transition",
            # Thermal
            "temperature":       "thermal",
            "overheat":          "thermal",
            "oil":               "thermal",
            "coolant":           "thermal",
            "thermal":           "thermal",
            # Performance
            "engine":            "engine_performance",
            "rpm":               "engine_performance",
            "performance":       "engine_performance",
            "gear":              "engine_performance",
            "speed":             "engine_performance",
        }
        
        self.SESSION_WORDS = {"session", "run", "outing", "log"}
    
    def extract_intent(self, query: str) -> dict:
        """
        Maps a natural language query to structured retrieval parameters.
        Returns states, sessions, and a raw domain hint for field set selection.
        """
        doc = self.nlp(query.lower())
        tokens = [t.lemma_ for t in doc] #lemmatisation
        text   = query.lower()
    
        # States: match lemmas against domain map
        states = []
        for token in tokens:
            if token in self.STATE_KEYWORDS:
                val = self.STATE_KEYWORDS[token]
                if val not in states:
                    states.append(val)
    
        # Sessions: "session N"/ordinal patterns
        sessions = []
        for i, token in enumerate(doc):
            if token.text in self.SESSION_WORDS:
                # look ahead up to 3 tokens for numbers 
                for j in range(i + 1, min(i + 4, len(doc))):
                    next_tok = doc[j]
                    if next_tok.like_num:
                        sessions.append(next_tok.text)
                        break
            # Session_words looking for language relatng to comparisons
            if token.like_num and i > 0:
                prev = doc[i - 1].text
                if prev in self.SESSION_WORDS or prev in ('and', ',', '&'):
                    if token.text not in sessions:
                        sessions.append(token.text)
    
        sessions = list(dict.fromkeys(sessions)) 
    
        # Domain: FIELD_SET selection in stg_tokeniser
        domain = "default"
        if any(s in ["Hard_Braking", "Trail_Braking"] for s in states):
            domain = "braking"
        elif any(s in ["WOT", "Partial_Throttle"] for s in states):
            domain = "throttle"
        elif "thermal" in states:
            domain = "thermal"
        elif "engine_performance" in states:
            domain = "engine"
    
        intent = {
            "states":   states   if states   else None,
            "sessions": sessions if sessions else None,
            "domain":   domain,
            "comparison": len(sessions) > 1,
            "raw":      query,
        }
    
        return intent
