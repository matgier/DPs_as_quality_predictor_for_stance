
"""
Semantic Analysis Package - Multi-API Support
Pakiet do analizy semantycznej obsługujący Claude, Gemini i ChatGPT
"""


__version__ = "2.0.0"

# Główne funkcje API
__all__ = [
    # Główne funkcje
    "extract_semantic_concepts_with_agents",
    "save_semantic_features", 
    "load_semantic_features",
    
    # Klasy koordynujące
    "AgentOrchestrator",
    "compare_providers",
    
    # Fabryka providerów
    "ProviderFactory",
    "get_provider_status_report",
    
    # Klasy bazowe
    "ModelConfig",
    "APIProvider"
]

# Informacje o dostępnych providerach
SUPPORTED_PROVIDERS = {
    "claude": {
        "name": "Anthropic Claude",
        "models": ["claude-3-7-sonnet-20250219", "claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"],
        "install": "pip install anthropic"
    },
    "gemini": {
        "name": "Google Gemini", 
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
        "install": "pip install google-generativeai"
    },
    "openai": {
        "name": "OpenAI ChatGPT",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "install": "pip install openai"
    }
}


def get_provider_info():
    """Zwróć informacje o obsługiwanych providerach"""
    return SUPPORTED_PROVIDERS

def check_setup():
    """Sprawdź status konfiguracji wszystkich providerów"""
    return get_provider_status_report()



"""
Moduł zawierający klasę bazową i interfejsy dla agentów semantycznych
obsługujących multiple API providers (Claude, Gemini, ChatGPT)
"""

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Konfiguracja modelu dla różnych providerów"""
    provider: str
    model_name: str
    max_tokens: int = 10000
    temperature: float = 0.0
    api_key: str = ""


class APIProvider(ABC):
    """Abstrakcyjny interface dla providerów API"""
    
    @abstractmethod
    def call_api(self, prompt: str, system_prompt: str, config: ModelConfig) -> str:
        """Wywołanie API z danym promptem i konfiguracją"""
        pass
    
    @abstractmethod
    def get_model_mapping(self) -> Dict[str, str]:
        """Mapowanie modeli Claude na modele danego providera"""
        pass


class SemanticAgent:
    """Bazowa klasa dla agentów semantycznych obsługująca multiple providers"""
    
    def __init__(self, config: ModelConfig, role: str = "generic", provider_instance: APIProvider = None):
        self.config = config
        self.role = role
        self.system_prompt = self._create_system_prompt()
        self.provider = provider_instance
        
        if not self.provider:
            self.provider = ProviderFactory.create_provider(config.provider)
    
    def _create_system_prompt(self) -> str:
        """Tworzy prompt systemowy specyficzny dla roli agenta"""
        prompts = {
            "reader": """You are a specialized Reading Agent with exceptional skills in comprehending and extracting key information from text.
Your task is to identify the most important passages, recurring themes, and significant concepts in the provided text.
Focus on finding passages that contain rich semantic information and could serve as the foundation for domain identification.
Provide only direct quotes from the text, ensuring they capture the core meaning and key concepts.""",
            
            "domain_extractor": """You are a specialized Domain Extraction Agent with expertise in identifying thematic domains in text.
Your task is to analyze the provided key passages and identify distinct thematic domains that are explicitly present.
For each domain, provide clear justification using direct references to the text.
Focus only on domains with substantial textual evidence, not implied or assumed domains.
Be precise and selective, prioritizing quality of domains over quantity.""",
            
            "polarizer": """You are a specialized Polarization Agent with expertise in identifying opposing aspects within domains.
Your task is to analyze each domain and find genuine bipolar representations in the text for MDS analysis.
For each domain, identify two clearly opposing aspects that represent opposite ends of a meaningful semantic dimension.
Ensure the polarization is authentic, naturally emerges from the text, and creates a true continuum for positioning texts.
Be specific and precise in identifying opposing characteristics that will MINIMIZE
 variance in MDS.""",
            
            "definer": """You are a specialized Definition Agent with expertise in creating precise semantic definitions for MDS analysis.
Your task is to formulate concise, comprehensive definitions (1 sentence) for each polar aspect.
Use language and terminology directly from the text, avoiding external concepts.
Each definition must be clearly distinct from others, directly opposed to its counterpart, and backed by specific text references.
Your output MUST contain a 'semantic_features' dictionary with each aspect name mapped to a list containing its definition.
Focus on clarity, precision, and creating definitions that represent clear endpoints on a semantic continuum.""",
        
            "verifier": """You are a specialized Verification Agent with expertise in critical review and validation.
Your task is to critically evaluate the semantic analysis results for accuracy, fidelity to the original text, and suitability for MDS analysis.
Verify that each domain and aspect exists in the text, polarization creates true semantic dimensions, and definitions represent clear endpoints.
Challenge any extractions that appear forced, redundant, or unsupported by text evidence.
Ensure all semantic features are optimized for positioning texts along a meaningful continuum.
If semantic features are missing or inadequate, you must create proper definitions based on available evidence.""",
            
            "mds_optimizer": """You are a specialized MDS Optimization Agent with expertise in creating semantic dimensions for text analysis.
Your task is to refine semantic definitions to ensure they represent true opposites on a meaningful continuum.
Focus on creating clear, distinct dimensions that will MINIMIZE variance when positioning texts in semantic space.
Each definition must be precise, comprehensive, and directly related to its opposing counterpart.
Ensure all dimensions are independent, meaningful, and capture significant variation in potential texts."""
        }
        
        return prompts.get(self.role, 
            """You are a specialized Semantic Analysis Agent with expertise in extracting concepts from text.
Follow instructions precisely and provide detailed justification for your analysis.""")
    
    def process(self, text: str, previous_results: Dict = None) -> Dict:
        """Przetwarzanie tekstu przez agenta - implementacja w klasach pochodnych"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def call_model(self, prompt: str, temperature: float = None) -> str:
        """Ujednolicone wywołanie API dla wszystkich providerów"""
        if temperature is not None:
            config = ModelConfig(
                provider=self.config.provider,
                model_name=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=temperature,
                api_key=self.config.api_key
            )
        else:
            config = self.config
            
        return self.provider.call_api(prompt, self.system_prompt, config)
    
    def extract_python_dict(self, response: str) -> Dict:
        """Extracts a Python dictionary from the response with improved handling for semantic_features"""
        try:
            # Pierwszy podejście: szukamy "semantic_features = {...}"
            dict_match = re.search(r'semantic_features\s*=\s*{(.*?)}(?=\s*$|\s*```|$)', response, re.DOTALL)
            if dict_match:
                dict_str = "{" + dict_match.group(1) + "}"
                
                # Oczyszczamy znalezione dane
                dict_str = re.sub(r'""".*?"""', '""', dict_str, flags=re.DOTALL)
                dict_str = re.sub(r"'''.*?'''", "''", dict_str, flags=re.DOTALL)
                
                # Naprawiamy format definicji wewnątrz list
                dict_str = re.sub(r"'([^']+)': \['([^']+)'\]", r"'\1': ['\2',]", dict_str)
                dict_str = re.sub(r'"([^"]+)": \["([^"]+)"\]', r'"\1": ["\2",]', dict_str)
                
                try:
                    import ast
                    return ast.literal_eval(dict_str)
                except Exception as e:
                    print(f"Failed to evaluate with ast: {e}")
                    
                    # Alternatywne podejście - konwersja na format JSON
                    fixed_str = re.sub(r"'([^']+)':", r'"\1":', dict_str)
                    fixed_str = re.sub(r"\['([^']+)'\]", r'["\1"]', fixed_str)
                    
                    try:
                        return json.loads(fixed_str)
                    except Exception as json_e:
                        print(f"Failed to parse JSON: {json_e}")
            
            # Drugie podejście: szukamy bloków kodu JSON/Python
            code_block_match = re.search(r'```(?:python|json)?\s*({.*?})\s*```', response, re.DOTALL)
            if code_block_match:
                dict_str = code_block_match.group(1).strip()
                try:
                    import ast
                    return ast.literal_eval(dict_str)
                except:
                    try:
                        fixed_str = re.sub(r"'([^']+)':", r'"\1":', dict_str)
                        fixed_str = re.sub(r"\['([^']+)'\]", r'["\1"]', fixed_str)
                        return json.loads(fixed_str)
                    except:
                        pass
            
            # Trzecie podejście: szukamy słownika w formatowaniu inline
            inline_dict_match = re.search(r'{(?:[^{}]|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\[(?:[^\[\]]|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')*\])*}', response)
            if inline_dict_match:
                dict_str = inline_dict_match.group(0)
                try:
                    import ast
                    return ast.literal_eval(dict_str)
                except:
                    pass
            
            # Czwarte podejście: ręczne wydobycie par klucz-wartość
            pairs = re.findall(r"['\"]([\w_]+)['\"]:\s*\[\s*['\"](.+?)['\"](?:,|\s*\])", response, re.DOTALL)
            if pairs:
                result = {}
                for key, value in pairs:
                    result[key] = [value]
                return result
            
            return {}
            
        except Exception as e:
            print(f"Error in extract_python_dict: {e}")
            return {}
    
    def verify_definitions_quality(self, semantic_features: Dict, polarized_domains: List[Dict]) -> List[str]:
        """Weryfikuje jakość definicji pod kątem ich przydatności do MDS"""
        issues = []
        
        for domain in polarized_domains:
            pole1_name = domain.get("pole1", {}).get("name", "")
            pole2_name = domain.get("pole2", {}).get("name", "")
            
            # Sprawdź, czy obie definicje istnieją
            if pole1_name not in semantic_features:
                issues.append(f"Brakująca definicja dla: {pole1_name}")
            if pole2_name not in semantic_features:
                issues.append(f"Brakująca definicja dla: {pole2_name}")
                continue
                
            # Sprawdź długość definicji
            if pole1_name in semantic_features and len(semantic_features[pole1_name][0]) < 30:
                issues.append(f"Definicja dla {pole1_name} jest zbyt krótka")
            if pole2_name in semantic_features and len(semantic_features[pole2_name][0]) < 30:
                issues.append(f"Definicja dla {pole2_name} jest zbyt krótka")
            
            # Sprawdź przeciwstawność definicji
            if pole1_name in semantic_features and pole2_name in semantic_features:
                pole1_def = semantic_features[pole1_name][0].lower()
                pole2_def = semantic_features[pole2_name][0].lower()
                common_words = set(pole1_def.split()) & set(pole2_def.split())
                if len(common_words) < 3:
                    issues.append(f"Definicje dla {pole1_name} i {pole2_name} mogą nie być wystarczająco powiązane")
        
        return issues
    
    def assess_definitions_with_model(self, semantic_features: Dict) -> str:
        """Używa modelu do oceny jakości definicji dla MDS"""
        features_info = ""
        for key, value in semantic_features.items():
            features_info += f"- {key}: {value[0]}\n"
        
        prompt = f"""
        # MDS Definition Quality Assessment
        
        Critically evaluate the following semantic definitions for their suitability in Multidimensional Scaling (MDS) analysis:
        
        {features_info}
        
        For each pair of definitions:
        1. Assess whether they truly represent opposite ends of a single semantic dimension
        2. Evaluate if they are specific enough to differentiate texts
        3. Check if they avoid overlap with other dimensions
        4. Determine if they would capture meaningful variance across texts
        
        Provide a detailed assessment for each dimension, rating them on:
        - Bipolarity (how well they represent true opposites): 1-5
        - Specificity (how precisely defined they are): 1-5
        - Independence (how distinct from other dimensions): 1-5
        - Measurability (how easily texts can be positioned): 1-5
        
        Return your assessment as a structured evaluation.
        """
        
        return self.call_model(prompt, temperature=0)
    



"""
Provider dla Anthropic Claude API
"""

import anthropic
import time
from typing import Dict


class ClaudeProvider(APIProvider):
    """Provider dla Anthropic Claude API"""
    
    def __init__(self):
        self.client = None
    
    def _initialize_client(self, api_key: str):
        """Inicjalizuje klienta Claude jeśli nie został jeszcze utworzony"""
        if not self.client:
            self.client = anthropic.Anthropic(api_key=api_key)
    
    def call_api(self, prompt: str, system_prompt: str, config: ModelConfig) -> str:
        """Wywołuje Claude API"""
        self._initialize_client(config.api_key)
        
        try:
            response = self.client.messages.create(
                model=config.model_name,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
            
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            # Ponów próbę po opóźnieniu
            time.sleep(2)
            try:
                response = self.client.messages.create(
                    model=config.model_name,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            except Exception as e:
                print(f"Claude API second attempt failed: {e}")
                return None
    
    def get_model_mapping(self) -> Dict[str, str]:
        """Mapowanie modeli Claude (bez mapowania - używamy oryginalnych nazw)"""
        return {
            "claude-3-7-sonnet-20250219": "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307"
        }
    
    @staticmethod
    def is_available() -> bool:
        """Sprawdza czy biblioteka Claude jest dostępna"""
        try:
            import anthropic
            return True
        except ImportError:
            print("Anthropic library not installed. Install with: pip install anthropic")
            return False
        

"""
Provider dla Google Gemini API
"""

import time
from typing import Dict

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiProvider(APIProvider):
    """Provider dla Google Gemini API"""
    
    def __init__(self):
        self.configured = False
    
    def _configure_gemini(self, api_key: str):
        """Konfiguruje Gemini API"""
        if not self.configured and GEMINI_AVAILABLE:
            genai.configure(api_key=api_key)
            self.configured = True
    
    def call_api(self, prompt: str, system_prompt: str, config: ModelConfig) -> str:
        """Wywołuje Gemini API"""
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI library not installed. Install with: pip install google-generativeai")
        
        self._configure_gemini(config.api_key)
        
        try:
            # Tworzenie modelu z instrukcją systemową
            model = genai.GenerativeModel(
                model_name=config.model_name,
                system_instruction=system_prompt
            )
            
            # Konfiguracja generacji
            generation_config = genai.types.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
            )
            
            # Generowanie odpowiedzi
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Ponów próbę po opóźnieniu
            time.sleep(2)
            try:
                model = genai.GenerativeModel(
                    model_name=config.model_name,
                    system_instruction=system_prompt
                )
                
                generation_config = genai.types.GenerationConfig(
                    temperature=config.temperature,
                    max_output_tokens=config.max_tokens,
                )
                
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                return response.text
            except Exception as e:
                print(f"Gemini API second attempt failed: {e}")
                return None
    
    def get_model_mapping(self) -> Dict[str, str]:
        """Mapowanie modeli Claude na modele Gemini"""
        return {
            "claude-3-7-sonnet-20250219": "gemini-1.5-pro",
            "claude-3-5-sonnet": "gemini-1.5-pro",
            "claude-3-opus": "gemini-1.5-pro", 
            "claude-3-haiku": "gemini-1.5-flash",
            # Bezpośrednie mapowania Gemini
            "gemini-1.5-pro": "gemini-1.5-pro",
            "gemini-1.5-flash": "gemini-1.5-flash",
            "gemini-1.0-pro": "gemini-1.0-pro"
        }
    
    @staticmethod
    def is_available() -> bool:
        """Sprawdza czy biblioteka Gemini jest dostępna"""
        return GEMINI_AVAILABLE
    

"""
Provider dla OpenAI ChatGPT API
"""

import time
from typing import Dict

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIProvider(APIProvider):
    """Provider dla OpenAI ChatGPT API"""
    
    def __init__(self):
        self.client = None
    
    def _initialize_client(self, api_key: str):
        """Inicjalizuje klienta OpenAI jeśli nie został jeszcze utworzony"""
        if not self.client and OPENAI_AVAILABLE:
            self.client = openai.OpenAI(api_key=api_key)
    
    def call_api(self, prompt: str, system_prompt: str, config: ModelConfig) -> str:
        """Wywołuje OpenAI API"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        
        self._initialize_client(config.api_key)
        
        try:
            # Przygotowanie wiadomości z system prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Ponów próbę po opóźnieniu
            time.sleep(2)
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                response = self.client.chat.completions.create(
                    model=config.model_name,
                    messages=messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
                
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API second attempt failed: {e}")
                return None
    
    def get_model_mapping(self) -> Dict[str, str]:
        """Mapowanie modeli Claude na modele OpenAI"""
        return {
            "claude-3-7-sonnet-20250219": "gpt-4o",
            "claude-3-5-sonnet": "gpt-4o",
            "claude-3-opus": "gpt-4o",
            "claude-3-haiku": "gpt-4o-mini",
            # Bezpośrednie mapowania OpenAI
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-4-turbo": "gpt-4-turbo-preview",
            "gpt-3.5-turbo": "gpt-3.5-turbo"
        }
    
    @staticmethod
    def is_available() -> bool:
        """Sprawdza czy biblioteka OpenAI jest dostępna"""
        return OPENAI_AVAILABLE
    


"""
Fabryka providerów API - centralne miejsce tworzenia i zarządzania providerami
"""

from typing import Dict, Optional


class ProviderFactory:
    """Fabryka do tworzenia providerów API"""
    
    _providers = {
        "claude": ClaudeProvider,
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "chatgpt": OpenAIProvider,  # Alias dla OpenAI
        "gpt": OpenAIProvider       # Alias dla OpenAI
    }
    
    @classmethod
    def create_provider(cls, provider_name: str) -> APIProvider:
        """Tworzy instancję providera na podstawie nazwy"""
        provider_name = provider_name.lower()
        
        if provider_name not in cls._providers:
            available_providers = list(cls._providers.keys())
            raise ValueError(f"Nieobsługiwany provider: {provider_name}. "
                           f"Dostępne providery: {available_providers}")
        
        provider_class = cls._providers[provider_name]
        
        # Sprawdź dostępność biblioteki
        if hasattr(provider_class, 'is_available') and not provider_class.is_available():
            raise ImportError(f"Provider {provider_name} nie jest dostępny. "
                            f"Sprawdź czy odpowiednia biblioteka jest zainstalowana.")
        
        return provider_class()
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """Zwraca listę dostępnych providerów i ich status"""
        status = {}
        for name, provider_class in cls._providers.items():
            if hasattr(provider_class, 'is_available'):
                status[name] = provider_class.is_available()
            else:
                status[name] = True  # Assume available if no check method
        return status
    
    @classmethod
    def create_model_config(cls, provider: str, api_key: str, 
                          model: str = None, max_tokens: int = 10000, 
                          temperature: float = 0.0) -> ModelConfig:
        """Tworzy konfigurację modelu z automatycznym mapowaniem"""
        provider_instance = cls.create_provider(provider)
        model_mapping = provider_instance.get_model_mapping()
        
        # Jeśli nie podano modelu, użyj domyślnego dla danego providera
        if model is None:
            if provider.lower() in ["claude"]:
                model = "claude-3-7-sonnet-20250219"
            elif provider.lower() in ["gemini"]:
                model = "gemini-1.5-pro"
            elif provider.lower() in ["openai", "chatgpt", "gpt"]:
                model = "gpt-4o"
        
        # Mapuj model jeśli potrzeba
        mapped_model = model_mapping.get(model, model)
        
        return ModelConfig(
            provider=provider.lower(),
            model_name=mapped_model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key
        )
    
    @classmethod
    def register_provider(cls, name: str, provider_class):
        """Rejestruje nowy provider"""
        if not issubclass(provider_class, APIProvider):
            raise TypeError("Provider musi dziedziczyć po APIProvider")
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def list_providers(cls) -> list:
        """Lista wszystkich zarejestrowanych providerów"""
        return list(cls._providers.keys())


def get_provider_status_report() -> str:
    """Generuje raport statusu wszystkich providerów"""
    status = ProviderFactory.get_available_providers()
    
    # Usuń duplikaty (aliasy)
    unique_providers = {}
    for name, available in status.items():
        if name not in ["chatgpt", "gpt"]:  # Pomijamy aliasy
            unique_providers[name] = available
    
    report = "=== Status Providerów API ===\n"
    for provider, available in unique_providers.items():
        status_text = "✓ Dostępny" if available else "✗ Niedostępny"
        report += f"{provider.capitalize()}: {status_text}\n"
    
    report += "\nAby zainstalować brakujące biblioteki:\n"
    if not status.get("claude", True):
        report += "- Claude: pip install anthropic\n"
    if not status.get("gemini", True):
        report += "- Gemini: pip install google-generativeai\n"
    if not status.get("openai", True):
        report += "- OpenAI: pip install openai\n"
    
    return report

"""
Implementacje konkretnych agentów semantycznych
"""

import json
import re
from typing import Dict, List


class ReadingAgent(SemanticAgent):
    """Agent odpowiedzialny za wstępne czytanie i identyfikację kluczowych fragmentów"""
    
    def __init__(self, config: ModelConfig, provider_instance=None):
        super().__init__(config, role="reader", provider_instance=provider_instance)
    
    def process(self, text: str, previous_results: Dict = None, context_len: int = 10) -> Dict:
        print(f"Processing {context_len} key passages")

        prompt = f"""
        # Reading Analysis Task
        
        Carefully analyze the following text and extract {context_len} key passages that contain the most important semantic information.
        These passages will be used to identify thematic domains for MDS analysis, so focus on selecting fragments that:
        1. Represent distinct concepts or themes that could form meaningful dimensions
        2. Contain rich semantic content that shows potential for polarization
        3. Reveal potential polarities or contrasting viewpoints that could be positioned on a continuum
        4. Cover the main ideas present in the text that would be useful for text classification
        
        Return your analysis as a JSON object with the following structure:
        ```json
        {{
            "key_passages": [
                "Direct quote 1 from the text",
                "Direct quote 2 from the text",
                ...
            ],
            "main_themes": [
                "Brief description of theme 1 that could form a semantic dimension",
                "Brief description of theme 2 that could form a semantic dimension",
                ...
            ]
        }}
        ```
        
        Text to analyze:
        
        {text}
        """
        
        response = self.call_model(prompt, temperature=0)
        print("Raw Reading Agent response:")
        print(response)
        
        # Ekstrahuj JSON z odpowiedzi
        try:
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_match = re.search(r'({.*})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    return {"error": "Could not extract JSON from response", "raw_response": response}
            
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw response: {response}")
            return {"error": "JSON decode error", "raw_response": response}


class DomainExtractionAgent(SemanticAgent):
    """Agent odpowiedzialny za ekstrakcję domen tematycznych"""
    
    def __init__(self, config: ModelConfig, provider_instance=None):
        super().__init__(config, role="domain_extractor", provider_instance=provider_instance)
    
    def process(self, text: str, previous_results: Dict, defined_semantic_quialities = None) -> Dict:
        key_passages = previous_results.get("key_passages", [])
        key_passages_text = "\n\n".join([f"- {passage}" for passage in key_passages])
        
        
        if defined_semantic_quialities is not None:
            prompt = f"""
            # Domain Extraction Task for MDS Analysis
                        
            Based on the key passages identified from the text, analyze how they relate to the following PREDEFINED SEMANTIC DOMAINS: {defined_semantic_quialities}
                        
            These are the key passages extracted from the text:
                        
            {key_passages_text}
                        
            For each domain specified in PREDEFINED SEMANTIC DOMAIN:
            1. Provide a clear analysis of how this domain applies to the text
            2. Justify why this is a significant domain using direct references to the text
            3. Identify which key passages support this domain
            4. Consider how this domain could form a meaningful continuum for positioning texts
            5. Extract EXACTLY 1 MAIN thematic domainS

            Return your analysis as a JSON object with the following structure:
            ```json
            {{
                "domains": [
                    {{
                        "name": "domain1",
                        "justification": "Explanation of why this is a significant domain for MDS analysis",
                        "supporting_passages": [
                            "Supporting passage 1",
                            "Supporting passage 2"
                        ]
                    }},
                    ...
                ]
            }}
            ```

            """
            
        else:
            prompt = f"""
            # Domain Extraction Task for MDS Analysis
            
            Based on the key passages identified from the text, extract EXACTLY 1 MAIN thematic domains that would serve as effective dimensions for MDS analysis.
            
            These are the key passages extracted from the text:
            
            {key_passages_text}
            
            For each domain you identify:
            1. Provide a clear, concise name for the domain that reflects a potential dimension for scaling texts
            2. Justify why this is a significant domain using direct references to the text
            3. Identify which key passages support this domain
            4. Consider how this domain could form a meaningful continuum for positioning texts
            5. Extract EXACTLY 1 MAIN thematic domains
            
            Return your analysis as a JSON object with the following structure:
            ```json
            {{
                "domains": [
                    {{
                        "name": "domain1",
                        "justification": "Explanation of why this is a significant domain for MDS analysis",
                        "supporting_passages": [
                            "Supporting passage 1",
                            "Supporting passage 2"
                        ]
                    }},
                    ...
                ]
            }}
            ```
            
            Important: 
            - Only identify domains that are explicitly present in the text with substantial evidence
            - Choose domains that would be useful for differentiating and positioning texts in a semantic space
            - Ensure domains are independent enough to serve as distinct dimensions in MDS
            - Select domains that naturally allow for polarization or scaling along a continuum
            """
            
        response = self.call_model(prompt, temperature=0)
        print("Raw Domain Extraction Agent response:")
        print(response)
        
        try:
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_match = re.search(r'({.*})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    return {"error": "Could not extract JSON from response", "raw_response": response}
            
            result = json.loads(json_str)
            
            # Połącz z poprzednimi wynikami
            result.update({
                "key_passages": previous_results.get("key_passages", []),
                "main_themes": previous_results.get("main_themes", [])
            })
            
            return result
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw response: {response}")
            return {"error": "JSON decode error", "raw_response": response}


class PolarizationAgent(SemanticAgent):
    """Agent odpowiedzialny za tworzenie skrajnie przeciwstawnych kategorii dla MDS"""
    
    def __init__(self, config: ModelConfig, provider_instance=None):
        super().__init__(config, role="polarizer", provider_instance=provider_instance)
    
    def process(self, text: str, previous_results: Dict) -> Dict:
        domains = previous_results.get("domains", [])
        
        domain_info = ""
        for i, domain in enumerate(domains):
            domain_name = domain.get("name", f"Domain {i+1}")
            justification = domain.get("justification", "")
            supporting_passages = domain.get("supporting_passages", [])
            
            domain_info += f"\n## Domain: {domain_name}\n"
            domain_info += f"Justification: {justification}\n"
            domain_info += "Supporting passages:\n"
            for passage in supporting_passages:
                domain_info += f"- {passage}\n"
        

        #  - First pole: "domain_beneficial", or "domain_positive", or "domain_strong", or "domain_effective"
        #    - Second pole: "domain_harmful", or "domain_negative", or "domain_weak", or "domain_ineffective"
        prompt = f"""
        # Extreme Polarization Task for MDS Analysis
        
        For each identified domain, create two MINIMALY CONTRASTING categories that will serve as perfect anchors for MDS analysis. These categories must create the clearest possible discrimination between texts.
        
        Here are the domains identified from the text:
        
        {domain_info}
        
        For each domain:
        1. Identify two categories that represent the absolute extremes of the semantic dimension
        2. Ensure these categories are PERFECTLY OPPOSITE in semantic space - they should represent the MINIMUM possible semantic distance
        3. Choose categories that will be easy to identify in texts - with distinctive vocabulary and clear signals
        4. Create categories where it would be nearly impossible for a text to score highly on both simultaneously
        5. Use EXACTLY the following naming convention for aspects:
           - First pole: "domain_base"
           - Second pole: "domain_opposite"
        6. Provide direct textual evidence for each pole from the original text
        
        CRUCIAL FOR MDS ANALYSIS: The polarities MUST:
        - Create perfect semantic anchors at opposite ends of a dimension
        - Have distinctive, non-overlapping vocabulary that embeddings can clearly differentiate
        - Represent clear conceptual opposites that would position texts along a meaningful continuum
        - Be formulated to MINIMIZE variance in how texts would score on this dimension
        - IMPORTANT: The dimension would capture MINIMUM variance in MDS analysis 

        Return your analysis as a JSON object with the following structure:
        ```json
        {{
            "polarized_domains": [
                {{
                    "domain": "domain1",
                    "pole1": {{
                        "name": "domain1_base",
                        "evidence": [
                            "Direct quote supporting this beneficial aspect"
                        ]
                    }},
                    "pole2": {{
                        "name": "domain1_opposite",
                        "evidence": [
                            "Direct quote supporting this harmful aspect"
                        ]
                    }},
                    "semantic_contrast_explanation": "Explanation of why these poles create perfect semantic contrast for MDS"
                }},
                ...
            ]
        }}
        ```
        


        IMPORTANT: For each domain, include a "semantic_contrast_explanation" that explains why the two poles will create MINIMUM semantic distance in embedding space.

        Follow the Core Moral Foundations:
        - Care/Harm - Value: Compassion, kindness, protecting others from suffering.
        - Fairness/Cheating - Value: Justice, equality, proportionality, reciprocity.
        - Loyalty/Betrayal - Value: Group loyalty, patriotism, self-sacrifice for the group.
        - Authority/Subversion - Value: Respect for tradition, social hierarchy, legitimate authority.
        - Sanctity/Degradation - Value: Purity, sacredness, avoiding disgust or contamination.
        - Liberty/Oppression - Value: Freedom from domination and tyranny.
        """
        

  
        

        response = self.call_model(prompt, temperature=0)
        print("Raw Polarization Agent response:")
        print(response)
        
        try:
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_match = re.search(r'({.*})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    return {"error": "Could not extract JSON from response", "raw_response": response}
            
            result = json.loads(json_str)
            
            # Zachowaj wyjaśnienia kontrastu semantycznego
            semantic_contrasts = {}
            if "polarized_domains" in result:
                for domain in result["polarized_domains"]:
                    if "semantic_contrast_explanation" in domain:
                        domain_name = domain.get("domain", "")
                        semantic_contrasts[domain_name] = domain["semantic_contrast_explanation"]
                        domain.pop("semantic_contrast_explanation", None)
            
            if semantic_contrasts:
                result["semantic_contrasts"] = semantic_contrasts
            
            # Połącz z poprzednimi wynikami
            result.update({
                "key_passages": previous_results.get("key_passages", []),
                "main_themes": previous_results.get("main_themes", []),
                "domains": previous_results.get("domains", [])
            })
            
            return result
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw response: {response}")
            return {"error": "JSON decode error", "raw_response": response}


class DefinitionAgent(SemanticAgent):
    """Agent odpowiedzialny za tworzenie skrajnie spolaryzowanych definicji dla MDS"""
    
    def __init__(self, config: ModelConfig, provider_instance=None):
        super().__init__(config, role="definer", provider_instance=provider_instance)
    
    def process(self, text: str, previous_results: Dict, text_sample: str) -> Dict:
        polarized_domains = previous_results.get("polarized_domains", [])
        
        # Utwórz listę wszystkich par nazw aspektów
        aspect_names = []
        for domain in polarized_domains:
            pole1 = domain.get("pole1", {})
            pole2 = domain.get("pole2", {})
            pole1_name = pole1.get("name", "")
            pole2_name = pole2.get("name", "")
            if pole1_name and pole2_name:
                aspect_names.append((pole1_name, pole2_name))
        
        print(f"Aspect name pairs to be defined: {aspect_names}")
        
        polarized_info = ""
        for domain in polarized_domains:
            domain_name = domain.get("domain", "")
            pole1 = domain.get("pole1", {})
            pole2 = domain.get("pole2", {})
            
            polarized_info += f"\n## Domain: {domain_name}\n"
            polarized_info += f"Pole 1: {pole1.get('name', '')}\n"
            polarized_info += "Evidence:\n"
            for evidence in pole1.get("evidence", []):
                polarized_info += f"- {evidence}\n"
            
            polarized_info += f"\nPole 2: {pole2.get('name', '')}\n"
            polarized_info += "Evidence:\n"
            for evidence in pole2.get("evidence", []):
                polarized_info += f"- {evidence}\n"
        
        prompt = f"""
        # Detailed Semantic Definitions for MDS Analysis
        
        Create precise, detailed definitions for each pole of the identified semantic dimensions.
        These definitions will serve as the foundation for positioning texts in MDS analysis.
        
        ## Polarized Dimensions:
        {polarized_info}
        
        ## Original Text Sample (for reference):
        ```
        {text_sample}
        ```
        
        ## Guidelines for Creating Definitions:
        
        1. Create a detailed definition (70-100 words) for each pole
        2. Definitions must be RICH IN DOMAIN-SPECIFIC VOCABULARY from the original text
        3. Include KEY INDICATOR TERMS that would signal this pole in a text
        4. Make definitions SPECIFIC and CONCRETE, not abstract or generic
        5. Ensure definitions clearly distinguish between poles
        6. BALANCE the length and detail between opposing poles
        7. Use EVALUATIVE LANGUAGE that makes positioning straightforward
        8. Create definitions that would work for raters who haven't read the original text
        
        ## Required Definition Structure (for each pole):
        
        1. Opening sentence: Clear statement of what this pole represents
        2. Detail sentence: Specific characteristics, approaches, or qualities
        3. Indicator sentence: Observable signs or markers in a text
        
        ## Output Format:
        
        Return your definitions as a Python dictionary with this EXACT format:
        ```python
        semantic_features = {{
            'pole1_name': ["Detailed definition that is 70-100 words long and rich in domain-specific vocabulary..."],
            'pole2_name': ["Detailed definition that is 70-100 words long and rich in domain-specific vocabulary..."],
            # Additional poles...
        }}
        ```
        
        REMEMBER:
        - Each definition MUST be at least 70 words long!
        - Definitions MUST use vocabulary from the original text
        - Each definition should be a single one-sentence string in a list
        - IMPORTANT: KEY INDICATOR TERMS must be written in UPPER CASE and embedded naturally within the sentence of each definition
        - IMPORTANT: The dimension would capture MINIMUM variance in MDS analysis 

        Follow the Core Moral Foundations:
        - Care/Harm - Value: Compassion, kindness, protecting others from suffering.
        - Fairness/Cheating - Value: Justice, equality, proportionality, reciprocity.
        - Loyalty/Betrayal - Value: Group loyalty, patriotism, self-sacrifice for the group.
        - Authority/Subversion - Value: Respect for tradition, social hierarchy, legitimate authority.
        - Sanctity/Degradation - Value: Purity, sacredness, avoiding disgust or contamination.
        - Liberty/Oppression - Value: Freedom from domination and tyranny.


        """
        
        response = self.call_model(prompt, temperature=0)
        print("Raw Definition Agent response:")
        print(response)
        
        # Ekstrahuj Python dict z odpowiedzi
        semantic_features_raw = self.extract_python_dict(response)
        
        if not semantic_features_raw:
            print("Failed to extract semantic_features dictionary, trying alternative approach")
            semantic_features = {}
            for domain in polarized_domains:
                pole1 = domain.get("pole1", {})
                pole2 = domain.get("pole2", {})
                pole1_name = pole1.get("name", "")
                pole2_name = pole2.get("name", "")
                
                if pole1_name and pole1.get("evidence"):
                    evidence = pole1.get("evidence", [""])[0]
                    semantic_features[pole1_name] = [f"EXTREMELY {pole1_name.replace('_', ' ')}: {evidence}"]
                
                if pole2_name and pole2.get("evidence"):
                    evidence = pole2.get("evidence", [""])[0]
                    semantic_features[pole2_name] = [f"EXTREMELY {pole2_name.replace('_', ' ')}: {evidence}"]
        else:
            semantic_features = {}
            for key, value in semantic_features_raw.items():
                if isinstance(value, list):
                    if not value:
                        semantic_features[key] = ["Definition not provided"]
                    else:
                        first_item = value[0] if isinstance(value[0], str) else str(value[0])
                        semantic_features[key] = [first_item]
                elif isinstance(value, str):
                    semantic_features[key] = [value]
                else:
                    semantic_features[key] = [str(value)]
        
        # Upewnij się, że każdy aspekt ma definicję
        for domain in polarized_domains:
            pole1 = domain.get("pole1", {})
            pole2 = domain.get("pole2", {})
            pole1_name = pole1.get("name", "")
            pole2_name = pole2.get("name", "")
            
            if pole1_name and pole1_name not in semantic_features:
                evidence = pole1.get("evidence", ["No evidence provided"])[0]
                semantic_features[pole1_name] = [f"ABSOLUTELY {pole1_name.replace('_', ' ')} represents the extreme where {evidence}"]
            
            if pole2_name and pole2_name not in semantic_features:
                evidence = pole2.get("evidence", ["No evidence provided"])[0]
                semantic_features[pole2_name] = [f"ABSOLUTELY {pole2_name.replace('_', ' ')} represents the extreme where {evidence}"]
        
        # Sprawdź jakość definicji
        quality_issues = self.verify_definitions_quality(semantic_features, polarized_domains)
        if quality_issues:
            print("Definition quality issues found:")
            for issue in quality_issues:
                print(f"- {issue}")
        
        # Połącz z poprzednimi wynikami
        full_result = {
            "key_passages": previous_results.get("key_passages", []),
            "main_themes": previous_results.get("main_themes", []),
            "domains": previous_results.get("domains", []),
            "polarized_domains": previous_results.get("polarized_domains", []),
            "semantic_features": semantic_features,
            "quality_issues": quality_issues
        }
        
        return full_result


class VerificationAgent(SemanticAgent):
    """Agent odpowiedzialny za weryfikację całej analizy pod kątem MDS"""
    
    def __init__(self, config: ModelConfig, provider_instance=None):
        super().__init__(config, role="verifier", provider_instance=provider_instance)
    
    def process(self, text: str, previous_results: Dict) -> Dict:
        semantic_features = previous_results.get("semantic_features", {})
        polarized_domains = previous_results.get("polarized_domains", [])
        quality_issues = previous_results.get("quality_issues", [])
        
        # Sprawdź czy semantic_features zawiera wszystkie spolaryzowane aspekty
        expected_aspects = []
        for domain in polarized_domains:
            pole1 = domain.get("pole1", {})
            pole2 = domain.get("pole2", {})
            pole1_name = pole1.get("name", "")
            pole2_name = pole2.get("name", "")
            
            if pole1_name:
                expected_aspects.append(pole1_name)
            if pole2_name:
                expected_aspects.append(pole2_name)
        
        missing_aspects = [aspect for aspect in expected_aspects if aspect not in semantic_features]
        if missing_aspects:
            print(f"Missing aspects in semantic_features: {missing_aspects}")
            
            # Uzupełnij brakujące aspekty
            for domain in polarized_domains:
                pole1 = domain.get("pole1", {})
                pole2 = domain.get("pole2", {})
                pole1_name = pole1.get("name", "")
                pole2_name = pole2.get("name", "")
                
                if pole1_name and pole1_name not in semantic_features and pole1.get("evidence"):
                    semantic_features[pole1_name] = [
                        f"Based on the text, {pole1_name} refers to aspects related to {domain.get('domain', '')}."
                    ]
                
                if pole2_name and pole2_name not in semantic_features and pole2.get("evidence"):
                    semantic_features[pole2_name] = [
                        f"Based on the text, {pole2_name} refers to aspects related to {domain.get('domain', '')}."
                    ]
        
        # Sprawdź format
        for key, value in semantic_features.items():
            if not isinstance(value, list):
                print(f"Converting {key} value to proper list format")
                semantic_features[key] = [str(value)]
            elif len(value) != 1:
                semantic_features[key] = [value[0] if value else "Definition not provided"]
        
        print(f"Aspects being verified: {list(semantic_features.keys())}")
        
        semantic_info = ""
        for key, value in semantic_features.items():
            definition = value[0] if isinstance(value, list) and len(value) > 0 else str(value)
            semantic_info += f"- {key}:\n  Definition: {definition}\n\n"
        
        polarization_info = ""
        for domain in polarized_domains:
            domain_name = domain.get("domain", "")
            pole1 = domain.get("pole1", {})
            pole2 = domain.get("pole2", {})
            
            polarization_info += f"\n## Domain: {domain_name}\n"
            polarization_info += f"Pole 1: {pole1.get('name', '')}\n"
            polarization_info += "Evidence:\n"
            for evidence in pole1.get("evidence", []):
                polarization_info += f"- {evidence}\n"
            
            polarization_info += f"\nPole 2: {pole2.get('name', '')}\n"
            polarization_info += "Evidence:\n"
            for evidence in pole2.get("evidence", []):
                polarization_info += f"- {evidence}\n"
                
        quality_info = ""
        if quality_issues:
            quality_info = "\n## Quality Issues Identified:\n"
            for issue in quality_issues:
                quality_info += f"- {issue}\n"
        
        prompt = f"""
        # MDS-Optimized Verification Task
        
        Critically evaluate and improve the semantic definitions for their suitability in Multidimensional Scaling (MDS) analysis.
        
        Here is the original text (excerpt):
        ```
        {text}...
        ```
        
        Here are the final semantic features extracted:
        
        {semantic_info}
        
        Here is the polarization analysis that led to these features:
        
        {polarization_info}
        
        {quality_info}
        
        For each semantic feature:
        1. Verify that it exists in the original text
        2. Check that the definition accurately reflects the text and is well-suited for MDS analysis
        3. Ensure the definition creates a clear dimension endpoint that texts can be positioned against
        4. Improve definitions to make them more precise, concise, and useful for MDS
        5. Ensure each pair of opposing definitions truly represents opposite ends of a meaningful continuum
        
        VERY IMPORTANT:
        1. The output format is critical - you must return a Python dictionary following EXACTLY this pattern:
        
        ```python
        semantic_features = {{
            'aspect1_name': ["Improved definition sentences for MDS analysis."],
            'aspect2_name': ["Improved definition sentences for MDS analysis."],
            ... (for all original semantic features with UNCHANGED names)
        }}
        ```
        
        1. Do NOT change the names of any semantic features
        2. Retain ALL the original semantic feature names
        3. Each value MUST be a list with EXACTLY one element - a string containing the improved definitions
        4. Each definition MUST be at least 70 words long!
        5. Focus on making the definitions optimally suited for positioning texts in a multidimensional semantic space
        6. Ensure opposing definitions within the same domain are truly polar opposites
        7. Do NOT add any explanation text outside the Python dictionary
        8. IMPORTANT: The dimension would capture MINIMUM variance in MDS analysis
        9. IMPORTANT: KEY INDICATOR TERMS must be written in UPPER CASE and embedded naturally within the sentence of each definition

        """
        
        response = self.call_model(prompt, temperature=0.0)
        print("Raw Verification Agent response:")
        print(response)
        
        verified_features = self.extract_python_dict(response)
        
        if not verified_features:
            print("Failed to extract verified features, using original semantic features")
            verified_features = semantic_features
        
        # Upewnij się, że verified_features zawiera wszystkie oczekiwane aspekty
        for aspect in expected_aspects:
            if aspect not in verified_features:
                print(f"Adding missing aspect after verification: {aspect}")
                verified_features[aspect] = semantic_features.get(aspect, ["Definition not provided."])
            
            if not isinstance(verified_features[aspect], list):
                verified_features[aspect] = [str(verified_features[aspect])]
            elif len(verified_features[aspect]) != 1:
                if len(verified_features[aspect]) < 1:
                    verified_features[aspect] = ["Definition not provided."]
                else:
                    verified_features[aspect] = [verified_features[aspect][0]]
        
        # Przeprowadź ocenę jakości definicji
        quality_assessment = self.assess_definitions_with_model(verified_features)
        
        # Utwórz końcowy wynik
        final_result = {
            "semantic_features": verified_features,
            "quality_assessment": quality_assessment,
            "agent_analysis": {
                "key_passages": previous_results.get("key_passages", []),
                "main_themes": previous_results.get("main_themes", []),
                "domains": [d.get("name") for d in previous_results.get("domains", [])],
                "polarized_domains": previous_results.get("polarized_domains", [])
            }
        }
        
        return final_result


class MDSOptimizationAgent(SemanticAgent):
    """Agent specjalizujący się w optymalizacji skrajnie spolaryzowanych definicji dla MDS"""
    
    def __init__(self, config: ModelConfig, provider_instance=None):
        super().__init__(config, role="mds_optimizer", provider_instance=provider_instance)
    
    def process(self, text: str, previous_results: Dict) -> Dict:
        semantic_features = previous_results.get("semantic_features", {})
        polarized_domains = previous_results.get("polarized_domains", [])
        
        # Przygotuj pary definicji do optymalizacji
        definition_pairs = {}
        for domain in polarized_domains:
            pole1 = domain.get("pole1", {})
            pole2 = domain.get("pole2", {})
            pole1_name = pole1.get("name", "")
            pole2_name = pole2.get("name", "")
            
            if pole1_name in semantic_features and pole2_name in semantic_features:
                definition_pairs[domain.get("domain", "")] = {
                    "pole1": {"name": pole1_name, "def": semantic_features[pole1_name][0]},
                    "pole2": {"name": pole2_name, "def": semantic_features[pole2_name][0]}
                }
        
        pairs_info = ""
        for domain, pair in definition_pairs.items():
            pairs_info += f"\n## Dimension: {domain}\n"
            pairs_info += f"Pole 1: {pair['pole1']['name']}\n"
            pairs_info += f"Definition: {pair['pole1']['def']}\n\n"
            pairs_info += f"Pole 2: {pair['pole2']['name']}\n"
            pairs_info += f"Definition: {pair['pole2']['def']}\n"
        
        prompt = f"""
        # Extreme MDS Optimization Task
        
        Review and MINIMIZE the polarization of the following pairs of semantic definitions for use in Multidimensional Scaling (MDS) analysis. Your goal is to create MINIMIZE DIFFERENTIATION between opposing poles.
        
        {pairs_info}
        
        For each pair of definitions:
        
        1. MINIMUM VARIANCE: Definitions clearly represent extreme endpoints that would MINIMIZE variance in positioning
        2. CONSISTENT MEASURABILITY: Each dimension has clear, observable criteria for consistently positioning texts
        3. DISTINCTIVE VOCABULARY: Each pole has distinctive terminology with minimal overlap with opposing pole
        4. BALANCED DETAIL: Both poles have similar levels of specificity and detail (70-100 words each)
        5. CLEAR OPPOSITION: Definitions should use direct opposing terms and concepts
        6. TEXT-BASED TERMINOLOGY: All vocabulary drawn directly from the original text, not generic terms
        7. INDEPENDENCE: Minimize conceptual overlap between different dimensions
        
        ## Expert MDS Considerations:

        1. Each definition MUST be at least 70 words long!
        2. IMPORTANT:Include 5-7 INDICATOR TERMS for each pole (specific words/phrases that signal alignment)
        3. Create truly ORTHOGONAL dimensions that measure distinct semantic aspects
        4. Ensure definitions ENABLE DISCRIMINATION between texts along each dimension
        5. Make definitions RESILIENT to different raters/interpreters by using clear, objective language
        
        Return your extremely polarized definitions in this exact format:
        ```python
        semantic_features = {{
            'pole1_name': ["Polarized definition"],
            'pole2_name': ["OPPOSITE polarized definition "],
            ...
        }}
        ```
        
        Your goal is to make these definitions so extremely polarized that:
        - Raters would have no ambiguity when scoring texts
        - Texts will distribute clearly across the dimension
        - IMPORTANT: The dimension would capture MINIMUM variance in MDS analysis
        - IMPORTANT: KEY INDICATOR TERMS must be written in UPPER CASE and embedded naturally within the sentence of each definition
        - The definitions create clear, unambiguous anchors for the extremes of each dimension
        """
        
        response = self.call_model(prompt, temperature=0.0)
        print("Raw MDS Optimization Agent response:")
        print(response)
        
        optimized_features = self.extract_python_dict(response)
        
        if optimized_features:
            # Aktualizuj tylko istniejące klucze
            for key in semantic_features:
                if key in optimized_features:
                    semantic_features[key] = optimized_features[key]
        
        # Oblicz metryki jakości dla zoptymalizowanych definicji
        # quality_metrics = self.evaluate_extreme_optimization(semantic_features, definition_pairs)
        quality_metrics = None

        # Zwróć zaktualizowane wyniki
        updated_results = previous_results.copy()
        updated_results["semantic_features"] = semantic_features
        updated_results["optimization_metrics"] = quality_metrics
        
        return updated_results
    
    def evaluate_extreme_optimization(self, optimized_features, original_pairs):
        """Ocenia jakość ekstremalnej polaryzacji definicji po optymalizacji"""
        metrics = {
            "polarization_score": 0,
            "evaluation_language_score": 0,
            "mutual_exclusivity_score": 0,
            "dimensions": {}
        }
        
        total_pairs = len(original_pairs)
        if total_pairs == 0:
            return metrics
            
        for domain, pair in original_pairs.items():
            pole1_name = pair["pole1"]["name"]
            pole2_name = pair["pole2"]["name"]
            
            if pole1_name in optimized_features and pole2_name in optimized_features:
                def1 = optimized_features[pole1_name][0].lower()
                def2 = optimized_features[pole2_name][0].lower()
                
                # Sprawdź ekstremalne słowa kluczowe
                extreme_words = ["absolutely", "completely", "extremely", "entirely", "utterly", 
                                "wholly", "fully", "totally", "fundamentally", "radically"]
                
                extreme_count1 = sum(1 for word in extreme_words if word in def1)
                extreme_count2 = sum(1 for word in extreme_words if word in def2)
                
                # Sprawdź wartościujące słownictwo
                evaluative_words = ["beneficial", "harmful", "effective", "ineffective", "positive", 
                                  "negative", "strong", "weak", "good", "bad", "superior", "inferior"]
                
                eval_count1 = sum(1 for word in evaluative_words if word in def1)
                eval_count2 = sum(1 for word in evaluative_words if word in def2)
                
                # Oblicz przeciwstawność (niski overlap słów)
                words1 = set(def1.split())
                words2 = set(def2.split())
                common_words = words1 & words2
                
                # Słowa, które powinny być wspólne (połączenie domeny)
                domain_words = set(domain.lower().split())
                
                # Wykluczenie wzajemne (poza słowami domenowymi)
                mutual_exclusivity = 1 - (len(common_words - domain_words) / 
                                         (len(words1) + len(words2) - len(domain_words)) * 2)
                extreme_count1 = 1
                extreme_count2 = 1
                # Metryki dla tej pary
                metrics["dimensions"][domain] = {
                    "extreme_language_score": (extreme_count1 + extreme_count2) / 2,
                    "evaluative_language_score": (eval_count1 + eval_count2) / 2,
                    "mutual_exclusivity": mutual_exclusivity,
                 
                }
                
                # Dodaj do ogólnych wyników
                metrics["polarization_score"] += (extreme_count1 + extreme_count2) / 2
                metrics["evaluation_language_score"] += (eval_count1 + eval_count2) / 2
                metrics["mutual_exclusivity_score"] += mutual_exclusivity
        
        # Normalizuj wyniki
        if total_pairs > 0:
            metrics["polarization_score"] /= total_pairs
            metrics["evaluation_language_score"] /= total_pairs
        
        return metrics
    

"""
Klasa koordynująca pracę wszystkich agentów semantycznych
"""

import json
from typing import Dict



class AgentOrchestrator:
    """Klasa koordynująca pracę wszystkich agentów z obsługą różnych providerów"""
    
    def __init__(self, api_key: str, provider: str = "claude", 
                 model: str = None, max_tokens: int = 10000, temperature: float = 0.0):
        """
        Inicjalizacja orchestratora
        
        Args:
            api_key: Klucz API dla wybranego providera
            provider: Nazwa providera ("claude", "gemini", "openai", "chatgpt", "gpt")
            model: Nazwa modelu (opcjonalne, będzie wybrane automatycznie)
            max_tokens: Maksymalna liczba tokenów w odpowiedzi
            temperature: Temperatura dla generacji tekstu
        """
        self.config = ProviderFactory.create_model_config(
            provider=provider,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Sprawdź dostępność providera
        status = ProviderFactory.get_available_providers()
        if not status.get(provider.lower(), False):
            available = [p for p, available in status.items() if available]
            raise RuntimeError(f"Provider {provider} nie jest dostępny. "
                             f"Dostępne providery: {available}")
        
        print(f"Inicjalizacja orchestratora z {provider} ({self.config.model_name})")
        
    def process_text(self, text: str, context_len: int = 100, defined_semantic_quialities = None) -> Dict:
        """Sekwencyjnie przetwarza tekst przez wszystkich agentów z dodatkową optymalizacją MDS"""
        
        # Inicjalizacja agentów
        provider_instance = ProviderFactory.create_provider(self.config.provider)
        
        reading_agent = ReadingAgent(self.config, provider_instance)
        domain_agent = DomainExtractionAgent(self.config, provider_instance)
        polarization_agent = PolarizationAgent(self.config, provider_instance)
        definition_agent = DefinitionAgent(self.config, provider_instance)
        mds_optimization_agent = MDSOptimizationAgent(self.config, provider_instance)
        verification_agent = VerificationAgent(self.config, provider_instance)

        # Sekwencyjne przetwarzanie
        print("Step 1: Reading and extracting key passages...")
        reading_results = reading_agent.process(text, None, context_len)
        self._print_results("Reading", reading_results)
        
        print("Step 2: Extracting domains...")
        domain_results = domain_agent.process(text, reading_results, defined_semantic_quialities)
        self._print_results("Domain", domain_results)
        
        print("Step 3: Polarizing domains...")
        polarization_results = polarization_agent.process(text, domain_results)
        self._print_results("Polarization", polarization_results)
        
        # Przygotuj próbkę tekstu dla definition_agent
        text_sample = text[:2000] if len(text) > 2000 else text
        
        print("Step 4: Creating definitions...")
        definition_results = definition_agent.process(text, polarization_results, text_sample)
        self._print_results("Definition", definition_results)
        
        # Sprawdź, czy semantic_features istnieje
        if "semantic_features" not in definition_results:
            print("WARNING: semantic_features missing from definition results!")
            definition_results["semantic_features"] = {}
            
            # Spróbuj utworzyć semantic_features z polarized_domains
            for domain in polarization_results.get("polarized_domains", []):
                pole1 = domain.get("pole1", {})
                pole2 = domain.get("pole2", {})
                pole1_name = pole1.get("name", "")
                pole2_name = pole2.get("name", "")
                
                if pole1_name and pole1.get("evidence"):
                    definition_results["semantic_features"][pole1_name] = [
                        f"Based on evidence: {pole1.get('evidence', [''])[0]}"
                    ]
                
                if pole2_name and pole2.get("evidence"):
                    definition_results["semantic_features"][pole2_name] = [
                        f"Based on evidence: {pole2.get('evidence', [''])[0]}"
                    ]
        
        print("Step 5: Optimizing definitions for MDS...")
        optimized_results = mds_optimization_agent.process(text, definition_results)
        print("Optimization metrics:", json.dumps(optimized_results.get("optimization_metrics", {}), indent=2))
        
        print("Step 6: Verifying results...")
        final_results = verification_agent.process(text, optimized_results)
        
        # Upewnij się, że semantic_features istnieje w końcowym wyniku
        if "semantic_features" not in final_results or not final_results["semantic_features"]:
            print("WARNING: No semantic_features in final results! Using optimized results instead.")
            if optimized_results.get("semantic_features"):
                final_results["semantic_features"] = optimized_results["semantic_features"]
            else:
                final_results["semantic_features"] = definition_results.get("semantic_features", {})
        
        # Przygotuj końcowy wynik w formacie odpowiednim dla MDS
        final_semantic_features = {}
        for key, value in final_results["semantic_features"].items():
            if isinstance(value, list) and value:
                final_semantic_features[key] = value
            else:
                final_semantic_features[key] = ["Definition not available."]
        
        # Dodaj metryki jakości
        quality_metrics = self._evaluate_definition_quality(final_semantic_features)
        
        # Utwórz końcowy wynik
        mds_ready_results = {
            "semantic_features": final_semantic_features,
            "quality_metrics": quality_metrics,
            "domains": [d.get("name") for d in domain_results.get("domains", [])],
            "polarized_domains": final_results.get("agent_analysis", {}).get("polarized_domains", []),
            "provider_info": {
                "provider": self.config.provider,
                "model": self.config.model_name
            }
        }
        
        return mds_ready_results
    
    def _print_results(self, step_name: str, results: Dict):
        """Pomocnicza metoda do wydruku wyników z ograniczeniem długości"""
        results_str = json.dumps(results, indent=2, ensure_ascii=False)
        if len(results_str) > 10000:
            print(f"{step_name} results (truncated):", results_str[:10000] + "...")
        else:
            print(f"{step_name} results:", results_str)
    
    def _evaluate_definition_quality(self, semantic_features: Dict) -> Dict:
        """Ocenia jakość definicji na podstawie cech statystycznych"""
        if not semantic_features:
            return {
                "dimension_count": 0,
                "avg_definition_length": 0,
                "completeness": 0,
                "dimensionality": {}
            }
            
        evaluation = {
            "dimension_count": len(semantic_features) // 2,
            "avg_definition_length": sum(len(def_list[0]) for def_list in semantic_features.values()) / len(semantic_features) if semantic_features else 0,
            "completeness": sum(1 for def_list in semantic_features.values() if len(def_list[0]) > 40) / len(semantic_features) if semantic_features else 0,
            "dimensionality": {}
        }
        
        # Grupuj aspekty według domen
        domains = {}
        for key in semantic_features:
            domain = key.split('_')[0] if '_' in key else key
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(key)
        
        # Oblicz jakość każdego wymiaru
        for domain, aspects in domains.items():
            if len(aspects) == 2:
                def1 = semantic_features[aspects[0]][0].lower()
                def2 = semantic_features[aspects[1]][0].lower()
                
                # Prostsze miary przeciwstawności
                common_words = len(set(def1.split()) & set(def2.split()))
                total_words = (len(def1.split()) + len(def2.split())) / 2
                
                # Miara przeciwstawności jako proporcja wspólnych słów do całości
                oppositeness = 1 - (common_words / total_words) if total_words > 0 else 0
                
                evaluation["dimensionality"][domain] = {
                    "oppositeness": oppositeness,
                    "definition_balance": min(len(def1), len(def2)) / max(len(def1), len(def2)) if max(len(def1), len(def2)) > 0 else 0
                }
        
        return evaluation
    
    def get_provider_info(self) -> Dict:
        """Zwraca informacje o aktualnie używanym providerze"""
        return {
            "provider": self.config.provider,
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
    
    def switch_provider(self, new_provider: str, api_key: str, model: str = None):
        """Przełącza na innego providera"""
        self.config = ProviderFactory.create_model_config(
            provider=new_provider,
            api_key=api_key,
            model=model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        print(f"Przełączono na {new_provider} ({self.config.model_name})")


def compare_providers(text: str, api_keys: Dict[str, str], context_len: int = 10) -> Dict:
    """
    Porównuje wyniki różnych providerów dla tego samego tekstu
    
    Args:
        text: Tekst do analizy
        api_keys: Słownik {provider: api_key}
        context_len: Liczba kluczowych fragmentów do ekstraktu
        
    Returns:
        Słownik z wynikami dla każdego providera
    """
    results = {}
    
    for provider, api_key in api_keys.items():
        print(f"\n{'='*50}")
        print(f"Analiza z providerem: {provider.upper()}")
        print(f"{'='*50}")
        
        try:
            orchestrator = AgentOrchestrator(api_key=api_key, provider=provider)
            provider_results = orchestrator.process_text(text, context_len)
            results[provider] = provider_results
            
            # Krótkie podsumowanie
            semantic_count = len(provider_results.get("semantic_features", {}))
            domains_count = len(provider_results.get("domains", []))
            print(f"Znaleziono {domains_count} domen, {semantic_count} cech semantycznych")
            
        except Exception as e:
            print(f"Błąd przy analizie z {provider}: {e}")
            results[provider] = {"error": str(e)}
    
    return results



import json
import os
import argparse
from typing import Dict, Optional



def extract_semantic_concepts_with_agents(
    text: str, 
    api_key: str, 
    provider: str = "claude",
    model: str = None,
    context_len: int = 10,
    max_tokens: int = 10000,
    temperature: float = 0.0,
    defined_semantic_domains: str = None,
) -> Dict:
    """
    Główna funkcja do ekstrakcji konceptów semantycznych z wykorzystaniem agentów
    
    Args:
        text: Tekst do analizy
        api_key: Klucz API dla wybranego providera
        provider: Nazwa providera ("claude", "gemini", "openai", "chatgpt", "gpt")
        model: Nazwa modelu (opcjonalne)
        context_len: Liczba kluczowych fragmentów do ekstraktu
        max_tokens: Maksymalna liczba tokenów w odpowiedzi
        temperature: Temperatura dla generacji tekstu
        
    Returns:
        Słownik z wynikami analizy semantycznej
    """
    
    orchestrator = AgentOrchestrator(
        api_key=api_key,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    results = orchestrator.process_text(text, context_len, defined_semantic_domains)
    
    # Sprawdź czy otrzymaliśmy oczekiwany format semantic_features
    if "semantic_features" in results:
        final_semantic_features = {}
        
        # Przeiteruj przez wszystkie cechy semantyczne
        for key, value in results["semantic_features"].items():
            if isinstance(value, list) and len(value) >= 1:
                final_semantic_features[key] = [value[0]]
            elif isinstance(value, str):
                final_semantic_features[key] = [value]
            else:
                final_semantic_features[key] = ["No definition provided"]
        
        # Sprawdź format nazw aspektów
        proper_format_features = {}
        for key, value in final_semantic_features.items():
            if '_' in key:
                parts = key.split('_')
                if len(parts) >= 2:
                    proper_format_features[key] = value
                else:
                    proper_format_features[f"{key}_feature"] = value
            else:
                proper_format_features[f"{key}_feature"] = value
        
        results["semantic_features"] = proper_format_features
        
    # Przygotuj końcowy wynik
    final_output = {
        "semantic_features": results.get("semantic_features", {}),
        "quality_metrics": results.get("quality_metrics", {}),
        "provider_info": results.get("provider_info", {}),
        "domains": results.get("domains", []),
        "polarized_domains": results.get("polarized_domains", [])
    }
    
    # Wydrukuj nazwy kluczy w semantic_features
    if "semantic_features" in final_output:
        print("\nFinal semantic features keys:")
        for key in final_output["semantic_features"].keys():
            print(f"  - {key}")
    
    return final_output


import json
from datetime import datetime
from typing import Dict

def save_semantic_features(semantic_features: Dict, base_filename: str = "semantic_features_MAXIMIZE"):
    """
    Zapisuje semantic_features do pliku JSON z datą w nazwie
    
    Args:
        semantic_features: Słownik z cechami semantycznymi do zapisania
        base_filename: Podstawowa nazwa pliku (bez rozszerzenia)
    """
    # Generowanie znacznika czasowego w formacie YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Tworzenie pełnej nazwy pliku z datą
    filename = f"{base_filename}_{timestamp}.json"
    
    # Zapisywanie do pliku
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(semantic_features, f, indent=2, ensure_ascii=False)
    
    print(f"Semantic features saved to {filename}")
    return filename  # Zwracamy nazwę pliku dla dalszego użytku


def load_semantic_features(filename: str = "semantic_features.json") -> Optional[Dict]:
    """Wczytuje semantic_features z pliku JSON"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"File {filename} not found")
        return None


def interactive_analysis():
    """Interaktywna analiza z wyborem providera"""
    print("=== Analiza Semantyczna - Multiple API Support ===\n")
    
    # Wyświetl status providerów
    print(get_provider_status_report())
    print()
    
    # Wybór providera
    provider = input("Wybierz provider (claude/gemini/openai): ").strip().lower()
    if not provider:
        provider = "claude"
    
    # Klucz API
    api_key = input(f"Podaj klucz API dla {provider}: ").strip()
    if not api_key:
        print("Klucz API jest wymagany!")
        return
    
    # Model (opcjonalnie)
    model = input("Podaj nazwę modelu (enter = domyślny): ").strip()
    if not model:
        model = None
    
    # Tekst do analizy
    print("\nPodaj tekst do analizy (zakończ pustą linią):")
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    
    text = "\n".join(lines)
    if not text.strip():
        print("Tekst nie może być pusty!")
        return
    
    # Parametry analizy
    try:
        context_len = int(input("Liczba kluczowych fragmentów (10): ") or "10")
    except ValueError:
        context_len = 10
    
    print(f"\nRozpoczynamy analizę z providerem {provider}...")
    
    try:
        results = extract_semantic_concepts_with_agents(
            text=text,
            api_key=api_key,
            provider=provider,
            model=model,
            context_len=context_len
        )
        
        print("\n" + "="*60)
        print("WYNIKI ANALIZY")
        print("="*60)
        
        # Wyświetl podstawowe informacje
        provider_info = results.get("provider_info", {})
        print(f"Provider: {provider_info.get('provider', 'unknown')}")
        print(f"Model: {provider_info.get('model', 'unknown')}")
        
        # Wyświetl domeny
        domains = results.get("domains", [])
        print(f"\nZnalezione domeny ({len(domains)}):")
        for i, domain in enumerate(domains, 1):
            print(f"  {i}. {domain}")
        
        # Wyświetl cechy semantyczne
        semantic_features = results.get("semantic_features", {})
        print(f"\nCechy semantyczne ({len(semantic_features)}):")
        for key, value in semantic_features.items():
            definition = value[0] if isinstance(value, list) and value else str(value)
            print(f"\n{key}:")
            print(f"  {definition[:200]}{'...' if len(definition) > 200 else ''}")
        
        # Metryki jakości
        quality = results.get("quality_metrics", {})
        if quality:
            print(f"\nMetryki jakości:")
            print(f"  Liczba wymiarów: {quality.get('dimension_count', 0)}")
            print(f"  Średnia długość definicji: {quality.get('avg_definition_length', 0):.1f}")
            print(f"  Kompletność: {quality.get('completeness', 0):.2%}")
        
        # Zapisz wyniki
        save_choice = input("\nCzy zapisać wyniki do pliku? (y/n): ").strip().lower()
        if save_choice == 'y':
            filename = input("Nazwa pliku (semantic_features.json): ").strip()
            if not filename:
                filename = "semantic_features.json"
            save_semantic_features(results, filename)
        
    except Exception as e:
        print(f"Błąd podczas analizy: {e}")


