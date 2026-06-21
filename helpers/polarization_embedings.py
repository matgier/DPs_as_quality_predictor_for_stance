from pickle import TRUE
from time import sleep
from matplotlib.patches import Patch
import numpy as np
import json
import requests
from typing import List, Dict, Union, Optional, Tuple
from sklearn.preprocessing import StandardScaler


from helpers.all_keys import *


def normalization_and_centralization(embedding_value: List[float], centred: bool = True, normalized: bool = True, normalization_type: str = 'l2'):
    # Centrowanie (jeśli wymagane)
    embedding = embedding_value
    if not isinstance(embedding_value, np.ndarray):
        embedding_array = np.array(embedding_value)
    # Normalizacja (jeśli wymagane)
    # if centred:
    #     mean_value = np.mean(embedding_array)
    #     embedding_array = embedding_array - mean_value
    if normalized:
        if normalization_type == "l2":
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm

        norm_value = np.linalg.norm(embedding_array)
        mean_value = np.mean(embedding_array)
        # print(f"    - Wartość normy: {norm_value}")
        # print(f"    - Wartość średnia: {mean_value}")

        # Zapis jako lista
        embedding = embedding_array.tolist()
    return embedding


def generate_embedding_ollama(
    text: str, 
    model: str = "mistral:latest", 
    host: str = "localhost", 
    port: int = 11434,
    normalize: bool = True,
    normalization_type: str = 'l2',
    timeout: int = 30
) -> List[float]:
    """
    Generuje embedding dla podanego tekstu używając wybranego modelu w Ollamie.
    
    Parametry:
    ----------
    text : str
        Tekst, który ma zostać przekształcony w embedding.
    model : str, domyślnie "mistral:latest"
        Nazwa modelu w Ollamie do generowania embeddingu. 
        Sugerowane modele: "nomic-embed-text", "llama2", "llama3", "mistral", "gemma".
    host : str, domyślnie "localhost"
        Host, na którym działa Ollama.
    port : int, domyślnie 11434
        Port, na którym nasłuchuje API Ollamy.
    normalize : bool, domyślnie True
        Czy znormalizować wektor embeddingu do jednostkowej długości.
    timeout : int, domyślnie 30
        Limit czasu w sekundach na odpowiedź od API.
        
    Zwraca:
    -------
    List[float]
        Lista floatów reprezentująca embedding tekstu
    
    Zgłasza:
    --------
    ConnectionError: Gdy nie można połączyć się z serwerem Ollama.
    ValueError: Gdy zwrócony embedding jest pusty lub gdy model nie istnieje.
    """
    # Upewnij się, że tekst nie jest pusty
    if not text or not text.strip():
        raise ValueError("Tekst nie może być pusty")
    
    # Przygotuj URL do API Ollamy
    url = f"http://{host}:{port}/api/embeddings"
    
    # Przygotuj dane do wysłania
    data = {
        "model": model,
        "prompt": text
    }
    
    try:
        # Wywołaj API z timeout
        response = requests.post(url, json=data, timeout=timeout)
        
        # Sprawdź, czy zapytanie się powiodło
        if response.status_code != 200:
            error_msg = f"Błąd API: {response.status_code}, {response.text}"
            raise ValueError(error_msg)
        
        # Pobierz embedding z odpowiedzi
        result = response.json()
        embedding = result.get("embedding")
        
        # Sprawdź, czy embedding nie jest pusty
        if not embedding:
            raise ValueError(f"Model '{model}' nie zwrócił embeddingu. Odpowiedź: {result}")
        
        embedding = normalization_and_centralization(embedding_value=embedding, normalized=normalize)

        return embedding
        
        
        
    except requests.exceptions.Timeout:
        raise ConnectionError(f"Timeout podczas łączenia z Ollamą na {host}:{port}. "
                             f"Serwer nie odpowiedział w ciągu {timeout} sekund.")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Nie można połączyć się z Ollamą na {host}:{port}. "
                             f"Upewnij się, że Ollama jest uruchomiona i dostępna.")
    except json.JSONDecodeError:
        raise ValueError(f"Nieprawidłowa odpowiedź z API Ollamy: {response.text}")



import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
import gc
import os

def generate_embedding(
    text: str, 
   # model_name: str = "mistralai/Mistral-7B-v0.1",
    # model_name: str = "meta-llama/Llama-3.1-8B",
    # model_name: str = "google/gemma-4-E4B",
    #model_name: str = "speakleash/Bielik-4.5B-v3",
    model_name: str = "Qwen/Qwen3-8B",
    layer_index: int = None,
    normalize: bool = True,
    normalization_type: str = 'l2',
    max_length: int = 10000
) -> List[float]:
    """
    Generuje najwyższej jakości embeddingi z Mistral-7B.
    
    PRIORYTET: MAKSYMALNA JAKOŚĆ I DOKŁADNOŚĆ
    ==========================================
    - Float32 dla pełnej precyzji numerycznej
    - Pełna długość kontekstu (4096 tokenów)
    - Prawidłowe mean pooling z attention mask
    - Brak chunking - pełny kontekst zachowany
    - Oryginalny model Mistral-7B bez kompresji
    
    Args:
        text: Tekst do przetworzenia
        model_name: Model Mistral-7B (nie zmieniać dla jakości)
        layer_index: Warstwa 16 (optymalna dla semantyki)
        normalize: Normalizacja L2 embeddingu
        normalization_type: 'l2' lub 'robust'
        max_length: Maksymalna długość (4096 dla Mistral)
        
    Returns:
        Lista floatów - embedding najwyższej jakości
    """
    
    # ========== WALIDACJA ==========
    if not text or not text.strip():
        raise ValueError("Tekst nie może być pusty")
    
    # ========== SETUP DLA MAKSYMALNEJ JAKOŚCI ==========
    # Zawsze float32 dla pełnej precyzji
    dtype = torch.float16
    
    # Wybór device - MPS dla M1 lub CPU
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     # Wyłączamy problematyczne optymalizacje
    #     # os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    #     # os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    #     print(f"🎯 Tryb MPS")
    # else:
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 8)
    #print(f"🎯 Tryb CPU")
    
    # ========== ŁADOWANIE MODELU ==========
   # print(f"📥 Ładowanie {model_name} w pełnej precyzji...")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        model_max_length=max_length
    )
    
    # Mistral może nie mieć pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model w pełnej precyzji
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True  # Sekwencyjne ładowanie
    ).to(device)
    
    model.eval()  # Tryb ewaluacji
    
    # ========== TOKENIZACJA ==========
    # Pełna tokenizacja bez truncation jeśli mieści się w max_length
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True  # Kluczowe dla jakości
    )
    
    # Przeniesienie na device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    #print(f"📏 Długość sekwencji: {inputs['input_ids'].shape[1]} tokenów")
    
    # ========== FORWARD PASS W PEŁNEJ PRECYZJI ==========
    with torch.inference_mode():
        # Bez autocast - chcemy pełną precyzję
        outputs = model(**inputs)
        print(f"Layer index: {layer_index}")
        # Ekstrakcja hidden states z wybranej warstwy
        hidden_states = outputs.hidden_states
        print(f"Model ma {len(hidden_states)} warstw, żądany indeks: {layer_index}")
        if layer_index >= len(hidden_states):
            raise ValueError(f"Model ma {len(hidden_states)} warstw, żądany indeks: {layer_index}")
        
        layer_output = hidden_states[layer_index + 1]  # [batch=1, seq_len, hidden_dim]
        
        # ========== MEAN POOLING Z ATTENTION MASK ==========
        # Kluczowe dla jakości - prawidłowe uśrednianie tylko rzeczywistych tokenów
        attention_mask = inputs['attention_mask'].unsqueeze(-1)  # [1, seq_len, 1]
        
        # Maskowanie padding tokens
        masked_output = layer_output * attention_mask
        
        # Suma tylko rzeczywistych tokenów
        sum_output = torch.sum(masked_output, dim=1)  # [1, hidden_dim]
        sum_mask = torch.sum(attention_mask, dim=1).clamp(min=1e-9)  # [1, 1]
        
        # Średnia ważona
        pooled_output = sum_output / sum_mask  # [1, hidden_dim]
        
        # Konwersja na numpy w pełnej precyzji
        embedding_tensor = pooled_output[0].detach()  # [hidden_dim]
        
        # Czyszczenie pamięci
        del outputs, hidden_states, layer_output, inputs
        if device.type == "mps":
            try:
                torch.mps.synchronize()
            except:
                pass
    
    # ========== KONWERSJA I NORMALIZACJA ==========
    # Konwersja na numpy zachowując float32
    if device.type == "mps":
        embedding_array = embedding_tensor.cpu().numpy().astype(np.float16)
    else:
        embedding_array = embedding_tensor.numpy()
    
    del embedding_tensor
    
    # Normalizacja dla lepszej porównywalności
    if normalize:
        if normalization_type == "l2":
            # Standardowa normalizacja L2
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm
                
        elif normalization_type == "robust":
            # Robust normalization - odporna na outliers
            median = np.median(embedding_array)
            mad = np.median(np.abs(embedding_array - median))
            if mad > 0:
                embedding_array = (embedding_array - median) / (1.4826 * mad)
    
    # ========== STATYSTYKI DLA WERYFIKACJI ==========
    # print(f"✅ Embedding najwyższej jakości:")
    # print(f"   - Wymiar: {len(embedding_array)}")
    # print(f"   - Norma: {np.linalg.norm(embedding_array):.4f}")
    # print(f"   - Średnia: {np.mean(embedding_array):.6f}")
    # print(f"   - Std: {np.std(embedding_array):.6f}")
    # print(f"   - Min/Max: [{np.min(embedding_array):.6f}, {np.max(embedding_array):.6f}]")
    
    # ========== CZYSZCZENIE ==========
    del model, tokenizer
    
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
            torch.mps.synchronize()
        except:
            pass
    
    gc.collect()
    
    # Zwracamy jako listę float32
    return embedding_array.tolist()






def generate_embedding_openai(
    text: str, 
    model: str = "text-embedding-3-large", 
    api_key: str = API_KEY_OPENAI,
    normalize: bool = True,
    normalization_type: str = 'l2',
    timeout: int = 30
) -> List[float]:
    """
    Generuje embedding dla podanego tekstu używając API OpenAI.
    """
    sleep(1)
    # print("----> generate_embedding_openai")
    # Upewnij się, że tekst nie jest pusty
    if not text or not text.strip():
        raise ValueError("Tekst nie może być pusty")
    
    try:
        #print("-> Using OPENAI to create embedding!")
        # Inicjalizacja klienta OpenAI
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Wywołanie API z timeout
        response = client.embeddings.create(
            input=text,
            model=model,
            timeout=timeout
        )
        
        # Pobierz embedding z odpowiedzi
        embedding = response.data[0].embedding
        
        # Sprawdź, czy embedding nie jest pusty
        if not embedding:
            raise ValueError(f"Model '{model}' nie zwrócił embeddingu.")

        embedding = normalization_and_centralization(embedding_value=embedding, normalized=normalize)
        return embedding
    except Exception as e:
        # Generyczna obsługa wyjątków dla uproszczenia
        raise ValueError(f"Błąd przy generowaniu embeddingu OpenAI: {str(e)}")


def generate_embedding_voyage(
    text: str, 
    model: str = "voyage-3-large", 
    api_key: str = None,
    input_type: str = None,
    normalize: bool = True,
    normalization_type: str = 'l2',

    timeout: int = 30
) -> List[float]:
    """
    Generuje embedding dla podanego tekstu używając API Voyage AI.
    """
    sleep(1)

    # Upewnij się, że tekst nie jest pusty
    if not text or not text.strip():
        raise ValueError("Tekst nie może być pusty")
    
    # Pobierz klucz API
    if api_key is None:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("Nie znaleziono klucza API. Podaj api_key jako parametr lub ustaw zmienną środowiskową VOYAGE_API_KEY.")
    
    try:
        print(f"-> Using VOYAGE AI to create embedding with model {model}!")
        
        # Endpoint API dla embeddingów
        url = "https://api.voyageai.com/v1/embeddings"
        
        # Przygotowanie nagłówków
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Przygotowanie danych
        data = {
            "model": model,
            "input": text
        }
        
        # Dodaj input_type jeśli został określony
        if input_type:
            if input_type not in ["query", "document"]:
                raise ValueError("input_type musi być 'query' lub 'document' lub None")
            data["input_type"] = input_type
        
        # Wykonanie żądania HTTP
        response = requests.post(
            url, 
            headers=headers, 
            json=data,
            timeout=timeout
        )
        
        # Sprawdzenie statusu odpowiedzi
        response.raise_for_status()
        
        # Parsowanie odpowiedzi
        result = response.json()
        
        # Sprawdzenie czy odpowiedź zawiera embedding
        if "data" not in result or len(result["data"]) == 0 or "embedding" not in result["data"][0]:
            raise ValueError(f"Model '{model}' nie zwrócił embeddingu. Odpowiedź API: {result}")
        
        # Pobierz embedding z odpowiedzi
        embedding = result["data"][0]["embedding"]
        
        # Sprawdź, czy embedding nie jest pusty
        if not embedding:
            raise ValueError(f"Model '{model}' zwrócił pusty embedding.")
        embedding = normalization_and_centralization(embedding_value=embedding, normalized=normalize)
        return embedding

        
    except Exception as e:
        # Generyczna obsługa wyjątków dla uproszczenia
        raise ValueError(f"Błąd przy generowaniu embeddingu Voyage: {str(e)}")

from google import genai
from google.genai.types import EmbedContentConfig
from time import sleep
from typing import List, Optional

def generate_embedding_gemini(
    text: str,
    model: str = "gemini-embedding-001",
    api_key: str = API_KEY_GEMINI,
    normalize: bool = True,
    timeout: int = 30,  # nadal “placeholder”, jeśli nie konfigurujesz transportu
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: Optional[int] = None,
    title: Optional[str] = None,
) -> List[float]:
    if not text or not text.strip():
        raise ValueError("Tekst nie może być pusty")
    """
    Generuje embedding dla podanego tekstu używając nowego Gemini API.
    
    Args:
        text: Tekst do przetworzenia
        model: Nazwa modelu (domyślnie gemini-embedding-001)
        api_key: Klucz API
        normalize: Czy normalizować embedding
        normalization_type: Typ normalizacji ('l2' lub 'robust')
        timeout: Timeout dla żądania
        task_type: Typ zadania dla embeddingu
        output_dimensionality: Opcjonalna redukcja wymiarowości
        title: Opcjonalny tytuł dla kontekstu
    """
    sleep(1)
    print("----> generate_embedding_gemini")

    # Walidacja wejścia
    if not text or not text.strip():
        raise ValueError("Tekst nie może być pusty")
    
    try:
        #print("-> Using GEMINI (new API) to create embedding!")
        
        # Inicjalizacja klienta z nowym API
        client = genai.Client(api_key=api_key)
        
        # Przygotowanie konfiguracji
        config_params = {
            "task_type": task_type
        }
        
        # Dodanie opcjonalnych parametrów jeśli są podane
        if output_dimensionality:
            config_params["output_dimensionality"] = output_dimensionality
        if title:
            config_params["title"] = title
            
        config = EmbedContentConfig(**config_params)
        
        # Wywołanie API z nową strukturą
        response = client.models.embed_content(
            model=model,
            contents=text,
            config=config
        )
        
        # Wyciągnięcie embeddingu z nowej struktury odpowiedzi
        if not response.embeddings or len(response.embeddings) == 0:
            raise ValueError(f"Model '{model}' nie zwrócił embeddingu.")
            
        embedding= response.embeddings[0].values
        if not embedding:
            raise ValueError(f"Embedding jest pusty dla modelu '{model}'.")

        embedding = normalization_and_centralization(embedding_value=embedding, normalized=normalize)
        return embedding

    except Exception as e:
        raise ValueError(f"Błąd przy generowaniu embeddingu Gemini: {str(e)}")



