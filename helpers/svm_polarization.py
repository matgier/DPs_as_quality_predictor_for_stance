"""
Moduł do generowania embeddingów z różnych API.
Obsługuje OpenAI, Voyage AI, Gemini oraz lokalną Ollamę.
"""

import os
import json
import requests
import numpy as np
from time import sleep
from typing import List, Dict, Union, Optional
from sklearn.preprocessing import RobustScaler
from helpers.all_keys import *

# Konfiguracja API
OPEN_AI_API = False

VOYAGE_AI_API = False

GEMINI_AI_API = True 





import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
import gc

from helpers.polarization_embedings import generate_embedding, generate_embedding_ollama, generate_embedding_gemini, generate_embedding_openai, generate_embedding_voyage


"""
Moduł z narzędziami do analizy semantycznej.
Zawiera funkcje do identyfikacji osi semantycznych i przetwarzania danych.
"""

import numpy as np
from typing import List, Dict, Union, Optional, Tuple



def get_embedding_generator(api_type: str = None, layer_in_model: int = None,  api_key: str = None):
    """
    Zwraca odpowiednią funkcję do generowania embeddingów na podstawie konfiguracji.
    
    Args:
        api_type: Typ API do użycia ('openai', 'voyage', 'gemini', lub None dla lokalnego)
        api_key: Klucz API dla wybranego dostawcy
        
    Returns:
        function: Funkcja do generowania embeddingów
    """
    if api_type == 'openai':
        return lambda text: generate_embedding_openai(text=text, api_key=API_KEY_OPENAI)
    elif api_type == 'voyage':
        return lambda text: generate_embedding_voyage(text=text, api_key=API_KEY_VOYAGE)
    elif api_type == 'gemini':
        return lambda text: generate_embedding_gemini(text=text, api_key=API_KEY_GEMINI)
    else:
        return lambda text: generate_embedding(text=text, layer_index=layer_in_model)
    



def preprocess_semantic_features(semantic_features: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Przetwarza słownik semantic_features, aby zawierał listy słów kluczowych zamiast długich tekstów.
    
    Args:
        semantic_features (Dict[str, List[str]]): Słownik z kategoriami semantycznymi i ich opisami
            
    Returns:
        Dict[str, List[str]]: Przetworzony słownik z kategoriami i listami słów kluczowych
    """
    processed_features = {}
    
    for category, texts in semantic_features.items():
        if len(texts) == 1 and len(texts[0].split()) > 15:
            # To jest długi tekst, zachowaj go jako jest
            processed_features[category] = texts
        else:
            # To już jest lista słów kluczowych, zachowuj bez zmian
            processed_features[category] = texts
            
    return processed_features


def compute_cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Tworzy macierz odległości kosinusowych między embedingami.
    
    Args:
        embeddings (numpy.ndarray): Macierz embedingów
        
    Returns:
        numpy.ndarray: Macierz odległości kosinusowych
    """
    from scipy.spatial.distance import pdist, squareform
    
    if embeddings.shape[0] <= 1:
        return np.zeros((embeddings.shape[0], embeddings.shape[0]))
    
    distances = pdist(embeddings, metric='cosine')
    return squareform(distances)


def calculate_centroid(embeddings: np.ndarray) -> np.ndarray:
    """
    Oblicza centroid (średnią) z listy embeddingów.
    
    Args:
        embeddings: Macierz embeddingów (n_samples, n_features)
        
    Returns:
        Centroid jako wektor numpy
    """
    if len(embeddings) == 0:
        raise ValueError("Lista embeddingów nie może być pusta")
    
    return np.mean(embeddings, axis=0)



def project_point_on_line(point: np.ndarray, line_start: np.ndarray, line_direction: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Rzutuje punkt na linię.
    
    Args:
        point: Punkt do rzutowania
        line_start: Punkt startowy linii
        line_direction: Znormalizowany wektor kierunku linii
        
    Returns:
        Tuple: (punkt_projekcji, odległość_wzdłuż_linii)
    """
    # Wektor od początku linii do punktu
    point_vector = point - line_start
    
    # Oblicz projekcję skalarną
    projection_length = np.dot(point_vector, line_direction)
    
    # Oblicz punkt projekcji
    projection_point = line_start + projection_length * line_direction
    
    return projection_point, projection_length


def calculate_orthogonal_distance(point: np.ndarray, line_start: np.ndarray, line_direction: np.ndarray) -> float:
    """
    Oblicza odległość ortogonalną punktu od linii.
    
    Args:
        point: Punkt
        line_start: Punkt startowy linii
        line_direction: Znormalizowany wektor kierunku linii
        
    Returns:
        Odległość ortogonalna
    """
    projection_point, _ = project_point_on_line(point, line_start, line_direction)
    return np.linalg.norm(point - projection_point)




import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Dict, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

class SVMPolarizer:
    """
    SVM z jądrem liniowym pracujący na PEŁNEJ wymiarowości embeddingów bez żadnej redukcji.
    Zoptymalizowany dla wysokowymiarowych danych (1536, 3072, 5000+ wymiarów).
    """
    
    def __init__(self, random_seed: int = 23, embedding_api: str = None, api_key: str = None, layer_in_model: int = None):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        
        self.embedding_api = embedding_api
        
        print(f"🎯 FULL-DIMENSION LINEAR SVM - bez redukcji wymiarowości")
        print(f"📡 API embeddingów: {embedding_api or 'lokalny model'}")
        print(f"📐 Strategia: pełna wymiarowość + linear kernel SVM + optymalizacja regularyzacji")
        
        # Globalne cache embeddingów
        self.global_embedding_cache = {}
        
        # Inicjalizacja zmiennych
        self.semantic_features = None
        self.axes = {}
        self.axes_data = {}
        self.is_trained = False

        self.layer_in_model = layer_in_model
        
        # Generator embeddingów
        self.embedding_generator = get_embedding_generator(embedding_api, self.layer_in_model, api_key)
        
        # Statystyki wydajności
        self.performance_stats = {
            "embedding_cache_hits": 0,
            "total_embedding_requests": 0,
            "training_times": {},
            "optimization_times": {}
        }
    
    def set_semantic_features(self, semantic_features: Dict[str, List[str]]) -> 'SVMPolarizer':
        """Ustawia kategorie z analizą pełnowymiarową."""
        processed_features = preprocess_semantic_features(semantic_features)
        self.semantic_features = processed_features
        
        print(f"📊 Ustawiono {len(processed_features)} kategorii semantycznych")
        
        # Pre-compute wszystkich embeddingów
        self._precompute_all_embeddings(processed_features)
        
        # Identyfikacja osi
        self.axes = self._identify_binary_axes(self.semantic_features)
        
        return self
    
    def _precompute_all_embeddings(self, semantic_features: Dict[str, List[str]]) -> None:
        """Pre-computuje embeddingi z analizą wymiarowości."""
        print("\n🔄 Pre-computing embeddingów na pełnej wymiarowości...")
        start_time = time.time()
        
        # Zbierz wszystkie unikalne słowa
        all_words = set()
        for phrases in semantic_features.values():
            all_words.update(phrases)
        
        all_words = list(all_words)
        print(f"   Unikalne frazy: {len(all_words)}")
        
        # Batch processing
        batch_size = 20
        
        for i in range(0, len(all_words), batch_size):
            batch = all_words[i:i+batch_size]
            print(f"   Batch {i//batch_size + 1}/{(len(all_words)-1)//batch_size + 1}")
            
            for word in batch:
                if word not in self.global_embedding_cache:
                    try:
                        embedding = self.embedding_generator(word)
                        self.global_embedding_cache[word] = embedding
                        self.performance_stats["total_embedding_requests"] += 1
                    except Exception as e:
                        print(f"❌ Błąd embeddingu dla '{word[:30]}...': {str(e)}")
                        # Fallback - użyj embedding o standardowej wymiarowości
                        if self.global_embedding_cache:
                            example_embedding = list(self.global_embedding_cache.values())[0]
                            empty_embedding = np.zeros_like(example_embedding)
                        else:
                            # Domyślne wymiary dla różnych API
                            if self.embedding_api == "openai":
                                empty_embedding = np.zeros(1536)  # text-embedding-ada-002
                            elif self.embedding_api == "voyage":
                                empty_embedding = np.zeros(1024)  # voyage-large
                            else:
                                empty_embedding = np.zeros(1536)  # fallback
                        self.global_embedding_cache[word] = empty_embedding
                else:
                    self.performance_stats["embedding_cache_hits"] += 1
        
        # Analiza wymiarowości po wczytaniu
        if self.global_embedding_cache:
            sample_embedding = list(self.global_embedding_cache.values())[0]
            embedding_dim = len(sample_embedding)
            print(f"   🎯 Wykryto wymiarowość embeddingów: {embedding_dim}")
            
            # Informacje o API i wymiarowości
            if self.embedding_api == "openai":
                print(f"   📡 OpenAI embeddingi - typowo 1536 wymiarów")
            elif self.embedding_api == "voyage":
                print(f"   📡 Voyage AI embeddingi - typowo 1024 wymiarów")
            elif self.embedding_api == "gemini":
                print(f"   📡 Gemini embeddingi - wymiarowość zmienna")
            else:
                print(f"   📡 Lokalne embeddingi - wymiarowość {embedding_dim}")
            
            # Optymalizacje na podstawie wymiarowości
            if embedding_dim >= 3000:
                print(f"   ⚡ BARDZO WYSOKOWYMIAROWE - aktywne optymalizacje SVM")
            elif embedding_dim >= 1500:
                print(f"   📐 WYSOKOWYMIAROWE - standardowe optymalizacje")
            elif embedding_dim >= 500:
                print(f"   📏 ŚREDNIOWYMIAROWE - podstawowe optymalizacje")
            else:
                print(f"   📊 NISKOWYMIAROWE - minimalne optymalizacje")
        
        duration = time.time() - start_time
        print(f"✅ Pre-computing zakończone w {duration:.1f}s")
    
    def _get_cached_embeddings(self, quality_list: List[str]) -> np.ndarray:
        """Pobiera embeddingi z cache."""
        embeddings = []
        for word in quality_list:
            if word in self.global_embedding_cache:
                embeddings.append(self.global_embedding_cache[word])
                self.performance_stats["embedding_cache_hits"] += 1
            else:
                try:
                    embedding = self.embedding_generator(word)
                    self.global_embedding_cache[word] = embedding
                    embeddings.append(embedding)
                    self.performance_stats["total_embedding_requests"] += 1
                except Exception as e:
                    print(f"❌ Błąd embeddingu: {str(e)}")
                    # Użyj zero embedding o odpowiedniej wymiarowości
                    if embeddings:
                        empty_embedding = np.zeros_like(embeddings[0])
                    else:
                        empty_embedding = np.zeros(1536)  # fallback
                    embeddings.append(empty_embedding)
        
        return np.array(embeddings)
    
    def _analyze_full_dimensionality(self, X_embeddings: np.ndarray, y_labels: List[int], 
                                   axis_name: str) -> Dict:
        """Analiza dla pełnowymiarowych danych."""
        print(f"\n🎯 ANALIZA PEŁNOWYMIAROWA - {axis_name}")
        
        n_samples, n_features = X_embeddings.shape
        print(f"   📐 PEŁNE wymiary: {n_samples} próbek × {n_features} features")
        print(f"   🎯 BEZ redukcji wymiarowości - wykorzystujemy pełną przestrzeń embeddingów")
        
        analysis = {
            "n_samples": n_samples,
            "n_features": n_features,
            "dimensionality_ratio": n_features / n_samples if n_samples > 0 else float('inf'),
            "uses_full_dimensionality": True
        }
        
        # Analiza stosunku wymiarów do próbek
        dim_ratio = analysis["dimensionality_ratio"]
        print(f"   ⚖️  Stosunek features/samples: {dim_ratio:.1f}")
        
        # Klasyfikacja ryzyka dla pełnowymiarowych danych
        if dim_ratio > 50:
            print(f"   🚨 EKSTREMALNE wymiary vs próbki - wymagana bardzo silna regularyzacja")
            analysis["risk_level"] = "extreme"
            analysis["recommended_c_range"] = [0.00001, 0.0001, 0.001, 0.01]
        elif dim_ratio > 20:
            print(f"   🔴 BARDZO WYSOKIE ryzyko overfitting - silna regularyzacja")
            analysis["risk_level"] = "very_high"
            analysis["recommended_c_range"] = [0.0001, 0.001, 0.01, 0.1]
        elif dim_ratio > 10:
            print(f"   🟠 WYSOKIE ryzyko overfitting - umiarkowana regularyzacja")
            analysis["risk_level"] = "high"
            analysis["recommended_c_range"] = [0.001, 0.01, 0.1, 1.0]
        elif dim_ratio > 5:
            print(f"   🟡 ŚREDNIE ryzyko overfitting - lekka regularyzacja")
            analysis["risk_level"] = "medium"
            analysis["recommended_c_range"] = [0.01, 0.1, 1.0, 10.0]
        else:
            print(f"   🟢 NISKIE ryzyko overfitting - standardowe C")
            analysis["risk_level"] = "low"
            analysis["recommended_c_range"] = [0.1, 1.0, 10.0, 100.0]
        
        # Analiza jakości embeddingów na pełnej wymiarowości
        feature_variances = np.var(X_embeddings, axis=0)
        
        # Statystyki wariancji features
        zero_var_count = np.sum(feature_variances < 1e-10)
        low_var_count = np.sum(feature_variances < 0.001)
        high_var_count = np.sum(feature_variances > 1.0)
        
        analysis["variance_stats"] = {
            "zero_variance_features": int(zero_var_count),
            "low_variance_features": int(low_var_count),
            "high_variance_features": int(high_var_count),
            "mean_variance": float(np.mean(feature_variances)),
            "std_variance": float(np.std(feature_variances))
        }
        
        print(f"   📊 Analiza wariancji features:")
        print(f"     Zero-variance: {zero_var_count}")
        print(f"     Low-variance (<0.001): {low_var_count}")
        print(f"     High-variance (>1.0): {high_var_count}")
        print(f"     Średnia wariancja: {np.mean(feature_variances):.6f}")
        
        # Separowalność na pełnej wymiarowości
        pos_mask = np.array(y_labels) == 1
        neg_mask = np.array(y_labels) == -1
        
        pos_embeddings = X_embeddings[pos_mask]
        neg_embeddings = X_embeddings[neg_mask]
        
        if len(pos_embeddings) > 0 and len(neg_embeddings) > 0:
            pos_centroid = np.mean(pos_embeddings, axis=0)
            neg_centroid = np.mean(neg_embeddings, axis=0)
            
            # Euclidean distance między centroidami
            centroid_distance = np.linalg.norm(pos_centroid - neg_centroid)
            
            # Cosine similarity między centroidami
            cos_sim = np.dot(pos_centroid, neg_centroid) / (np.linalg.norm(pos_centroid) * np.linalg.norm(neg_centroid))
            
            analysis["centroid_analysis"] = {
                "euclidean_distance": float(centroid_distance),
                "cosine_similarity": float(cos_sim),
                "cosine_distance": float(1 - cos_sim)
            }
            
            print(f"   🎯 Separowalność centroidów:")
            print(f"     Euclidean distance: {centroid_distance:.4f}")
            print(f"     Cosine similarity: {cos_sim:.4f}")
            print(f"     Cosine distance: {1 - cos_sim:.4f}")
            
            # Dla embeddingów cosine distance często bardziej informacyjny
            if 1 - cos_sim > 0.1:
                print(f"   ✅ Dobra separowalność w przestrzeni cosinusowej")
            elif 1 - cos_sim > 0.05:
                print(f"   ⚠️  Średnia separowalność w przestrzeni cosinusowej")
            else:
                print(f"   ❌ Niska separowalność w przestrzeni cosinusowej")
        
        # Rekomendacje dla pełnowymiarowych danych
        recommendations = []
        
        if dim_ratio > 20:
            recommendations.append("KRYTYCZNE: Użyj bardzo niskich wartości C (< 0.01)")
            recommendations.append("Rozważ dodanie znacznie więcej danych treningowych")
            recommendations.append("Monitoruj overfitting - użyj cross-validation")
        
        if dim_ratio > 10:
            recommendations.append("Użyj regularyzacji jeśli overfitting jest problemem")
            recommendations.append("Rozważ early stopping podczas trenowania")
        
        if zero_var_count > n_features * 0.05:
            recommendations.append(f"Usuń {zero_var_count} features o zerowej wariancji")
        
        if analysis.get("centroid_analysis", {}).get("cosine_distance", 0) < 0.05:
            recommendations.append("Bardzo niska separowalność - może potrzeba lepszych embeddingów")
            recommendations.append("Rozważ użycie innego API embeddingów")
        
        analysis["recommendations"] = recommendations
        
        if recommendations:
            print(f"   💡 REKOMENDACJE PEŁNOWYMIAROWE:")
            for rec in recommendations:
                print(f"     • {rec}")
        
        return analysis
    
    def _optimize_linear_svm(self, X_scaled: np.ndarray, y_labels: List[int], 
                            recommended_c_range: List[float]) -> Tuple[Dict, float]:
        """Optymalizacja SVM z jądrem liniowym dla pełnej wymiarowości."""
        print(f"   🎯 OPTYMALIZACJA LINEAR KERNEL SVM - PEŁNA WYMIAROWOŚĆ")
        
        n_samples, n_features = X_scaled.shape
        print(f"   📐 Optymalizacja na {n_features} wymiarach (bez redukcji)")
        
        # Użyj rekomendowanych wartości C na podstawie analizy wymiarowości
        c_values = recommended_c_range.copy()
        
        # Dodaj dodatkowe wartości dla dokładniejszej optymalizacji
        if max(c_values) >= 1.0:
            # Dla niskiego ryzyka overfitting - szerszy zakres
            additional_c = [50.0, 100.0, 500.0]
            c_values.extend([c for c in additional_c if c not in c_values])
        else:
            # Dla wysokiego ryzyka overfitting - więcej niskich wartości
            additional_c = [max(c_values) * 2, max(c_values) * 5]
            c_values.extend([c for c in additional_c if c not in c_values])
        
        c_values = sorted(c_values)
        print(f"   🔧 Testowanie {len(c_values)} wartości C: {c_values}")
        print(f"   🎯 Używam SVC z kernel='linear' (pełne funkcjonalności + probability)")
        
        # Cross-validation setup
        cv_folds = min(5, n_samples // 3) if n_samples >= 15 else max(2, n_samples // 2)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_seed)
        print(f"   📊 Cross-validation: {cv_folds} folds")
        
        best_score = 0
        best_c = c_values[len(c_values)//2]  # Start z środkowej wartości
        all_results = []
        
        for c in c_values:
            try:
                # Standardowy SVC z jądrem liniowym
                model = SVC(
                    kernel='linear', 
                    C=c, 
                    random_state=self.random_seed, 
                    probability=True,
                    max_iter=10000  # Zwiększone dla stabilności
                )
                
                # Cross-validation
                scores = cross_val_score(model, X_scaled, y_labels, cv=cv, scoring='accuracy')
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                
                all_results.append({
                    "C": c,
                    "mean_score": avg_score,
                    "std_score": std_score
                })
                
                print(f"     C={c:10.6f}: {avg_score:.4f} (+/- {std_score*2:.4f})")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_c = c
                    
            except Exception as e:
                print(f"     C={c:10.6f}: Błąd - {str(e)}")
                continue
        
        print(f"\n   🏆 NAJLEPSZA KONFIGURACJA:")
        print(f"     C = {best_c}")
        print(f"     CV Score = {best_score:.4f}")
        print(f"     Model type = SVC (linear kernel)")
        
        # Analiza stabilności wyników
        if len(all_results) > 2:
            scores = [r["mean_score"] for r in all_results]
            score_range = max(scores) - min(scores)
            print(f"   📊 Stabilność wyników:")
            print(f"     Zakres scores: {score_range:.4f}")
            print(f"     Średni score: {np.mean(scores):.4f}")
            
            if score_range < 0.05:
                print(f"   ✅ Wysoka stabilność - wyniki mało zależne od C")
            elif score_range < 0.15:
                print(f"   ⚠️  Średnia stabilność - C ma umiarkowany wpływ")
            else:
                print(f"   🚨 Niska stabilność - C ma duży wpływ na wyniki")
        
        # Zwróć najlepsze parametry
        best_params = {
            "kernel": "linear",
            "C": best_c,
            "probability": True,
            "max_iter": 10000
        }
        
        return best_params, best_score
    
    def train(self) -> 'SVMPolarizer':
        """Trenowanie na pełnej wymiarowości."""


        if not self.semantic_features or not self.axes:
            raise ValueError("Musisz najpierw ustawić kategorie semantyczne.")
        
        print(f"\n🎯 FULL-DIMENSION LINEAR SVM - TRENOWANIE {len(self.axes)} OSI")
        print("📐 PEŁNA WYMIAROWOŚĆ EMBEDDINGÓW - bez redukcji")
        
        total_start_time = time.time()
        
        for axis_name, axis_info in self.axes.items():
            print(f"\n" + "="*70)
            print(f"🎯 OSI: {axis_name}")
            print("="*70)
            
            axis_start_time = time.time()
            
            pos_category = axis_info['positive']
            neg_category = axis_info['negative']
            
            pos_words = self.semantic_features[pos_category]
            neg_words = self.semantic_features[neg_category]
            
            X_words = pos_words + neg_words
            y_labels = [1] * len(pos_words) + [-1] * len(neg_words)
            
            print(f"📊 Dane: {len(pos_words)} pozytywnych + {len(neg_words)} negatywnych")
            
            # Walidacja danych
            # if len(X_words) < 6:
            #     print("❌ Za mało danych dla SVM (minimum 6 próbek)")
            #     pass
            
            # if len(pos_words) < 2 or len(neg_words) < 2:
            #     print("❌ Za mało próbek w jednej z klas (minimum 2 w każdej)")
            #     pass
            
            # Embeddingi z cache - PEŁNA WYMIAROWOŚĆ
            X_embeddings = self._get_cached_embeddings(X_words)
            print(f"📐 PEŁNE embeddingi: {X_embeddings.shape}")
            
            # ANALIZA PEŁNOWYMIAROWA
            full_dim_analysis = self._analyze_full_dimensionality(X_embeddings, y_labels, axis_name)
            
            # SKALOWANIE (bez redukcji wymiarów!)
            print(f"   🔧 Skalowanie na pełnej wymiarowości...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_embeddings)
            print(f"   ✅ Skalowanie zakończone: {X_scaled.shape}")
            
            # OPTYMALIZACJA SVM
            recommended_c = full_dim_analysis["recommended_c_range"]
            opt_start_time = time.time()
            best_params, best_cv_score = self._optimize_linear_svm(
                X_scaled, y_labels, recommended_c)
            opt_duration = time.time() - opt_start_time
            
            self.performance_stats["optimization_times"][axis_name] = opt_duration
            
            # TRENOWANIE FINALNEGO MODELU
            print(f"   🎯 Trenowanie finalnego modelu na {X_scaled.shape[1]} wymiarach...")
            
            try:
                final_model = SVC(
                    kernel='linear',
                    C=best_params["C"], 
                    random_state=self.random_seed, 
                    probability=best_params["probability"],
                    max_iter=best_params["max_iter"]
                )
                print(1)
                final_model.fit(X_scaled, y_labels)
                print(2)

                # Ewaluacja finalna
                cv_scores = cross_val_score(final_model, X_scaled, y_labels, cv=3, scoring='accuracy')
                print(3)

                final_cv_score = np.mean(cv_scores)
                print(4)

                cv_std = np.std(cv_scores)
                print(5)

                print(f"   ✅ Final CV accuracy: {final_cv_score:.4f} (+/- {cv_std*2:.4f})")
                
                # Informacje o modelu na pełnej wymiarowości
                support_ratio = sum(final_model.n_support_) / len(X_scaled)
                print(f"   🎯 Support vectors: {final_model.n_support_} ({support_ratio:.1%})")
                
                # Analiza hiperpłaszczyzny na pełnej wymiarowości
                weights = final_model.coef_[0]
                bias = final_model.intercept_[0]
                weights_norm = np.linalg.norm(weights)
                margin = 1.0 / weights_norm if weights_norm > 0 else 0
                
                print(f"   📏 Hiperpłaszczyzna na {len(weights)} wymiarach:")
                print(f"     Norma wag: {weights_norm:.6f}")
                print(f"     Bias: {bias:.6f}")
                print(f"     Margin: {margin:.6f}")
                
                # Analiza najważniejszych wymiarów
                top_weights_idx = np.argsort(np.abs(weights))[-10:]
                top_weights_values = weights[top_weights_idx]
                print(f"   🔝 Top 5 najważniejszych wymiarów:")
                for i, (idx, val) in enumerate(zip(top_weights_idx[-5:], top_weights_values[-5:])):
                    print(f"     {i+1}. Wymiar {idx}: {val:.6f}")
                
            except Exception as e:
                print(f"   ❌ Błąd trenowania: {str(e)}")
                final_model = None
                final_cv_score = 0.0
                cv_std = 0.0
            
            # Zapisz wszystkie dane
            axis_duration = time.time() - axis_start_time
            self.performance_stats["training_times"][axis_name] = axis_duration
            
            axis_data = {
                "axis_name": axis_name,
                "labels": {
                    "positive": pos_category,
                    "negative": neg_category
                },
                
                # Model i preprocessing
                "svm_model": final_model,
                "scaler": scaler,
                "svm_params": best_params,
                "cv_score": final_cv_score,
                "cv_std": cv_std,
                
                # Pełnowymiarowa analiza
                "full_dim_analysis": full_dim_analysis,
                "embedding_dimensions": X_embeddings.shape[1],
                "uses_full_dimensionality": True,
                
                # Dane treningowe
                "training_words": X_words,
                "training_labels": y_labels,
                "n_positive": len(pos_words),
                "n_negative": len(neg_words),
                
                # Performance
                "training_time": axis_duration,
                "optimization_time": opt_duration,
                "kernel_type": "linear"
            }
            
            # Model-specific metadata
            if final_model is not None:
                axis_data["hyperplane_weights"] = final_model.coef_[0].tolist()
                axis_data["hyperplane_bias"] = float(final_model.intercept_[0])
                axis_data["hyperplane_weights_norm"] = float(np.linalg.norm(final_model.coef_[0]))
                axis_data["margin"] = float(1.0 / np.linalg.norm(final_model.coef_[0])) if np.linalg.norm(final_model.coef_[0]) > 0 else 0.0
                axis_data["n_support_vectors"] = final_model.n_support_.tolist()
                axis_data["support_vector_ratio"] = float(sum(final_model.n_support_) / len(X_scaled))
            
            self.axes_data[axis_name] = axis_data
            
            print(f"   ⏱️  Czas trenowania: {axis_duration:.1f}s")
            print(f"   🎯 Finalny CV score: {final_cv_score:.4f}")
            
            # ANALIZA PROBLEMÓW PEŁNOWYMIAROWYCH
            if final_cv_score < 0.7:
                print(f"   🚨 PROBLEMY PEŁNOWYMIAROWE:")
                if full_dim_analysis["dimensionality_ratio"] > 20:
                    print(f"     • EKSTREMALNE wymiary/próbki = {full_dim_analysis['dimensionality_ratio']:.1f}")
                if full_dim_analysis["risk_level"] in ["extreme", "very_high"]:
                    print(f"     • Wysokie ryzyko overfitting na pełnej wymiarowości")
                
                print(f"     💡 DZIAŁANIA:")
                for rec in full_dim_analysis["recommendations"]:
                    print(f"       • {rec}")
        
        total_duration = time.time() - total_start_time
        
        self.is_trained = True
        print(f"\n🎉 FULL-DIMENSION LINEAR SVM ZAKOŃCZONE!")
        print(f"⏱️  Całkowity czas: {total_duration:.1f}s")
        print(f"📐 Wszystkie modele trenowane na PEŁNEJ wymiarowości embeddingów")
        
        # GLOBALNE PODSUMOWANIE PEŁNOWYMIAROWE
        self._print_full_dimension_summary()
        
        return self
    
    def _print_full_dimension_summary(self) -> None:
        """Podsumowanie dla pełnowymiarowych danych."""
        print(f"\n📐 PODSUMOWANIE PEŁNOWYMIAROWE")
        print("="*60)
        
        all_scores = []
        all_dimensions = []
        all_margins = []
        risk_levels = []
        
        for axis_name, axis_data in self.axes_data.items():
            cv_score = axis_data.get("cv_score", 0.0)
            all_scores.append(cv_score)
            
            dimensions = axis_data.get("embedding_dimensions", 0)
            all_dimensions.append(dimensions)
            
            margin = axis_data.get("margin", 0.0)
            all_margins.append(margin)
            
            full_dim_analysis = axis_data.get("full_dim_analysis", {})
            risk_level = full_dim_analysis.get("risk_level", "unknown")
            risk_levels.append(risk_level)
            
            print(f"\n📐 {axis_name}:")
            print(f"   CV Score: {cv_score:.4f}")
            print(f"   Wymiary: {dimensions} (pełne)")
            print(f"   Margin: {margin:.6f}")
            print(f"   Ryzyko: {risk_level}")
            print(f"   C: {axis_data['svm_params'].get('C', 'N/A')}")
        
        if all_scores:
            print(f"\n📊 STATYSTYKI GLOBALNE:")
            print(f"   Średnia accuracy: {np.mean(all_scores):.4f}")
            print(f"   Min accuracy: {np.min(all_scores):.4f}")
            print(f"   Max accuracy: {np.max(all_scores):.4f}")
            print(f"   Standardowa wymiarowość: {all_dimensions[0] if all_dimensions else 'N/A'}")
            print(f"   Średni margin: {np.mean(all_margins):.6f}")
            
            # Analiza ryzyka
            from collections import Counter
            risk_counts = Counter(risk_levels)
            print(f"\n🎯 ROZKŁAD RYZYKA OVERFITTING:")
            for risk, count in risk_counts.items():
                print(f"   {risk}: {count} osi")
            
            # Zalecenia na podstawie globalnych wyników
            high_risk_count = sum(1 for risk in risk_levels if risk in ["extreme", "very_high", "high"])
            
            if high_risk_count > len(self.axes_data) * 0.7:
                print(f"\n🚨 GLOBALNE OSTRZEŻENIE:")
                print(f"   Większość osi ma wysokie ryzyko overfitting")
                print(f"   Rozważ dodanie więcej danych treningowych")
                print(f"   Użyj krzyżowej walidacji podczas ewaluacji")
            
            if np.mean(all_scores) > 0.8:
                print(f"\n✅ DOSKONAŁE WYNIKI na pełnej wymiarowości!")
                print(f"   Linear SVM dobrze radzi sobie z wysokowymiarowymi embeddingami")
            elif np.mean(all_scores) < 0.6:
                print(f"\n⚠️  NISKIE WYNIKI - może potrzeba:")
                print(f"   • Lepszych danych treningowych")
                print(f"   • Innego API embeddingów")
                print(f"   • Więcej danych na klasę")
    
    def calculate_polarization(self, text: str = None, embedding: np.ndarray = None, 
                              axis_name: str = None) -> Dict:
        """Analiza polaryzacji na pełnej wymiarowości."""
        if text:
            print(f"🔍 Analiza pełnowymiarowa: {text[:100]}...")
        
        if not self.is_trained:
            raise ValueError("Model nie został wytrenowany.")
        
        # Embedding
        if embedding is None and text is not None:
            if text in self.global_embedding_cache:
                embedding = self.global_embedding_cache[text]
                self.performance_stats["embedding_cache_hits"] += 1
            else:
                embedding = self.embedding_generator(text)
                self.global_embedding_cache[text] = embedding
                self.performance_stats["total_embedding_requests"] += 1
        elif embedding is None:
            raise ValueError("Musisz podać tekst lub embedding")
        
        X_test = np.array(embedding).reshape(1, -1)
        axes_to_analyze = [axis_name] if axis_name else list(self.axes_data.keys())
        
        results = {}
        
        for axis in axes_to_analyze:
            if axis not in self.axes_data:
                continue
                
            axis_data = self.axes_data[axis]
            svm_model = axis_data.get("svm_model")
            scaler = axis_data.get("scaler")
            
            if svm_model is None or scaler is None:
                continue
            
            try:
                # SKALOWANIE NA PEŁNEJ WYMIAROWOŚCI
                X_scaled = scaler.transform(X_test)
                print(f"     Skalowanie na {X_scaled.shape[1]} wymiarach")
                
                # PREDYKCJA PEŁNOWYMIAROWA
                svm_prediction = svm_model.predict(X_scaled)[0]
                decision_value = svm_model.decision_function(X_scaled)[0]
                probabilities = svm_model.predict_proba(X_scaled)[0]
                
                # Interpretacja
                sigmoid_position = 2 / (1 + np.exp(-decision_value)) - 1
                
                direction = "positive" if svm_prediction == 1 else "negative"
                direction_label = (axis_data["labels"]["positive"] if svm_prediction == 1 
                                 else axis_data["labels"]["negative"])
                strength = abs(sigmoid_position)
                
                # Analiza wkładu wymiarów (dla linear SVM)
                weights = svm_model.coef_[0]
                feature_contributions = X_scaled.flatten() * weights
                
                # Top contributing dimensions
                top_contrib_idx = np.argsort(np.abs(feature_contributions))[-10:]
                top_contributions = {
                    "indices": top_contrib_idx.tolist(),
                    "values": feature_contributions[top_contrib_idx].tolist(),
                    "weights": weights[top_contrib_idx].tolist()
                }
                
                # Wynik z pełnowymiarowymi metrykami
                results[axis] = {
                    "axis_name": axis,
                    "labels": axis_data["labels"],
                    
                    # Predykcja
                    "svm_prediction": int(svm_prediction),
                    "decision_value": float(decision_value),
                    "probabilities": {
                        "negative": float(probabilities[0]),
                        "positive": float(probabilities[1])
                    },
                    
                    # Interpretacja
                    "direction": direction,
                    "direction_label": direction_label,
                    "strength": float(strength),
                    "normalized_position": float(sigmoid_position),
                    "confidence": float(max(probabilities)),
                    "distance_from_hyperplane": float(abs(decision_value)),
                    
                    # Pełnowymiarowe informacje
                    "full_dimensionality": {
                        "embedding_dimensions": axis_data["embedding_dimensions"],
                        "uses_full_dimensionality": True,
                        "dimensionality_ratio": axis_data["full_dim_analysis"]["dimensionality_ratio"],
                        "top_contributing_dimensions": top_contributions,
                        "hyperplane_norm": axis_data.get("hyperplane_weights_norm", 0.0)
                    },
                    
                    # Model quality
                    "model_quality": {
                        "cv_score": axis_data["cv_score"],
                        "risk_level": axis_data["full_dim_analysis"]["risk_level"],
                        "margin": axis_data.get("margin", 0.0),
                        "support_vector_ratio": axis_data.get("support_vector_ratio", 0.0)
                    },
                    
                    # Technical info
                    "kernel_type": "linear",
                    "regularization_C": axis_data["svm_params"].get("C", 1.0)
                }
                
                # Ostrzeżenia pełnowymiarowe
                warnings = []
                
                risk_level = axis_data["full_dim_analysis"]["risk_level"]
                if risk_level in ["extreme", "very_high"]:
                    warnings.append("Wysokie ryzyko overfitting na pełnej wymiarowości")
                
                if axis_data["cv_score"] < 0.7:
                    warnings.append("Niska accuracy - wyniki mogą być nieprecyzyjne")
                
                if axis_data["full_dim_analysis"]["dimensionality_ratio"] > 50:
                    warnings.append("Ekstremalna wymiarowość vs próbki - bardzo ryzykowne")
                
                # Sprawdź czy główne wymiary mają sens
                top_weights = np.abs(weights[top_contrib_idx[-5:]])
                if np.std(top_weights) / np.mean(top_weights) < 0.1:
                    warnings.append("Bardzo równomierny rozkład wag - może brak wyraźnego wzorca")
                
                if warnings:
                    results[axis]["warnings"] = warnings
                
                # Szczegółowy output
                dims_info = results[axis]["full_dimensionality"]
                print(f"  📐 {axis}: {direction_label} (siła: {strength:.3f})")
                print(f"     Pełne wymiary: {dims_info['embedding_dimensions']}")
                print(f"     Top wymiary: {top_contrib_idx[-3:].tolist()}")
                if warnings:
                    print(f"     ⚠️  Ostrzeżenia: {len(warnings)}")
                
            except Exception as e:
                print(f"❌ Błąd predykcji pełnowymiarowej dla osi {axis}: {str(e)}")
                continue
        
        return results
    
    def analyze_feature_importance(self, axis_name: str, top_n: int = 20) -> Dict:
        """Analiza ważności features dla konkretnej osi na pełnej wymiarowości."""
        if axis_name not in self.axes_data:
            return {"error": f"Oś '{axis_name}' nie istnieje"}
        
        axis_data = self.axes_data[axis_name]
        svm_model = axis_data.get("svm_model")
        
        if svm_model is None:
            return {"error": f"Brak modelu SVM dla osi '{axis_name}'"}
        
        # Pobierz wagi hiperpłaszczyzny
        weights = svm_model.coef_[0]
        bias = svm_model.intercept_[0]
        
        # Sortuj według ważności (absolute value)
        importance_idx = np.argsort(np.abs(weights))[::-1]
        
        # Top N najważniejszych features
        top_features = {
            "axis_name": axis_name,
            "total_dimensions": len(weights),
            "bias": float(bias),
            "weights_norm": float(np.linalg.norm(weights)),
            "top_features": []
        }
        
        for i in range(min(top_n, len(weights))):
            idx = importance_idx[i]
            top_features["top_features"].append({
                "dimension_index": int(idx),
                "weight": float(weights[idx]),
                "absolute_weight": float(abs(weights[idx])),
                "rank": i + 1
            })
        
        # Statystyki rozkładu wag
        top_features["weight_statistics"] = {
            "mean_weight": float(np.mean(weights)),
            "std_weight": float(np.std(weights)),
            "min_weight": float(np.min(weights)),
            "max_weight": float(np.max(weights)),
            "mean_abs_weight": float(np.mean(np.abs(weights))),
            "weight_concentration": float(np.sum(np.abs(weights[:top_n])) / np.sum(np.abs(weights)))
        }
        
        return top_features
    
    def get_full_dimension_report(self) -> Dict:
        """Szczegółowy raport o pracy na pełnej wymiarowości."""
        if not self.is_trained:
            return {"error": "Model nie został wytrenowany"}
        
        report = {
            "overview": {
                "strategy": "full_dimensionality",
                "total_axes": len(self.axes_data),
                "embedding_api": self.embedding_api,
                "dimensionality_reduction": "none",
                "model_type": "SVC_linear_kernel"
            },
            "dimensionality_analysis": {},
            "model_performance": {},
            "computational_efficiency": {},
            "recommendations": []
        }
        
        all_scores = []
        all_dimensions = []
        all_margins = []
        all_training_times = []
        risk_levels = []
        
        for axis_name, axis_data in self.axes_data.items():
            # Analiza wymiarowości
            full_dim_analysis = axis_data["full_dim_analysis"]
            report["dimensionality_analysis"][axis_name] = {
                "embedding_dimensions": axis_data["embedding_dimensions"],
                "dimensionality_ratio": full_dim_analysis["dimensionality_ratio"],
                "risk_level": full_dim_analysis["risk_level"],
                "variance_stats": full_dim_analysis["variance_stats"]
            }
            
            # Performance modelu
            report["model_performance"][axis_name] = {
                "cv_score": axis_data["cv_score"],
                "cv_std": axis_data.get("cv_std", 0.0),
                "margin": axis_data.get("margin", 0.0),
                "regularization_C": axis_data["svm_params"].get("C", 1.0),
                "support_vector_ratio": axis_data.get("support_vector_ratio", 0.0)
            }
            
            # Zbieranie danych do analiz
            all_scores.append(axis_data["cv_score"])
            all_dimensions.append(axis_data["embedding_dimensions"])
            all_margins.append(axis_data.get("margin", 0.0))
            all_training_times.append(axis_data["training_time"])
            risk_levels.append(full_dim_analysis["risk_level"])
        
        # Efficiency analysis
        report["computational_efficiency"] = {
            "avg_training_time": float(np.mean(all_training_times)),
            "total_training_time": float(np.sum(all_training_times)),
            "avg_dimensions_used": float(np.mean(all_dimensions)),
            "embedding_cache_hit_rate": self.performance_stats["embedding_cache_hits"] / 
                                      max(self.performance_stats["total_embedding_requests"] + 
                                          self.performance_stats["embedding_cache_hits"], 1)
        }
        
        # Globalne statystyki
        if all_scores:
            report["overview"]["avg_cv_score"] = float(np.mean(all_scores))
            report["overview"]["min_cv_score"] = float(np.min(all_scores))
            report["overview"]["max_cv_score"] = float(np.max(all_scores))
            report["overview"]["score_std"] = float(np.std(all_scores))
        
        # Rekomendacje
        recommendations = []
        
        # Analiza ryzyka
        high_risk_count = sum(1 for risk in risk_levels if risk in ["extreme", "very_high"])
        if high_risk_count > len(self.axes_data) * 0.5:
            recommendations.append("Więcej niż połowa osi ma wysokie ryzyko overfitting")
            recommendations.append("Rozważ dodanie więcej danych treningowych")
        
        # Analiza accuracy
        if np.mean(all_scores) < 0.7:
            recommendations.append("Niska średnia accuracy - sprawdź jakość danych")
            recommendations.append("Rozważ inne API embeddingów")
        
        # Analiza margin
        if np.mean(all_margins) < 0.01:
            recommendations.append("Bardzo małe margins - modele mogą być niestabilne")
            recommendations.append("Zwiększ regularyzację (zmniejsz C)")
        
        # Analiza efektywności
        avg_dim = np.mean(all_dimensions)
        if avg_dim > 3000:
            recommendations.append("Bardzo wysokowymiarowe embeddingi - monitoruj wydajność")
        
        recommendations.append("SVC z linear kernel zapewnia pełną funkcjonalność probability")
        recommendations.append("Brak konieczności dodatkowej kalibracji prawdopodobieństw")
        
        report["recommendations"] = recommendations
        
        return report
    
    def _identify_binary_axes(self, semantic_features: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
        """Identyfikacja binarnych osi."""
        axes = {}
        categories = list(semantic_features.keys())
        processed = set()
        
        suffixes = [
            ('_beneficial', '_harmful'),
            ('_effective', '_ineffective'), 
            ('_strong', '_weak'),   # Dostosowane do dostarczonych danych
            ('_positive', '_negative'), 
            ('_respected', '_restricted'),
            ('_progressive', '_traditional'),   # Dostosowane do dostarczonych danych
            ('_regulated', '_free_market'),
            ('_base', '_opposite')
        ]


        
        for category in categories:
            if category in processed:
                continue
            
            for pos_suffix, neg_suffix in suffixes:
                if category.endswith(pos_suffix):
                    base = category[:-len(pos_suffix)]
                    opposite = base + neg_suffix
                    if opposite in categories:
                        axes[base] = {'positive': category, 'negative': opposite}
                        processed.update([category, opposite])
                        print(f"✅ Oś pełnowymiarowa: {base} ({category} ↔ {opposite})")
                        break
                elif category.endswith(neg_suffix):
                    base = category[:-len(neg_suffix)]
                    opposite = base + pos_suffix
                    if opposite in categories:
                        axes[base] = {'positive': opposite, 'negative': category}
                        processed.update([category, opposite])
                        print(f"✅ Oś pełnowymiarowa: {base} ({opposite} ↔ {category})")
                        break
        
        print(f"📐 Zidentyfikowano {len(axes)} osi dla pełnowymiarowego SVM z linear kernel")
        return axes