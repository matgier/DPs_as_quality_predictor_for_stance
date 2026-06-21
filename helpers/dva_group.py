import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

from helpers.polarization_embedings import generate_embedding_gemini, generate_embedding_openai, API_KEY_GEMINI, API_KEY_OPENAI, generate_embedding, generate_embedding_ollama, generate_embedding_voyage, API_KEY_VOYAGE


def perform_extended_dva_analysis(semantic_features, layer_in_model=None,  method=None, name_of_the_results="results_semantic_properities", name_of_the_simulation=None):
    """
    Przeprowadza rozszerzoną analizę DVA dla wszystkich tekstów w każdej kategorii.
    
    Args:
        semantic_features: Słownik z tekstami w formacie:
          {"category_level": [text1, text2, ...], ...}
        method: Metoda generowania embeddingów ('gemini', 'openai', 'voyage' lub inna)
        model: Model dla lokalnego generowania embeddingów
        
    Returns:
        Tuple: (all_embeddings_dict, extended_dva_results)
    """
    # 1. Generuj embeddingi dla wszystkich tekstów
    all_embeddings_dict = {}
    
    for key, texts in semantic_features.items():
        if not texts:  # Sprawdź czy lista tekstów nie jest pusta
            print(f"Brak tekstów dla klucza {key}")
            continue
            
        print(f"----> Przetwarzanie {key}: {len(texts)} tekstów")
        embeddings_list = []
        
        for i, text in enumerate(texts):
            print(f"  Tekst {i+1}/{len(texts)}: {text[:50]}...")
            
            if method == 'gemini':
                embedding = generate_embedding_gemini(text=text, api_key=API_KEY_GEMINI)
            elif method == 'openai':
                embedding = generate_embedding_openai(text=text, api_key=API_KEY_OPENAI)
            elif method == 'voyage':
                embedding = generate_embedding_voyage(text=text, api_key=API_KEY_VOYAGE)
            elif method == 'ollama':
                embedding = generate_embedding_ollama(text=text)
            else:
                embedding = generate_embedding(text=text, layer_index=layer_in_model)

            
            # Sprawdź czy embedding został poprawnie wygenerowany
            if embedding is None or len(embedding) == 0:
                print(f"    UWAGA: Embedding dla tekstu {i+1} jest pusty lub None!")
                continue
            


            embeddings_list.append(embedding)
            print(f"    - Wygenerowano embedding o wymiarze {len(embedding)}")
        
        all_embeddings_dict[key] = embeddings_list
        print(f"  Łącznie wygenerowano {len(embeddings_list)} embeddingów dla {key}")
    
    # 2. Przeprowadź rozszerzoną analizę DVA
    extended_dva_results = extended_dva_analysis(all_embeddings_dict, name_of_the_simulation)
    
    name_of_the_generated_results_with_method = f"{name_of_the_results}_{method}" 
    # 3. Wizualizuj wyniki z boxplotami
    visualize_extended_dva_results(extended_dva_results, name_of_the_generated_results_with_method)
    
    return all_embeddings_dict, extended_dva_results





def extended_dva_analysis(embeddings_dict: Dict[str, List[np.ndarray]], name_of_the_simulation: str) -> Dict[str, Any]:
    """
    Przeprowadza analizę DVA dla wielu embeddingów w każdej kategorii.
    
    Args:
        embeddings_dict: Słownik z listami embeddingów dla każdej kategorii
        
    Returns:
        Dict zawierający wyniki analizy z wieloma pomiarami dla każdej domeny
    """
    results = {}
    
    # categories = {
    #      "vaccine_safety": ["beneficial", "harmful"],
    #      "vaccine_efficacy": ["effective", "ineffective"],
    #      "vaccination_obligation": ["beneficial", "harmful"]
    # }


    # categories = {
    #      "vaccine_stance": ["base", "opposite"],

    # }

    categories = {
         "government_intervention": ["base", "opposite"],

    }

    # categories = {
    #     "government_role": ["strong", "weak"],
    #     "social_policy": ["progressive", "traditional"],
    #     "economic_policy": ["regulated", "free_market"]
    # }



    # categories = {
    #     "care": ["beneficial", "harmful"],
    #     "fairness": ["beneficial", "harmful"],
    #     "liberty": ["beneficial", "harmful"],
    #     "authority": ["strong", "weak"],
    #     "sanctity": ["beneficial", "harmful"],
    #     "loyalty": ["beneficial", "harmful"]
    # }

    
    # categories = {
    #     "vaccine_effect": ["effective", "ineffective"],
    #     "institutions_trust": ["positive", "negative"],
    #     "vaccine_obligation": ["positive", "negative"]
    # }

    

    # categories = {
    #     "vaccine_safety": ["beneficial", "harmful"],
    #     "individual_choice": ["positive", "negative"],
    #     "medical_authority": ["positive", "negative"]
    # }
    

    # categories = {
    #     "vaccine_safe": ["positive", "negative"],
    #     "vaccine_effect": ["effective", "ineffective"],
    #     "vaccine_science": ["positive", "negative"]
    # }
    print("\n=== ROZSZERZONA ANALIZA DVA (GEOMETRYCZNA) ===\n")
    
    for domain, levels in categories.items():
        print(f"\nDomena: {domain}")
        
        pos_key = f"{domain}_{levels[0]}"
        neg_key = f"{domain}_{levels[1]}"
        
        if pos_key not in embeddings_dict or neg_key not in embeddings_dict:
            print(f"  - Brak embeddingów dla domeny {domain}")
            continue
        
        pos_embeddings = embeddings_dict[pos_key]
        neg_embeddings = embeddings_dict[neg_key]
        
        if not pos_embeddings or not neg_embeddings:
            print(f"  - Puste listy embeddingów dla domeny {domain}")
            continue
        
        print(f"  - Liczba embeddingów {levels[0]}: {len(pos_embeddings)}")
        print(f"  - Liczba embeddingów {levels[1]}: {len(neg_embeddings)}")
        
        # Listy do zbierania wyników dla wszystkich par
        cos_similarities = []
        projection_differences = []
        pos_projections = []
        neg_projections = []
        spearman_correlations = []
        pearson_correlations = []
        
        # Analizuj wszystkie możliwe pary między pos i neg embeddingami
        for i, pos_embedding in enumerate(pos_embeddings):
            for j, neg_embedding in enumerate(neg_embeddings):
                pos_embedding = np.array(pos_embedding)
                neg_embedding = np.array(neg_embedding)
                
                # # Wektor różnicy
                # diff_vector = pos_embedding - neg_embedding
                
                # # Podobieństwo kosinusowe
                # cos_similarity = np.dot(pos_embedding, neg_embedding)
                # cos_similarities.append(cos_similarity)
                
                # # Projekcje na wektor różnicy
                # pos_projection = np.dot(pos_embedding, diff_vector)
                # neg_projection = np.dot(neg_embedding, diff_vector)
                
                # pos_projections.append(pos_projection)
                # neg_projections.append(neg_projection)
                
                # # Różnica projekcji
                # projection_difference = pos_projection - neg_projection
                # projection_differences.append(projection_difference)


                # Wektor różnicy
                diff_vector = pos_embedding - neg_embedding
                diff_norm = np.linalg.norm(diff_vector)

                # Podobieństwo kosinusowe (dla info)
                cos_similarity = np.dot(pos_embedding, neg_embedding)
                cos_similarities.append(cos_similarity)
                eps = 1e-12  # zabezpieczenie przed dzieleniem przez 0


                # Jednostkowy kierunek (oś różnicy)
                if diff_norm > eps:
                    u = diff_vector / diff_norm
                else:
                    u = np.zeros_like(diff_vector)  # wektory identyczne → brak kierunku

                # Projekcje na jednostkowy kierunek
                pos_projection = np.dot(pos_embedding, u)
                neg_projection = np.dot(neg_embedding, u)

                pos_projections.append(pos_projection)
                neg_projections.append(neg_projection)

                # Różnica projekcji
                projection_difference = pos_projection - neg_projection
                projection_differences.append(projection_difference)
                
                # Korelacje
                try:
                    spearman_corr, _ = spearmanr(pos_embedding, neg_embedding)
                    pearson_corr, _ = pearsonr(pos_embedding, neg_embedding)
                    spearman_correlations.append(spearman_corr)
                    pearson_correlations.append(pearson_corr)
                except Exception as e:
                    print(f"    Błąd przy obliczaniu korelacji dla pary {i+1},{j+1}: {e}")
                    spearman_correlations.append(np.nan)
                    pearson_correlations.append(np.nan)
        
        # Oblicz statystyki opisowe
        results[domain] = {
            'cos_similarities': cos_similarities,
            'projection_differences': projection_differences,
            'pos_projections': pos_projections,
            'neg_projections': neg_projections,
            'spearman_correlations': spearman_correlations,
            'pearson_correlations': pearson_correlations,
            # Statystyki opisowe
            'cos_similarity_stats': {
                'mean': np.mean(cos_similarities),
                'std': np.std(cos_similarities),
                'median': np.median(cos_similarities),
                'min': np.min(cos_similarities),
                'max': np.max(cos_similarities)
            },
            'projection_diff_stats': {
                'mean': np.mean(projection_differences),
                'std': np.std(projection_differences),
                'median': np.median(projection_differences),
                'min': np.min(projection_differences),
                'max': np.max(projection_differences)
            },
            'spearman_stats': {
                'mean': np.nanmean(spearman_correlations),
                'std': np.nanstd(spearman_correlations),
                'median': np.nanmedian(spearman_correlations),
                'min': np.nanmin(spearman_correlations),
                'max': np.nanmax(spearman_correlations)
            },
            'pearson_stats': {
                'mean': np.nanmean(pearson_correlations),
                'std': np.nanstd(pearson_correlations),
                'median': np.nanmedian(pearson_correlations),
                'min': np.nanmin(pearson_correlations),
                'max': np.nanmax(pearson_correlations)
            },
            'n_pairs': len(cos_similarities)
        }
        
        # Wyświetl podsumowanie statystyk
        print(f"  - Liczba analizowanych par: {len(cos_similarities)}")
        print(f"  - Średnie podobieństwo kosinusowe: {np.mean(cos_similarities):.4f} ± {np.std(cos_similarities):.4f}")
        print(f"  - Średnia różnica projekcji: {np.mean(projection_differences):.4f} ± {np.std(projection_differences):.4f}")
        print(f"  - Średnia korelacja Spearmana: {np.nanmean(spearman_correlations):.4f} ± {np.nanstd(spearman_correlations):.4f}")
        print(f"  - Średnia korelacja Pearsona: {np.nanmean(pearson_correlations):.4f} ± {np.nanstd(pearson_correlations):.4f}")
    
    # Podsumowanie wyników w tabeli
    print("\n=== PODSUMOWANIE ROZSZERZONEJ ANALIZY DVA ===\n")
    print(f"{'Domena':<25} | {'N par':<6} | {'Cos (μ±σ)':<15} | {'Proj.Diff (μ±σ)':<18} | {'Spearman (μ±σ)':<18} | {'Pearson (μ±σ)':<18}")
    print("-" * 120)
    
    output_lines = []
    output_lines.append("\n=== PODSUMOWANIE ROZSZERZONEJ ANALIZY DVA ===\n")
    output_lines.append(f"{'Domena':<25} | {'N par':<6} | {'Cos (μ±σ)':<15} | {'Proj.Diff (μ±σ)':<18} | {'Spearman (μ±σ)':<18} | {'Pearson (μ±σ)':<18}")
    output_lines.append("-" * 120)
    
    for domain, result in results.items():
        n_pairs = result['n_pairs']
        cos_mean = result['cos_similarity_stats']['mean']
        cos_std = result['cos_similarity_stats']['std']
        proj_mean = result['projection_diff_stats']['mean']
        proj_std = result['projection_diff_stats']['std']
        spear_mean = result['spearman_stats']['mean']
        spear_std = result['spearman_stats']['std']
        pears_mean = result['pearson_stats']['mean']
        pears_std = result['pearson_stats']['std']
        
        output_lines.append(f"{domain:<25} | {n_pairs:<6} | {cos_mean:.3f}±{cos_std:.3f}    | {proj_mean:.3f}±{proj_std:.3f}        | {spear_mean:.3f}±{spear_std:.3f}")
    
    with open(f"./results_politics/{name_of_the_simulation}_dva_metrics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    return results


def visualize_extended_dva_results(dva_results: Dict[str, Any], name_of_the_results: str) -> None:
    """
    Wizualizuje wyniki rozszerzonej analizy DVA za pomocą boxplotów.
    
    Args:
        dva_results: Wyniki z funkcji extended_dva_analysis
    """
    domains = list(dva_results.keys())
    if not domains:
        print("Brak danych do wizualizacji.")
        return
    
    # Przygotuj dane do boxplotów
    boxplot_data = []
    
    for domain in domains:
        result = dva_results[domain]
        
        # Dodaj dane dla podobieństwa kosinusowego
        for value in result['cos_similarities']:
            boxplot_data.append({
                'Domena': domain,
                'Metryka': 'Podobieństwo kosinusowe',
                'Wartość': value
            })
        
        # Dodaj dane dla różnicy projekcji
        for value in result['projection_differences']:
            boxplot_data.append({
                'Domena': domain,
                'Metryka': 'Różnica projekcji',
                'Wartość': value
            })
        
        # Dodaj dane dla korelacji Spearmana
        for value in result['spearman_correlations']:
            if not np.isnan(value):
                boxplot_data.append({
                    'Domena': domain,
                    'Metryka': 'Korelacja Spearmana',
                    'Wartość': value
                })
        
        # Dodaj dane dla korelacji Pearsona
        for value in result['pearson_correlations']:
            if not np.isnan(value):
                boxplot_data.append({
                    'Domena': domain,
                    'Metryka': 'Korelacja Pearsona',
                    'Wartość': value
                })
    
    df = pd.DataFrame(boxplot_data)
    
    # Utwórz wielopanelowy wykres boxplotów
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Rozszerzona analiza DVA - Rozkłady metryk dla wszystkich par', fontsize=16)
    
    # 1. Podobieństwo kosinusowe
    cos_data = df[df['Metryka'] == 'Podobieństwo kosinusowe']
    sns.boxplot(data=cos_data, x='Domena', y='Wartość', ax=axes[0,0])
    axes[0,0].set_title('Podobieństwo kosinusowe')
    axes[0,0].set_ylabel('Wartość (niższa = większa polaryzacja)')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 2. Różnica projekcji
    proj_data = df[df['Metryka'] == 'Różnica projekcji']
    sns.boxplot(data=proj_data, x='Domena', y='Wartość', ax=axes[0,1])
    axes[0,1].set_title('Różnica projekcji (DVA)')
    axes[0,1].set_ylabel('Wartość (wyższa = większa polaryzacja)')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 3. Korelacja Spearmana
    spear_data = df[df['Metryka'] == 'Korelacja Spearmana']
    sns.boxplot(data=spear_data, x='Domena', y='Wartość', ax=axes[1,0])
    axes[1,0].set_title('Korelacja Spearmana')
    axes[1,0].set_ylabel('Wartość korelacji')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 4. Korelacja Pearsona
    pears_data = df[df['Metryka'] == 'Korelacja Pearsona']
    sns.boxplot(data=pears_data, x='Domena', y='Wartość', ax=axes[1,1])
    axes[1,1].set_title('Korelacja Pearsona')
    axes[1,1].set_ylabel('Wartość korelacji')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('extended_dva_boxplots.png', dpi=300, bbox_inches='tight')
    print("Boxploty rozszerzonej analizy DVA zostały zapisane jako 'extended_dva_boxplots.png'.")
    
    # Dodatkowy wykres: Porównanie projekcji beneficial vs harmful
    fig, ax = plt.subplots(figsize=(14, 8))
    
    projection_data = []
    for domain in domains:
        result = dva_results[domain]
        
        # Dodaj projekcje beneficial
        for value in result['pos_projections']:
            projection_data.append({
                'Domena': domain,
                'Typ': 'Pozytywny',
                'Projekcja': value
            })
        
        # Dodaj projekcje harmful
        for value in result['neg_projections']:
            projection_data.append({
                'Domena': domain,
                'Typ': 'Negatywny',
                'Projekcja': value
            })
    
    proj_df = pd.DataFrame(projection_data)
    
    sns.boxplot(data=proj_df, x='Domena', y='Projekcja', hue='Typ', ax=ax)
    ax.set_xlabel("Kryterium polaryzacyjne", fontweight='bold', fontsize=18, labelpad=20)
    ax.set_ylabel('Wartość projekcji \n na znormalizowany wektor różnicy', fontweight='bold', fontsize=18, fontname='Arial', labelpad=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.7, alpha=0.7)

    ax.legend(
        title='Typ tekstu',
        fontsize=16,
        title_fontsize=18,
        loc='center left',          # punkt odniesienia legendy
        bbox_to_anchor=(1.02, 0.5)  # przesunięcie względem osi
    )    
    plt.tight_layout()
    plt.savefig(f'{name_of_the_results}_dva_projections_boxplot.png', dpi=400, bbox_inches='tight')
    print("Boxplot projekcji został zapisany jako 'extended_dva_projections_boxplot.png'.")
    
    # Wykres heatmapy średnich wartości
    fig, ax = plt.subplots(figsize=(12, 8))
    
    heatmap_data = []
    for domain in domains:
        result = dva_results[domain]
        heatmap_data.append({
            'Domena': domain,
            'Podobieństwo kosinusowe': result['cos_similarity_stats']['mean'],
            'Różnica projekcji': result['projection_diff_stats']['mean'],
            'Korelacja Spearmana': result['spearman_stats']['mean'],
            'Korelacja Pearsona': result['pearson_stats']['mean']
        })
    
    heatmap_df = pd.DataFrame(heatmap_data).set_index('Domena')
    
    sns.heatmap(heatmap_df, annot=True, cmap='coolwarm', center=0, fmt=".4f", ax=ax)
    ax.set_title('Heatmapa średnich wartości metryk DVA')
    
    plt.tight_layout()
    plt.savefig(f'{name_of_the_results}_dva_means_heatmap.png', dpi=300, bbox_inches='tight')
    print("Heatmapa średnich wartości została zapisana jako 'extended_dva_means_heatmap.png'.")
    
    # Wykres statystyk opisowych w tabeli
    create_summary_table(dva_results, name_of_the_results)


def create_summary_table(dva_results: Dict[str, Any], name_of_the_results: str) -> None:
    """
    Tworzy tabelę ze statystykami opisowymi dla wszystkich metryk.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Przygotuj dane do tabeli
    table_data = []
    
    for domain, result in dva_results.items():
        # Podobieństwo kosinusowe
        cos_stats = result['cos_similarity_stats']
        table_data.append([
            domain, 'Podobieństwo kosinusowe', result['n_pairs'],
            f"{cos_stats['mean']:.4f}", f"{cos_stats['std']:.4f}",
            f"{cos_stats['median']:.4f}", f"{cos_stats['min']:.4f}",
            f"{cos_stats['max']:.4f}"
        ])
        
        # Różnica projekcji
        proj_stats = result['projection_diff_stats']
        table_data.append([
            '', 'Różnica projekcji', '',
            f"{proj_stats['mean']:.4f}", f"{proj_stats['std']:.4f}",
            f"{proj_stats['median']:.4f}", f"{proj_stats['min']:.4f}",
            f"{proj_stats['max']:.4f}"
        ])
        
        # Korelacja Spearmana
        spear_stats = result['spearman_stats']
        table_data.append([
            '', 'Korelacja Spearmana', '',
            f"{spear_stats['mean']:.4f}", f"{spear_stats['std']:.4f}",
            f"{spear_stats['median']:.4f}", f"{spear_stats['min']:.4f}",
            f"{spear_stats['max']:.4f}"
        ])
        
        # Korelacja Pearsona
        pears_stats = result['pearson_stats']
        table_data.append([
            '', 'Korelacja Pearsona', '',
            f"{pears_stats['mean']:.4f}", f"{pears_stats['std']:.4f}",
            f"{pears_stats['median']:.4f}", f"{pears_stats['min']:.4f}",
            f"{pears_stats['max']:.4f}"
        ])
        
        # Dodaj pusty wiersz dla separacji
        table_data.append(['', '', '', '', '', '', '', ''])
    
    # Usuń ostatni pusty wiersz
    if table_data:
        table_data = table_data[:-1]
    
    columns = ['Domena', 'Metryka', 'N par', 'Średnia', 'Odch. std', 'Mediana', 'Min', 'Max']
    
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Stylizacja tabeli
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Kolorowanie wierszy naprzemiennie
    for i in range(1, len(table_data) + 1):
        if i % 8 < 4 and i % 8 != 0:  # Pierwsze 4 wiersze każdej domeny
            for j in range(len(columns)):
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Statystyki opisowe rozszerzonej analizy DVA', fontsize=14, weight='bold', pad=20)
    plt.savefig(f'{name_of_the_results}_properyty_metrics.png', dpi=400, bbox_inches='tight')
    print("Tabela statystyk opisowych została zapisana jako 'extended_dva_summary_table.png'.")
    plt.show()


# Przykład użycia:
# all_embeddings, results = perform_extended_dva_analysis(semantic_features, method='openai')
