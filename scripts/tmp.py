import matplotlib
matplotlib.use('Agg') # Forza backend non interattivo per salvataggio file
import matplotlib.pyplot as plt
import numpy as np

# --- 1. PREPARAZIONE DATI ---
models = ['Hybrid', 'Baseline', 'Full Crop']

# Dati Accuratezza (ACC)
acc_orig = [0.7067, 0.6316, 0.3722]
acc_crop = [0.7257, 0.6302, 0.3862]

# Dati F1-Score (F1)
f1_orig = [0.6938, 0.5790, 0.2959]
f1_crop = [0.6957, 0.5628, 0.2927]

# Dati Velocità (MS/IMG)
ms_orig = [74.65, 76.43, 73.85]
ms_crop = [39.32, 39.76, 36.86]

# --- 2. CONFIGURAZIONE GRAFICA ---
x = np.arange(len(models))
width = 0.35  # Larghezza delle barre

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
plt.rcParams.update({'font.size': 10})

def plot_bars(ax, orig_data, crop_data, title, ylabel, ylim=None):
    rects1 = ax.bar(x - width/2, orig_data, width, label='Original', color='#3498db', edgecolor='white', linewidth=0.7)
    rects2 = ax.bar(x + width/2, crop_data, width, label='Cropped', color='#2ecc71', edgecolor='white', linewidth=0.7)
    
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    if ylim: ax.set_ylim(ylim)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Etichette sui valori
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

# --- 3. GENERAZIONE DEI SOTTOGRAFICI ---
# Accuratezza
plot_bars(ax1, acc_orig, acc_crop, 'Confronto Accuratezza (ACC)', 'Score (0-1)', [0, 0.9])

# F1 Score
plot_bars(ax2, f1_orig, f1_crop, 'Confronto F1-Score', 'Score (0-1)', [0, 0.9])

# Latenza (MS/IMG)
plot_bars(ax3, ms_orig, ms_crop, 'Velocità di Inferenza', 'Millisecondi (ms)')

plt.tight_layout()

# --- 4. SALVATAGGIO ---
output_name = 'confronto_performance_totale.png'
plt.savefig(output_name, dpi=300)

print(f"Analisi completata!")
print(f"Grafico salvato come: {output_name}")

# --- 5. BREVE ANALISI STAMPATA ---
print("\n--- INSIGHT RAPIDI ---")
print(f"1. Velocità: Il cropping ha dimezzato i tempi di inferenza (~75ms -> ~39ms).")
print(f"2. Hybrid: Migliora l'accuratezza col crop (+1.9%) mantenendo un F1 solido.")
print(f"3. Baseline: Col crop è più veloce ma perde leggermente in precisione/F1.")