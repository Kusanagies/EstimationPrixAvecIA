import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# ==========================================
# 1. CONNEXION MYSQL
# ==========================================
moteur = create_engine("mysql+pymysql://root:1618@localhost:3306/EstimationIA")

print("📊 Analyse spatiale des DPE par département...")

# ==========================================
# 2. REQUÊTE SQL (Corrigée)
# ==========================================
requete_analyse = """
SELECT 
    LEFT(v.code_commune, 2) AS departement,
    AVG(v.valeur_fonciere / v.surface_reelle_bati) AS prix_m2_moyen,
    (SUM(CASE WHEN d.etiquette_dpe IN ('A', 'B', 'C') THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_DPE_Vert,
    (SUM(CASE WHEN d.etiquette_dpe = 'D' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_DPE_Jaune,
    (SUM(CASE WHEN d.etiquette_dpe IN ('E', 'F', 'G') THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_DPE_Rouge
FROM valeurs_foncieres v
JOIN dpe_logements_france d ON v.code_commune = d.code_insee_ban
WHERE v.latitude IS NOT NULL 
  AND v.surface_reelle_bati > 9
  AND LEFT(v.code_commune, 2) IN ('75', '77', '78', '91', '92', '93', '94', '95')
GROUP BY LEFT(v.code_commune, 2);
"""

# ==========================================
# 3. TRAITEMENT ET AFFICHAGE
# ==========================================
df_analyse = pd.read_sql(requete_analyse, con=moteur)

# Tri par prix décroissant pour mettre en évidence le contraste
df_analyse = df_analyse.sort_values(by='prix_m2_moyen', ascending=False)
df_analyse = df_analyse.set_index('departement')

print("✅ Données récupérées. Génération de la Heatmap...")

# Génération de la carte de chaleur
plt.figure(figsize=(10, 6))

# On affiche le Prix au m², la part de passoires (Rouge) et la part d'éco-performants (Vert)
sns.heatmap(df_analyse[['prix_m2_moyen', 'pct_DPE_Vert', 'pct_DPE_Rouge']], 
            annot=True, 
            fmt=".1f", 
            cmap="YlOrRd", # Échelle de couleurs allant du jaune au rouge
            linewidths=1)

plt.title("Confirmation du Paradoxe : Prix au m² vs Performance Énergétique en IDF", pad=20)
plt.ylabel("Département", weight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()