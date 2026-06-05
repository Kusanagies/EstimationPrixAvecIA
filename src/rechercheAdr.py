import pandas as pd
from sqlalchemy import create_engine

def chercher_dpe_100_local(code_postal, numero_rue, nom_rue):
    """
    Recherche un DPE en croisant la table BAN et la table DPE directement en local.
    """
    print(f" Recherche locale pour : {numero_rue} {nom_rue}, {code_postal}...")
    
    # ==========================================
    # 1. CONNEXION MYSQL
    # ==========================================
    USER = 'root'
    PASSWORD = '1618'
    HOTE = 'localhost'
    PORT = '3306'
    DB = 'EstimationIA'
    
    chaine_connexion = f"mysql+pymysql://{USER}:{PASSWORD}@{HOTE}:{PORT}/{DB}"
    moteur = create_engine(chaine_connexion)
    
    # ==========================================
    # 2. LA REQUÊTE SQL OPTIMISÉE (La magie du JOIN)
    # ==========================================
    # On remplace les espaces par des '%' pour que la recherche soit souple
    # Exemple : "rue belfort" deviendra "%rue%belfort%"
    mots_rue = nom_rue.split()
    recherche_nom_rue = "%" + "%".join(mots_rue) + "%"
    
    # La requête fait le travail des deux tables en même temps !
    requete_sql = f"""
    SELECT 
        a.numero, 
        a.nom_voie, 
        a.nom_commune, 
        d.etiquette_dpe, 
        d.etiquette_ges, 
        d.surface_habitable_logement,
        d.annee_construction
    FROM adresses_ban a
    -- On fait la jointure grâce au Code INSEE
    JOIN dpe_logements_france d ON a.code_insee = d.code_insee_ban
    WHERE a.code_postal = '{code_postal}'
      AND a.nom_voie LIKE '{recherche_nom_rue}'
      AND a.numero = '{numero_rue}'
    LIMIT 5;
    """
    
    try:
        # On exécute la requête et on récupère un tableau Pandas
        resultats = pd.read_sql(requete_sql, con=moteur)
        
        if len(resultats) > 0:
            print("\n BINGO ! Voici le(s) logement(s) trouvé(s) 100% en local :")
            # Affichage propre du tableau
            print(resultats.to_string(index=False))
        else:
            print("\nℹ L'adresse n'existe pas dans la base locale, ou aucun DPE n'y est associé.")
            print("Astuce : Vérifiez l'orthographe exacte de la rue.")
            
    except Exception as e:
        print(f" Erreur lors de la requête locale : {e}")

# ==========================================
# 3. TEST EN DIRECT
# ==========================================
# Testez avec une adresse dont vous êtes sûr qu'elle possède un DPE !
chercher_dpe_100_local(code_postal="11000", numero_rue="25", nom_rue="Rue de Belfort")