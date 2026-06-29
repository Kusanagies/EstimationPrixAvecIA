"""
API REST D'ESTIMATION IMMOBILIERE (FastAPI)
============================================
Expose la fonction estimer() de estimer.py via une API REST propre.

Installation :  pip install fastapi uvicorn
Lancer avec   :  uvicorn serveur:app --reload --port 5000
Puis ouvrir   :
  - http://127.0.0.1:5000        (le front index.html)
  - http://127.0.0.1:5000/docs   (documentation interactive auto-generee)
"""

from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Logique metier (inchangee). Adapte le nom si besoin :
#   from estimer_catboost import estimer
from estimer import estimer

DOSSIER = Path(__file__).resolve().parent

app = FastAPI(
    title="API Estimation Immobiliere",
    description="Estime un bien (maison ou appartement) a partir de son adresse.",
    version="1.0.0",
)


# ==========================================
# MODELES DE DONNEES (validation automatique)
# ==========================================
class DemandeEstimation(BaseModel):
    adresse: str = Field(..., min_length=5, description="Adresse postale du bien")
    type_bien: str = Field(..., pattern="^(maisons|appartements)$",
                           description="'maisons' ou 'appartements'")
    surface: float = Field(..., gt=9, le=300, description="Surface habitable en m2")
    nb_pieces: int = Field(..., gt=0, description="Nombre de pieces principales")
    surface_terrain: float = Field(default=0, ge=0, description="Surface du terrain en m2 (0 si aucun)")


class ReponseEstimation(BaseModel):
    adresse: str
    type_retenu: str
    prix_m2_bas: int
    prix_m2_milieu: int
    prix_m2_haut: int
    prix_total_bas: int
    prix_total_milieu: int
    prix_total_haut: int


# ==========================================
# ROUTES
# ==========================================
@app.get("/")
def index():
    """Sert la page HTML du front."""
    return FileResponse(DOSSIER / "index.html")


@app.get("/api/sante")
def sante():
    """Health check : verifie que le service tourne."""
    return {"status": "ok"}


@app.get("/api/adresses")
def autocompletion_adresses(q: str):
    """Relais vers l'API BAN pour l'auto-completion d'adresse."""
    if len(q) < 4:
        return {"suggestions": []}
    try:
        r = requests.get(
            "https://api-adresse.data.gouv.fr/search/",
            params={"q": q, "limit": 5},
            timeout=10,
        )
        data = r.json()
    except Exception:
        raise HTTPException(status_code=503, detail="Service de geocodage indisponible")

    suggestions = [f["properties"]["label"] for f in data.get("features", [])]
    return {"suggestions": suggestions}


@app.post("/api/estimation", response_model=ReponseEstimation)
def creer_estimation(demande: DemandeEstimation):
    """
    Estime un bien. Renvoie les trois prix (bas / milieu / haut),
    au m2 et en total.
    """
    resultat = estimer(
        adresse=demande.adresse,
        surface=demande.surface,
        type_bien=demande.type_bien,
        nb_pieces=demande.nb_pieces,
        surface_terrain=demande.surface_terrain,
    )

    # Traduction des cas metier en codes HTTP parlants
    if "erreur" in resultat:
        raise HTTPException(status_code=400, detail=resultat["erreur"])
    if "suggestions" in resultat:
        raise HTTPException(
            status_code=422,
            detail="Adresse trop imprecise, merci de preciser.",
        )

    return resultat