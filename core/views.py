from django.shortcuts import render
import joblib
import numpy as np
import os
from django.conf import settings

# ---------------- SAFE INTEGER FETCH ----------------
def intval(request, key, default=0):
    """
    Safely fetch an integer from POST data.
    Handles missing keys and empty strings.
    """
    value = request.POST.get(key, None)
    if value in (None, ""):
        return default
    return int(value)

# ---------------- LOAD MODELS ONCE ----------------
BASE_DIR = settings.BASE_DIR

MODELS = {
    "famenet": joblib.load(os.path.join(BASE_DIR, "xgboost_model.joblib")),
    "random_forest": joblib.load(os.path.join(BASE_DIR, "random_forest_model.joblib")),
    "lightgbm": joblib.load(os.path.join(BASE_DIR, "lightgbm_model.joblib")),
    "decision_tree": joblib.load(os.path.join(BASE_DIR, "decision_tree_model.joblib")),
}

# ---------------- VIEW ----------------
def index(request):
    context = {}

    if request.method == "POST":

        # ---------- MODEL ----------
        model_key = request.POST.get("model", "famenet")
        model = MODELS.get(model_key)

        # ---------- BASIC / NUMERIC ----------
        age = intval(request, "age")
        years_at_company = intval(request, "years_at_company")
        monthly_income = intval(request, "monthly_income")
        work_life = intval(request, "work_life_balance")
        job_satisfaction = intval(request, "job_satisfaction")
        performance = intval(request, "performance_rating")
        promotions = intval(request, "num_promotions")
        overtime = intval(request, "overtime")
        distance = intval(request, "distance_from_home")
        education = intval(request, "education_level")
        dependents = intval(request, "num_dependents")
        job_level = intval(request, "job_level")
        company_tenure = intval(request, "company_tenure")
        remote = intval(request, "remote_work")
        leadership = intval(request, "leadership_opportunities")
        innovation = intval(request, "innovation_opportunities")
        reputation = intval(request, "company_reputation")
        recognition = intval(request, "employee_recognition")
        gender_male = intval(request, "gender_male")

        # ---------- CATEGORICAL ----------
        job_role = request.POST.get("job_role", "other")
        marital = request.POST.get("marital_status", "other")
        company_size = request.POST.get("company_size", "large")

        # ---------- ONE-HOT MAPS ----------
        job_role_map = {
            "finance": [1, 0, 0, 0],
            "healthcare": [0, 1, 0, 0],
            "media": [0, 0, 1, 0],
            "technology": [0, 0, 0, 1],
            "other": [0, 0, 0, 0],
        }

        marital_map = {
            "married": [1, 0],
            "single": [0, 1],
            "other": [0, 0],
        }

        company_size_map = {
            "medium": [1, 0],
            "small": [0, 1],
            "large": [0, 0],  # baseline
        }

        # ---------- FINAL FEATURE VECTOR (27) ----------
        X_employee = np.array([[
            age,
            years_at_company,
            monthly_income,
            work_life,
            job_satisfaction,
            performance,
            promotions,
            overtime,
            distance,
            education,
            dependents,
            job_level,
            company_tenure,
            remote,
            leadership,
            innovation,
            reputation,
            recognition,
            gender_male,
            *job_role_map[job_role],
            *marital_map[marital],
            *company_size_map[company_size]
        ]])

        # ---------- PREDICTION ----------
        prediction = int(model.predict(X_employee)[0])
        probability = float(model.predict_proba(X_employee)[0][1])

        # ---------- CONTEXT ----------
        context = {
            "predicted": prediction,          # 0 or 1
            "probability": round(probability * 100, 2),
            "model_name": model_key.replace("_", " ").title(),
        }

    return render(request, "core/display.html", context)