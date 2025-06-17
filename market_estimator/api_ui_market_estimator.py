import streamlit as st
import json

# ---------------- Logic Layer ----------------
class MarketEstimator:
    def __init__(self):
        self.industry_tam_map = {
            "Mental Health Tech": 12_000_000_000  # $12B
        }
        self.region_factor = {
            "India": 0.21,  # 21% of TAM is SAM in India
        }
        self.audience_factor = {
            "college students": 0.06,  # 6% of SAM assumed
            "working professionals": 0.12,
            "general population": 0.20
        }

    def estimate(self, industry, region, audience, penetration=0.01):
        tam = self.industry_tam_map.get(industry, 0)
        sam = tam * self.region_factor.get(region, 0)
        som = sam * self.audience_factor.get(audience, 0) * (penetration / 0.01)

        response = {
            "TAM": f"${tam/1e9:.0f}B",
            "SAM": f"${sam/1e9:.2f}B",
            "SOM": f"${som/1e6:.0f}M",
            "assumptions": [
                "TAM derived from global market data (Statista 2023)",
                f"SAM = ~{self.region_factor[region]*100:.0f}% of TAM focused on {region}",
                f"SOM = ~{self.audience_factor[audience]*100:.0f}% of SAM assuming {penetration*100:.1f}% penetration of target"
            ],
            "region": region,
            "audience": audience
        }
        return response

# ---------------- Streamlit Interface ----------------
st.set_page_config(page_title="API-style Market Estimator", layout="centered")
st.title("🧮 Market Opportunity Estimator (API UI Style)")
st.markdown("Fill in inputs and get a structured API-style market estimate.")

# --- Input Section ---
industry = st.text_input("Industry", value="Mental Health Tech")
region = st.selectbox("Region", ["India"])
audience = st.selectbox("Target Audience", ["college students", "working professionals", "general population"])
penetration = st.slider("Penetration Estimate (%)", 0.1, 10.0, 1.0, step=0.1)

# --- Run Estimation ---
estimator = MarketEstimator()
response = estimator.estimate(industry, region, audience, penetration / 100)

# --- Output JSON Format ---
st.markdown("### 📤 API Response")
st.code(json.dumps(response, indent=2), language="json")
