import streamlit as st
import json
import os
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()


# Import plotly safely
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set page config first
st.set_page_config(page_title="TAM SAM SOM Calculator", page_icon="📊", layout="wide")

# Try to import optional libraries
try:
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
    # print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# Load comprehensive market data
@st.cache_data
def load_comprehensive_data():
    """Load industry and market data from processed files"""
    try:
        with open("industries_data.json", "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {
            "industries": [
                "Artificial Intelligence",
                "E-commerce",
                "EdTech",
                "FinTech",
                "HealthTech",
            ],
            "countries": ["United States", "China", "Europe", "India", "Global"],
            "regions": ["North America", "Asia", "Europe", "Middle East"],
            "market_data": {
                "Artificial Intelligence": {
                    "global_tam": 800.0,
                    "regional_multipliers": {
                        "United States": 0.35,
                        "China": 0.20,
                        "Europe": 0.18,
                        "India": 0.08,
                        "Global": 1.0,
                    },
                }
            },
        }


# Load the comprehensive data
comprehensive_data = load_comprehensive_data()
MARKET_DATA = comprehensive_data["market_data"]
AVAILABLE_INDUSTRIES = comprehensive_data["industries"]
AVAILABLE_COUNTRIES = comprehensive_data["countries"]

# Target audience penetration rates
AUDIENCE_PENETRATION = {
    "college students": 0.06,
    "young professionals": 0.08,
    "small businesses": 0.04,
    "enterprises": 0.02,
    "general consumers": 0.05,
    "seniors": 0.03,
    "teenagers": 0.07,
    "families": 0.05,
}


def get_market_estimates(
    industry: str, region: str
) -> Tuple[float, float, float, float]:
    """Get TAM, SAM, SOM and penetration rate using Gemini AI only."""

    import re
    import datetime

    # Safety bounds
    def sanitize(value: str, min_val: float, max_val: float, default: float) -> float:
        try:
            match = re.search(r"(\d+\.?\d*)", value)
            val = float(match.group(1)) if match else default
            return min(max(val, min_val), max_val)
        except:
            return default

    if not GEMINI_AVAILABLE:
        st.error(
            "Gemini API is not available. Please check your API key or environment."
        )
        return 100.0, 10.0, 1.0, 0.01  # dummy values

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        # model = genai.GenerativeModel("gemini-1.5-pro")

        # Step 1: Get TAM
        tam_prompt = f"""
        You are a market research analyst. What is the estimated global Total Addressable Market (TAM) in 2024 for the {industry} industry?

        Only respond with a single number, in billions USD. No text or explanation.
        Example: 245.3
        """
        tam_response = model.generate_content(tam_prompt)
        tam = sanitize(tam_response.text.strip(), 10.0, 10000.0, 500.0)

        # Step 2: Get Regional Share
        region_prompt = f"""
        # What percentage of the global {industry} market does {region} represent?
        # Respond with a number only, no % symbol. Example: 4.5
        You are a market analyst. Estimate what percentage of the global {industry} market in 2024 is attributed **only to {region}**.
        Respond with a number **only** (e.g., 0.25 or 5.0). No % symbol or explanation.
        Assume {region} is a country with {('high' if region in ['United States', 'India', 'China'] else 'low')} digital economy activity.
        """
        region_response = model.generate_content(region_prompt)
        regional_percent = sanitize(region_response.text.strip(), 0.1, 50.0, 5.0)
        regional_multiplier = regional_percent / 100.0

        # Step 3: Calculate SAM
        sam = tam * regional_multiplier

        # Step 4: Get Penetration Rate
        penetration_prompt = f"""
        What is a realistic market penetration rate in {region} for a new company entering the {industry} industry?

        Respond with a number only, no % symbol. Example: 2.5
        """
        penetration_response = model.generate_content(penetration_prompt)
        penetration_rate = (
            sanitize(penetration_response.text.strip(), 0.5, 20.0, 2.5) / 100.0
        )

        # Step 5: Calculate SOM
        som = sam * penetration_rate

        # Step 6: Log responses
        log_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "industry": industry,
            "region": region,
            "tam_raw": tam_response.text.strip(),
            "regional_raw": region_response.text.strip(),
            "penetration_raw": penetration_response.text.strip(),
            "tam_final": tam,
            "sam_final": sam,
            "som_final": som,
            "penetration_final": penetration_rate,
        }

        with open("gemini_logs.txt", "a") as f:
            f.write(json.dumps(log_data) + "\n")

        return tam, sam, som, penetration_rate

    except Exception as e:
        st.error(f"Gemini call failed: {e}")
        return 100.0, 10.0, 1.0, 0.01  # fallback safety values


# def get_market_estimates(
#     industry: str, region: str
# ) -> Tuple[float, float, float, float]:
#     """Get TAM, regional share, and penetration rate using Gemini AI, then calculate SAM and SOM"""

#     api_key = os.getenv("GEMINI_API_KEY", "")

#     if api_key and GEMINI_AVAILABLE:
#         try:
#             model = genai.GenerativeModel("gemini-1.5-flash")
#             model = genai.GenerativeModel("gemini-1.5-pro")

#             # Get TAM from Gemini
#             tam_prompt = f"""
#             You are a market research expert. Provide the Total Addressable Market (TAM) for the {industry} industry globally in 2024.

#             Respond with ONLY a number in billions USD (example: 245.7).
#             Consider current market conditions, growth trends, and industry maturity.
#             No currency symbols, text, or explanations - just the numerical value.
#             """

#             tam_response = model.generate_content(tam_prompt)
#             tam_text = tam_response.text.strip()

#             import re

#             tam_match = re.search(r"(\d+\.?\d*)", tam_text)
#             tam = float(tam_match.group(1)) if tam_match else 500.0

#             # Get regional market share from Gemini
#             regional_prompt = f"""
#             You are a market research expert. What percentage of the global {industry} market does {region} represent?

#             Respond with ONLY a percentage number (example: 25.3).
#             Consider economic size, market development, and regional factors.
#             No % symbol, text, or explanations - just the numerical value.
#             """

#             regional_response = model.generate_content(regional_prompt)
#             regional_text = regional_response.text.strip()

#             regional_match = re.search(r"(\d+\.?\d*)", regional_text)
#             regional_share_percent = (
#                 float(regional_match.group(1)) if regional_match else 5.0
#             )
#             regional_multiplier = regional_share_percent / 100

#             # Calculate SAM
#             sam = tam * regional_multiplier

#             # Get penetration rate from Gemini
#             penetration_prompt = f"""
#             You are a market research expert. What is a realistic market penetration rate for a new company in the {industry} industry in {region}?

#             Respond with ONLY a percentage number (example: 3.5).
#             Consider competition, market saturation, and typical adoption rates.
#             No % symbol, text, or explanations - just the numerical value.
#             """

#             penetration_response = model.generate_content(penetration_prompt)
#             penetration_text = penetration_response.text.strip()

#             penetration_match = re.search(r"(\d+\.?\d*)", penetration_text)
#             penetration_percent = (
#                 float(penetration_match.group(1)) if penetration_match else 2.0
#             )
#             penetration_rate = penetration_percent / 100

#             # Calculate SOM
#             som = sam * penetration_rate

#             return tam, sam, som, penetration_rate

#         except Exception as e:
#             st.warning(
#                 "Unable to connect to market research API. Using industry estimates."
#             )

#     # Fallback to industry estimates when Gemini unavailable
#     if industry in MARKET_DATA:
#         industry_data = MARKET_DATA[industry]
#         tam = industry_data["global_tam"]
#         regional_multiplier = industry_data["regional_multipliers"].get(region, 0.1)
#     else:
#         tam = 100.0
#         regional_multiplier = 0.1

#     # Calculate SAM
#     sam = tam * regional_multiplier

#     # Calculate SOM with default penetration rate
#     penetration_rate = 0.025  # Default 2.5%
#     som = sam * penetration_rate

#     return tam, sam, som, penetration_rate


def generate_fallback_assumptions(
    industry: str,
    region: str,
    tam: float,
    sam: float,
    som: float,
    penetration_rate: float,
) -> List[str]:
    """Generate professional assumptions"""
    regional_share = (sam / tam) * 100 if tam > 0 else 0
    return [
        f"TAM of ${tam:.1f}B pulled from market research databases for {industry} industry",
        f"SAM calculated as TAM × Regional Share ({regional_share:.1f}%) = ${sam:.1f}B for {region}",
        f"SOM calculated as SAM × Penetration Rate ({penetration_rate*100:.1f}%) = ${som:.1f}B",
    ]


def generate_assumptions_with_gemini(
    industry: str,
    region: str,
    tam: float,
    sam: float,
    som: float,
    penetration_rate: float,
) -> List[str]:
    """Generate assumptions using Gemini AI with fallback"""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or not GEMINI_AVAILABLE:
        return generate_fallback_assumptions(
            industry, region, tam, sam, som, penetration_rate
        )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        # model = genai.GenerativeModel("gemini-1.5-pro")

        regional_share = (sam / tam) * 100 if tam > 0 else 0
        prompt = f"""
        Generate exactly 3 concise assumptions for market sizing calculations:
        
        Industry: {industry}
        Region: {region}
        TAM: ${tam:.1f}B (from market research)
        Regional Share: {regional_share:.1f}%
        SAM: ${sam:.1f}B (calculated as TAM × Regional Share)
        Penetration Rate: {penetration_rate*100:.1f}%
        SOM: ${som:.1f}B (calculated as SAM × Penetration Rate)
        
        Format as:
        - [How TAM was determined from market research]
        - [How SAM was calculated using regional market share]  
        - [How SOM was calculated using penetration rate]
        """

        response = model.generate_content(prompt)

        if response.text:
            assumptions = []
            lines = response.text.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and (
                    line.startswith("-") or line.startswith("•") or line.startswith("*")
                ):
                    assumption = line[1:].strip()
                    if assumption:
                        assumptions.append(assumption)

            return (
                assumptions[:3]
                if assumptions
                else generate_fallback_assumptions(
                    industry, region, tam, sam, som, penetration_rate
                )
            )
        else:
            return generate_fallback_assumptions(
                industry, region, tam, sam, som, penetration_rate
            )

    except Exception:
        return generate_fallback_assumptions(
            industry, region, tam, sam, som, penetration_rate
        )


def create_market_funnel_chart(tam: float, sam: float, som: float, industry: str):
    """Create a funnel chart showing TAM, SAM, SOM progression"""
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure(
        go.Funnel(
            y=[
                "TAM (Total Addressable Market)",
                "SAM (Serviceable Addressable Market)",
                "SOM (Serviceable Obtainable Market)",
            ],
            x=[tam, sam, som],
            textinfo="value+percent initial",
            textfont=dict(size=14),
            marker=dict(color=["#FF6B6B", "#4ECDC4", "#45B7D1"]),
            connector=dict(line=dict(color="lightgray", dash="dot", width=3)),
        )
    )

    fig.update_layout(
        title=f"Market Opportunity Funnel - {industry}",
        font=dict(size=12),
        height=500,
        margin=dict(l=20, r=20, t=80, b=20),
    )

    return fig


def create_size_comparison_chart(tam: float, sam: float, som: float):
    """Create a bar chart comparing TAM, SAM, SOM values"""
    if not PLOTLY_AVAILABLE:
        return None

    categories = ["TAM", "SAM", "SOM"]
    values = [tam, sam, som]  # All values in billions
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=[f"${v:.1f}B" for v in values],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Market Size Comparison",
        yaxis_title="Market Size (Billions USD)",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )

    return fig


def create_regional_breakdown_chart(industry: str, region: str):
    """Create a pie chart showing regional market distribution"""
    if not PLOTLY_AVAILABLE:
        return None

    if industry in MARKET_DATA:
        regional_data = MARKET_DATA[industry]["regional_multipliers"]
        regions = list(regional_data.keys())
        percentages = [v * 100 for v in regional_data.values()]

        colors = ["#FF6B6B" if r == region else "#E8E8E8" for r in regions]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=regions,
                    values=percentages,
                    marker_colors=colors,
                    textinfo="label+percent",
                    textfont_size=12,
                )
            ]
        )

        fig.update_layout(
            title=f"Regional Market Distribution - {industry}",
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False,
        )

        return fig

    # Fallback chart for unknown industries
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["United States", "China", "Europe", "India", "Others"],
                values=[35, 20, 18, 8, 19],
                marker_colors=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
            )
        ]
    )

    fig.update_layout(
        title=f"Estimated Regional Distribution - {industry}",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )

    return fig


def format_currency(amount: float, unit: str = "B") -> str:
    """Format currency amount with appropriate unit"""
    if unit == "B":
        return f"${amount:.1f}B"
    elif unit == "M":
        return f"${amount:.0f}M"
    else:
        return f"${amount:.2f}{unit}"


def main():
    st.title("📊 TAM, SAM, SOM Market Opportunity Calculator")
    st.markdown(
        "Calculate Total Addressable Market (TAM), Serviceable Addressable Market (SAM), and Serviceable Obtainable Market (SOM) using comprehensive industry data."
    )

    # Create two columns for better layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Input Parameters")

        # Industry selection
        st.markdown("**Industry Selection**")
        industry = st.selectbox(
            "Select Industry",
            options=AVAILABLE_INDUSTRIES + ["Other"],
            index=0,
            help="Select from comprehensive industry database",
        )

        if industry == "Other":
            industry = st.text_input(
                "Custom Industry", placeholder="e.g., Autonomous Vehicles"
            )

        # Region selection
        st.markdown("**Region Selection**")
        region = st.selectbox(
            "Select Region",
            options=AVAILABLE_COUNTRIES + ["Other"],
            index=0,
            help="Select from comprehensive geographic database",
        )

        if region == "Other":
            region = st.text_input("Custom Region", placeholder="e.g., Southeast Asia")

        # Calculate button
        if st.button("🔍 Calculate Market Opportunity", type="primary"):
            if industry and region:
                # Store results in session state
                st.session_state.calculation_done = True
                st.session_state.industry = industry
                st.session_state.region = region

                # Calculate market estimates
                tam, sam, som, actual_penetration = get_market_estimates(
                    industry, region
                )
                regional_share = (sam / tam) * 100 if tam > 0 else 0
                st.session_state.regional_share = regional_share

                st.session_state.tam = tam
                st.session_state.sam = sam
                st.session_state.som = som
                st.session_state.penetration_rate = actual_penetration

                # Generate assumptions
                assumptions = generate_assumptions_with_gemini(
                    industry, region, tam, sam, som, actual_penetration
                )
                st.session_state.assumptions = assumptions

                st.rerun()
            else:
                st.error("Please fill in all required fields.")

    with col2:
        st.subheader("📈 Market Opportunity Results")

        # Initialize session state variables if they don't exist
        if "calculation_done" not in st.session_state:
            st.session_state.calculation_done = False

        if st.session_state.calculation_done and all(
            hasattr(st.session_state, attr)
            for attr in [
                "tam",
                "sam",
                "som",
                "assumptions",
                "industry",
                "region",
                "penetration_rate",
            ]
        ):
            # Display metrics
            # metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                st.metric(
                    label="TAM (Total Addressable Market)",
                    value=format_currency(st.session_state.tam, "B"),
                    help="Total global market size for this industry",
                )

            with metric_col2:
                st.metric(
                    label="SAM (Serviceable Addressable Market)",
                    value=format_currency(st.session_state.sam, "B"),
                    help=f"Market size in {st.session_state.region}",
                )

            with metric_col3:
                st.metric(
                    label="SOM (Serviceable Obtainable Market)",
                    value=format_currency(st.session_state.som * 1000, "M"),
                    help="Obtainable market based on calculated penetration rate",
                )
            with metric_col4:
                st.metric(
                    label="Regional Market Share",
                    value=f"{st.session_state.regional_share:.2f}%",
                    help=f"{st.session_state.region}'s share of global {st.session_state.industry} TAM",
                )

            # Market Visualization with Plotly Charts
            st.subheader("📊 Market Visualization")

            if PLOTLY_AVAILABLE:
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(
                    ["Market Funnel", "Size Comparison", "Regional Breakdown"]
                )

                with tab1:
                    funnel_chart = create_market_funnel_chart(
                        st.session_state.tam,
                        st.session_state.sam,
                        st.session_state.som,
                        st.session_state.industry,
                    )
                    if funnel_chart:
                        st.plotly_chart(funnel_chart, use_container_width=True)
                    else:
                        st.error("Unable to generate funnel chart")

                with tab2:
                    comparison_chart = create_size_comparison_chart(
                        st.session_state.tam, st.session_state.sam, st.session_state.som
                    )
                    if comparison_chart:
                        st.plotly_chart(comparison_chart, use_container_width=True)
                    else:
                        st.error("Unable to generate comparison chart")

                with tab3:
                    regional_chart = create_regional_breakdown_chart(
                        st.session_state.industry, st.session_state.region
                    )
                    if regional_chart:
                        st.plotly_chart(regional_chart, use_container_width=True)
                    else:
                        st.info(
                            "Regional breakdown chart not available for this industry"
                        )
            else:
                st.warning(
                    "Plotly charts unavailable due to system dependencies. Showing text visualization:"
                )

                # Fallback text visualization
                tam_width = 100
                sam_width = int((st.session_state.sam / st.session_state.tam) * 100)
                som_width = int(
                    (st.session_state.som * 1000 / st.session_state.tam) * 100
                )

                st.markdown("**Market Funnel Visualization:**")
                st.markdown(f"TAM: {'█' * tam_width} ${st.session_state.tam:.1f}B")
                st.markdown(
                    f"SAM: {'█' * sam_width} ${st.session_state.sam:.1f}B ({sam_width}% of TAM)"
                )
                st.markdown(
                    f"SOM: {'█' * som_width} ${st.session_state.som * 1000:.0f}M ({som_width}% of TAM)"
                )

            # Display assumptions
            st.subheader("💡 Market Assumptions")
            for i, assumption in enumerate(st.session_state.assumptions, 1):
                st.markdown(f"• {assumption}")

            # Market Breakdown
            st.subheader("📋 Detailed Breakdown")

            st.markdown("**TAM (Total Addressable Market)**")
            st.markdown(f"- Value: {format_currency(st.session_state.tam, 'B')}")
            st.markdown(f"- Description: Global market size")

            st.markdown("**SAM (Serviceable Addressable Market)**")
            st.markdown(f"- Value: {format_currency(st.session_state.sam, 'B')}")
            st.markdown(f"- Description: Market size in {st.session_state.region}")

            st.markdown("**SOM (Serviceable Obtainable Market)**")
            st.markdown(f"- Value: {format_currency(st.session_state.som * 1000, 'M')}")
            st.markdown(f"- Description: Obtainable market based on penetration rate")

            st.markdown("**Penetration Rate**")
            st.markdown(f"- Value: {st.session_state.penetration_rate*100:.1f}%")
            st.markdown(f"- Description: Market penetration assumption")

            # JSON Output
            st.subheader("📋 JSON Output")
            json_output = {
                "TAM": format_currency(st.session_state.tam, "B"),
                "SAM": format_currency(st.session_state.sam, "B"),
                "SOM": format_currency(st.session_state.som * 1000, "M"),
                "assumptions": st.session_state.assumptions,
                "region": st.session_state.region,
                "industry": st.session_state.industry,
                "penetration_rate": f"{st.session_state.penetration_rate*100:.1f}%",
            }

            st.json(json_output)

            # Download button for JSON
            json_string = json.dumps(json_output, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_string,
                file_name=f"market_analysis_{st.session_state.industry.lower().replace(' ', '_')}.json",
                mime="application/json",
            )
        else:
            st.info(
                "👈 Fill in the parameters on the left and click 'Calculate Market Opportunity' to see results."
            )

            # Show available data info
            st.subheader("📊 Available Data")
            st.markdown(
                f"**Industries:** {len(AVAILABLE_INDUSTRIES)} industries available"
            )
            st.markdown(
                f"**Regions:** {len(AVAILABLE_COUNTRIES)} countries/regions available"
            )

            # Sample industries
            st.markdown("**Sample Industries:**")
            sample_industries = AVAILABLE_INDUSTRIES[:10]
            for industry in sample_industries:
                st.markdown(f"• {industry}")

    # Footer with methodology
    st.markdown("---")
    st.subheader("📚 Methodology")

    method_col1, method_col2, method_col3 = st.columns(3)

    with method_col1:
        st.markdown(
            """
        **TAM (Total Addressable Market)**
        - Global market size for the entire industry
        - Based on comprehensive industry research
        - Represents maximum theoretical revenue
        """
        )

    with method_col2:
        st.markdown(
            """
        **SAM (Serviceable Addressable Market)**
        - Regional subset of TAM
        - Accounts for geographic constraints
        - TAM × Regional Market Share
        """
        )

    with method_col3:
        st.markdown(
            """
        **SOM (Serviceable Obtainable Market)**
        - Realistic market capture potential
        - Considers target audience penetration
        - SAM × Penetration Rate
        """
        )


if __name__ == "__main__":
    main()
