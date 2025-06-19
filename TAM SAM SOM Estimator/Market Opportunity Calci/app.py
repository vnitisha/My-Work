import streamlit as st
import json
from dotenv import load_dotenv
import os
from typing import Dict, Any, List, Tuple, Optional

load_dotenv()

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

st.set_page_config(page_title="TAM SAM SOM Calculator", page_icon="📊", layout="wide")

try:
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


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

comprehensive_data = load_comprehensive_data()
MARKET_DATA = comprehensive_data["market_data"]
AVAILABLE_INDUSTRIES = comprehensive_data["industries"]
AVAILABLE_COUNTRIES = comprehensive_data["countries"]

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
    industry: str,
    region: str,
    target_audience: str,
    penetration_estimate: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """Calculate TAM, SAM, and SOM based on comprehensive data"""

    if industry in MARKET_DATA:
        industry_data = MARKET_DATA[industry]
        tam = industry_data["global_tam"]
        regional_multiplier = industry_data["regional_multipliers"].get(region, 0.1)
    else:
        tam = 100.0
        regional_multiplier = 0.1

    # Calculate SAM
    sam = tam * regional_multiplier

    # Calculate SOM
    if penetration_estimate is not None:
        penetration_rate = penetration_estimate
    else:
        penetration_rate = AUDIENCE_PENETRATION.get(target_audience.lower(), 0.05)

    som = sam * penetration_rate

    return tam, sam, som, penetration_rate


def generate_fallback_assumptions(
    industry: str,
    region: str,
    target_audience: str,
    tam: float,
    sam: float,
    som: float,
    penetration_rate: float,
) -> List[str]:
    """Generate professional assumptions"""
    return [
        f"TAM of ${tam:.1f}B derived from industry market research and analyst reports",
        f"SAM calculation reflects {region} market share based on regional economic indicators",
        f"SOM estimates {penetration_rate*100:.1f}% market penetration for {target_audience} segment",
    ]


def generate_assumptions_with_gemini(
    industry: str,
    region: str,
    target_audience: str,
    tam: float,
    sam: float,
    som: float,
    penetration_rate: float,
) -> List[str]:
    """Generate assumptions using Gemini AI with fallback"""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or not GEMINI_AVAILABLE:
        return generate_fallback_assumptions(
            industry, region, target_audience, tam, sam, som, penetration_rate
        )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        Generate exactly 3 concise assumptions for market sizing:
        
        Industry: {industry}
        Region: {region}
        Target: {target_audience}
        TAM: ${tam:.1f}B | SAM: ${sam:.1f}B | SOM: ${som:.1f}M
        Penetration: {penetration_rate*100:.1f}%
        
        Format as:
        - [TAM assumption in <100 chars]
        - [SAM assumption in <100 chars]  
        - [SOM assumption in <100 chars]
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
                    industry, region, target_audience, tam, sam, som, penetration_rate
                )
            )
        else:
            return generate_fallback_assumptions(
                industry, region, target_audience, tam, sam, som, penetration_rate
            )

    except Exception:
        return generate_fallback_assumptions(
            industry, region, target_audience, tam, sam, som, penetration_rate
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
            x=[tam, sam, som * 1000],  # Convert SOM to millions for visualization
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
    values = [tam, sam, som * 1000]  # Convert SOM to millions
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

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Input Parameters")

        st.markdown("**Industry Selection**")
        display_industries = AVAILABLE_INDUSTRIES[:50]
        display_industries.append("Other")

        industry = st.selectbox(
            "Select Industry",
            options=display_industries,
            index=0,
            help="Select from comprehensive industry database",
        )

        if industry == "Other":
            industry = st.text_input(
                "Custom Industry", placeholder="e.g., Autonomous Vehicles"
            )


        st.markdown("**Region Selection**")
        display_regions = AVAILABLE_COUNTRIES[:30]
        display_regions.append("Other")

        region = st.selectbox(
            "Select Region",
            options=display_regions,
            index=0,
            help="Select from comprehensive geographic database",
        )

        if region == "Other":
            region = st.text_input("Custom Region", placeholder="e.g., Southeast Asia")

        audience_options = list(AUDIENCE_PENETRATION.keys()) + ["Other"]
        target_audience = st.selectbox(
            "Target Audience",
            options=audience_options,
            index=0,
            help="Select your primary target audience segment",
        )

        if target_audience == "Other":
            target_audience = st.text_input(
                "Custom Target Audience", placeholder="e.g., rural farmers"
            )

        use_custom_penetration = st.checkbox("Use custom penetration estimate")
        penetration_estimate = None

        if use_custom_penetration:
            penetration_estimate = (
                st.slider(
                    "Penetration Estimate (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    help="Percentage of SAM you expect to capture",
                )
                / 100.0
            )

        # Calculate button
        if st.button("🔍 Calculate Market Opportunity", type="primary"):
            if industry and region and target_audience:
                # Store results in session state
                st.session_state.calculation_done = True
                st.session_state.industry = industry
                st.session_state.region = region
                st.session_state.target_audience = target_audience
                st.session_state.penetration_estimate = penetration_estimate

                # Calculate market estimates
                tam, sam, som, actual_penetration = get_market_estimates(
                    industry, region, target_audience, penetration_estimate
                )
                st.session_state.tam = tam
                st.session_state.sam = sam
                st.session_state.som = som
                st.session_state.penetration_rate = actual_penetration

                # Generate assumptions
                assumptions = generate_assumptions_with_gemini(
                    industry, region, target_audience, tam, sam, som, actual_penetration
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
                "target_audience",
                "penetration_rate",
            ]
        ):
            # Display metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)

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
                    help=f"Obtainable market for {st.session_state.target_audience}",
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
            st.markdown(
                f"- Description: Obtainable market for {st.session_state.target_audience}"
            )

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
                "audience": st.session_state.target_audience,
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
