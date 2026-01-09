"""
Aadhaar Sentinel Interactive Dashboard

Streamlit-based dashboard for monitoring district ALRI scores with:
- District heatmap colored by ALRI score
- Time-series charts for selected districts
- Alerts panel for threshold-crossing districts
- Filtering by state, district, and time period

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import os

# Import project modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.storage.serializer import ALRIRecord, ALRIStorage
from src.explainability.reason_codes import ReasonCode, Severity


class AadhaarSentinelDashboard:
    """
    Interactive Streamlit dashboard for Aadhaar Sentinel ALRI monitoring.
    
    Provides visualization of district-level ALRI scores with:
    - Heatmap view of all districts
    - Detailed time-series for selected districts
    - Alert panel for high-risk districts
    - Filtering capabilities
    
    Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
    """
    
    # ALRI score thresholds for alerts
    ALERT_THRESHOLDS = {
        'critical': 75.0,
        'high': 50.0,
        'medium': 25.0
    }
    
    # Color scale for ALRI scores
    ALRI_COLOR_SCALE = [
        [0.0, '#2ecc71'],    # Green - Low risk
        [0.25, '#f1c40f'],   # Yellow - Medium risk
        [0.50, '#e67e22'],   # Orange - High risk
        [0.75, '#e74c3c'],   # Red - Critical risk
        [1.0, '#8e44ad']     # Purple - Extreme risk
    ]
    
    def __init__(self, storage: Optional[ALRIStorage] = None):
        """
        Initialize the dashboard.
        
        Args:
            storage: ALRIStorage instance for loading data
        """
        self.storage = storage or ALRIStorage()
        self._alri_data: List[ALRIRecord] = []
        self._time_series_data: Optional[pd.DataFrame] = None
    
    def load_data(self, filepath: str) -> None:
        """
        Load ALRI data from storage.
        
        Args:
            filepath: Path to JSON file with ALRI records
        """
        try:
            self._alri_data = self.storage.load(filepath)
        except FileNotFoundError:
            self._alri_data = []
    
    def set_data(self, records: List[ALRIRecord]) -> None:
        """
        Set ALRI data directly.
        
        Args:
            records: List of ALRIRecord instances
        """
        self._alri_data = records
    
    def set_time_series_data(self, df: pd.DataFrame) -> None:
        """
        Set time-series data for detailed charts.
        
        Args:
            df: DataFrame with time-series enrollment/update data
        """
        self._time_series_data = df
    
    def _get_alri_dataframe(self) -> pd.DataFrame:
        """Convert ALRI records to DataFrame for visualization."""
        if not self._alri_data:
            return pd.DataFrame()
        
        records = []
        for r in self._alri_data:
            record_dict = {
                'district': r.district,
                'state': r.state,
                'alri_score': r.alri_score,
                'coverage_risk': r.sub_scores.get('coverage', 0.0),
                'instability_risk': r.sub_scores.get('instability', 0.0),
                'biometric_risk': r.sub_scores.get('biometric', 0.0),
                'anomaly_factor': r.sub_scores.get('anomaly', 0.0),
                'reason_codes': ', '.join(r.reason_codes) if r.reason_codes else '',
                'computed_at': r.computed_at
            }
            records.append(record_dict)
        
        return pd.DataFrame(records)
    
    def render_filters(self) -> Dict[str, Any]:
        """
        Render state/district/time filters in sidebar.
        
        Requirements: 11.5
        
        Returns:
            Dictionary with selected filter values
        """
        st.sidebar.header("🔍 Filters")
        
        df = self._get_alri_dataframe()
        
        filters = {}
        
        # State filter
        if not df.empty and 'state' in df.columns:
            states = ['All States'] + sorted(df['state'].unique().tolist())
            selected_state = st.sidebar.selectbox(
                "Select State",
                options=states,
                index=0
            )
            filters['state'] = None if selected_state == 'All States' else selected_state
        else:
            filters['state'] = None
        
        # District filter (filtered by state)
        if not df.empty and 'district' in df.columns:
            if filters['state']:
                district_df = df[df['state'] == filters['state']]
            else:
                district_df = df
            
            districts = ['All Districts'] + sorted(district_df['district'].unique().tolist())
            selected_district = st.sidebar.selectbox(
                "Select District",
                options=districts,
                index=0
            )
            filters['district'] = None if selected_district == 'All Districts' else selected_district
        else:
            filters['district'] = None
        
        # Time period filter
        st.sidebar.subheader("Time Period")
        
        # Default to last 6 months
        default_end = datetime.now()
        default_start = default_end - timedelta(days=180)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=default_end,
                key="end_date"
            )
        
        filters['date_range'] = (
            start_date.isoformat() if start_date else None,
            end_date.isoformat() if end_date else None
        )
        
        # ALRI score threshold filter
        st.sidebar.subheader("Risk Threshold")
        min_score = st.sidebar.slider(
            "Minimum ALRI Score",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=5.0
        )
        filters['min_score'] = min_score
        
        return filters
    
    def render_heatmap(self, alri_scores: pd.DataFrame, filters: Dict[str, Any] = None) -> None:
        """
        Render district heatmap colored by ALRI score.
        
        Requirements: 11.1
        
        Args:
            alri_scores: DataFrame with district ALRI scores
            filters: Optional filter dictionary to apply
        """
        st.subheader("📊 District ALRI Heatmap")
        
        if alri_scores.empty:
            st.warning("No ALRI data available. Please load data first.")
            return
        
        # Apply filters
        df = alri_scores.copy()
        
        if filters:
            if filters.get('state'):
                df = df[df['state'] == filters['state']]
            if filters.get('district'):
                df = df[df['district'] == filters['district']]
            if filters.get('min_score'):
                df = df[df['alri_score'] >= filters['min_score']]
        
        if df.empty:
            st.info("No districts match the selected filters.")
            return
        
        # Create heatmap using treemap for district visualization
        # Group by state and district for hierarchical view
        fig = px.treemap(
            df,
            path=['state', 'district'],
            values='alri_score',
            color='alri_score',
            color_continuous_scale=self.ALRI_COLOR_SCALE,
            range_color=[0, 100],
            title='District ALRI Scores by State',
            hover_data={
                'alri_score': ':.1f',
                'coverage_risk': ':.2f',
                'instability_risk': ':.2f',
                'biometric_risk': ':.2f'
            }
        )
        
        fig.update_layout(
            height=500,
            margin=dict(t=50, l=25, r=25, b=25)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Also show a bar chart for easier comparison
        st.subheader("📈 ALRI Score Comparison")
        
        # Sort by ALRI score descending
        df_sorted = df.sort_values('alri_score', ascending=False).head(20)
        
        fig_bar = px.bar(
            df_sorted,
            x='district',
            y='alri_score',
            color='alri_score',
            color_continuous_scale=self.ALRI_COLOR_SCALE,
            range_color=[0, 100],
            title='Top 20 Districts by ALRI Score',
            labels={'alri_score': 'ALRI Score', 'district': 'District'},
            hover_data=['state', 'coverage_risk', 'instability_risk', 'biometric_risk']
        )
        
        fig_bar.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    def render_district_detail(self, district: str, state: str = None) -> None:
        """
        Render detailed time-series charts for selected district.
        
        Requirements: 11.2
        
        Args:
            district: District name to display
            state: Optional state name for filtering
        """
        st.subheader(f"📋 District Detail: {district.title()}")
        
        # Get district data
        df = self._get_alri_dataframe()
        
        if df.empty:
            st.warning("No data available for this district.")
            return
        
        # Filter for selected district
        district_df = df[df['district'].str.lower() == district.lower()]
        if state:
            district_df = district_df[district_df['state'].str.lower() == state.lower()]
        
        if district_df.empty:
            st.info(f"No ALRI data found for district: {district}")
            return
        
        # Display current ALRI metrics
        latest = district_df.iloc[0]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="ALRI Score",
                value=f"{latest['alri_score']:.1f}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Coverage Risk",
                value=f"{latest['coverage_risk']:.2f}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Instability Risk",
                value=f"{latest['instability_risk']:.2f}",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Biometric Risk",
                value=f"{latest['biometric_risk']:.2f}",
                delta=None
            )
        
        with col5:
            st.metric(
                label="Anomaly Factor",
                value=f"{latest['anomaly_factor']:.2f}",
                delta=None
            )
        
        # Sub-score breakdown chart
        st.subheader("Sub-Score Breakdown")
        
        sub_scores = {
            'Coverage Risk': latest['coverage_risk'],
            'Instability Risk': latest['instability_risk'],
            'Biometric Risk': latest['biometric_risk'],
            'Anomaly Factor': latest['anomaly_factor']
        }
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=list(sub_scores.values()),
            theta=list(sub_scores.keys()),
            fill='toself',
            name='Sub-Scores',
            line_color='#3498db'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            height=350
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Time-series charts if data available
        if self._time_series_data is not None and not self._time_series_data.empty:
            self._render_time_series_charts(district, state)
        
        # Reason codes
        if latest['reason_codes']:
            st.subheader("🏷️ Reason Codes")
            reason_codes = latest['reason_codes'].split(', ')
            for code in reason_codes:
                if code:
                    st.markdown(f"- **{code}**")
    
    def _render_time_series_charts(self, district: str, state: str = None) -> None:
        """Render time-series charts for enrollment, demographic, and biometric data."""
        ts_df = self._time_series_data.copy()
        
        # Filter for district
        ts_df = ts_df[ts_df['district'].str.lower() == district.lower()]
        if state:
            ts_df = ts_df[ts_df['state'].str.lower() == state.lower()]
        
        if ts_df.empty:
            return
        
        st.subheader("📈 Time-Series Trends")
        
        # Create date column if not exists
        if 'date' not in ts_df.columns:
            if all(col in ts_df.columns for col in ['year', 'month', 'day']):
                ts_df['date'] = pd.to_datetime(ts_df[['year', 'month', 'day']])
            elif all(col in ts_df.columns for col in ['year', 'month']):
                ts_df['date'] = pd.to_datetime(ts_df['year'].astype(str) + '-' + ts_df['month'].astype(str) + '-01')
        
        # Aggregate by date
        if 'date' in ts_df.columns:
            ts_df = ts_df.sort_values('date')
            
            # Create subplots for different metrics
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Enrollments', 'Demographic Updates', 'Biometric Updates'),
                vertical_spacing=0.1
            )
            
            # Enrollment trend
            if 'total_enrollment_age' in ts_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=ts_df['date'],
                        y=ts_df['total_enrollment_age'],
                        mode='lines+markers',
                        name='Enrollments',
                        line=dict(color='#2ecc71')
                    ),
                    row=1, col=1
                )
            
            # Demographic updates trend
            if 'total_demographic_age' in ts_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=ts_df['date'],
                        y=ts_df['total_demographic_age'],
                        mode='lines+markers',
                        name='Demographic Updates',
                        line=dict(color='#3498db')
                    ),
                    row=2, col=1
                )
            
            # Biometric updates trend
            if 'total_biometric_age' in ts_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=ts_df['date'],
                        y=ts_df['total_biometric_age'],
                        mode='lines+markers',
                        name='Biometric Updates',
                        line=dict(color='#9b59b6')
                    ),
                    row=3, col=1
                )
            
            fig.update_layout(
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts_panel(self, alerts: List[ALRIRecord] = None) -> None:
        """
        Render alerts panel showing threshold-crossing districts.
        
        Requirements: 11.3, 11.4
        
        Args:
            alerts: Optional list of ALRIRecord for alerts. 
                   If None, generates from loaded data.
        """
        st.subheader("🚨 Risk Alerts")
        
        if alerts is None:
            # Generate alerts from loaded data
            alerts = self._generate_alerts()
        
        if not alerts:
            st.success("✅ No districts currently exceed risk thresholds.")
            return
        
        # Group alerts by severity
        critical_alerts = []
        high_alerts = []
        medium_alerts = []
        
        for alert in alerts:
            if alert.alri_score >= self.ALERT_THRESHOLDS['critical']:
                critical_alerts.append(alert)
            elif alert.alri_score >= self.ALERT_THRESHOLDS['high']:
                high_alerts.append(alert)
            elif alert.alri_score >= self.ALERT_THRESHOLDS['medium']:
                medium_alerts.append(alert)
        
        # Display critical alerts
        if critical_alerts:
            st.error(f"🔴 **CRITICAL** ({len(critical_alerts)} districts)")
            for alert in critical_alerts[:5]:  # Show top 5
                with st.expander(f"{alert.district.title()}, {alert.state.title()} - Score: {alert.alri_score:.1f}"):
                    self._render_alert_detail(alert)
        
        # Display high alerts
        if high_alerts:
            st.warning(f"🟠 **HIGH RISK** ({len(high_alerts)} districts)")
            for alert in high_alerts[:5]:
                with st.expander(f"{alert.district.title()}, {alert.state.title()} - Score: {alert.alri_score:.1f}"):
                    self._render_alert_detail(alert)
        
        # Display medium alerts
        if medium_alerts:
            st.info(f"🟡 **MEDIUM RISK** ({len(medium_alerts)} districts)")
            for alert in medium_alerts[:5]:
                with st.expander(f"{alert.district.title()}, {alert.state.title()} - Score: {alert.alri_score:.1f}"):
                    self._render_alert_detail(alert)
        
        # Summary statistics
        st.subheader("📊 Alert Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Alerts", len(alerts))
        with col2:
            st.metric("Critical", len(critical_alerts), delta=None)
        with col3:
            st.metric("High Risk", len(high_alerts), delta=None)
        with col4:
            st.metric("Medium Risk", len(medium_alerts), delta=None)
    
    def _render_alert_detail(self, alert: ALRIRecord) -> None:
        """Render detailed information for a single alert."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sub-Scores:**")
            st.markdown(f"- Coverage Risk: {alert.sub_scores.get('coverage', 0):.2f}")
            st.markdown(f"- Instability Risk: {alert.sub_scores.get('instability', 0):.2f}")
            st.markdown(f"- Biometric Risk: {alert.sub_scores.get('biometric', 0):.2f}")
            st.markdown(f"- Anomaly Factor: {alert.sub_scores.get('anomaly', 0):.2f}")
        
        with col2:
            st.markdown("**Reason Codes:**")
            if alert.reason_codes:
                for code in alert.reason_codes:
                    st.markdown(f"- {code}")
            else:
                st.markdown("- No specific reason codes")
        
        st.markdown(f"*Computed at: {alert.computed_at}*")
    
    def _generate_alerts(self) -> List[ALRIRecord]:
        """Generate alerts from loaded ALRI data."""
        alerts = []
        for record in self._alri_data:
            if record.alri_score >= self.ALERT_THRESHOLDS['medium']:
                alerts.append(record)
        
        # Sort by score descending
        alerts.sort(key=lambda x: x.alri_score, reverse=True)
        return alerts
    
    def run(self) -> None:
        """
        Run the Streamlit dashboard application.
        
        This is the main entry point for the dashboard.
        """
        # Page configuration
        st.set_page_config(
            page_title="Aadhaar Sentinel - ALRI Dashboard",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("🛡️ Aadhaar Sentinel")
        st.markdown("### Aadhaar Lifecycle Risk Index (ALRI) Monitoring Dashboard")
        st.markdown("---")
        
        # Render filters in sidebar
        filters = self.render_filters()
        
        # Main content area
        tab1, tab2, tab3 = st.tabs(["📊 Heatmap", "📋 District Detail", "🚨 Alerts"])
        
        with tab1:
            df = self._get_alri_dataframe()
            self.render_heatmap(df, filters)
        
        with tab2:
            # District selection for detail view
            df = self._get_alri_dataframe()
            if not df.empty:
                selected_district = st.selectbox(
                    "Select District for Detail View",
                    options=sorted(df['district'].unique().tolist()),
                    key="detail_district"
                )
                if selected_district:
                    # Get state for the selected district
                    state = df[df['district'] == selected_district]['state'].iloc[0]
                    self.render_district_detail(selected_district, state)
            else:
                st.info("No data available. Please load ALRI data first.")
        
        with tab3:
            self.render_alerts_panel()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "*Aadhaar Sentinel - Decision Support System for UIDAI | "
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        )


def create_sample_data() -> List[ALRIRecord]:
    """Create sample ALRI data for demonstration."""
    import random
    
    states = ['karnataka', 'maharashtra', 'uttar pradesh', 'bihar', 'haryana', 'meghalaya']
    districts_by_state = {
        'karnataka': ['bengaluru urban', 'mysuru', 'mangaluru', 'hubli-dharwad'],
        'maharashtra': ['mumbai', 'pune', 'nagpur', 'aurangabad'],
        'uttar pradesh': ['lucknow', 'kanpur nagar', 'ghaziabad', 'aligarh', 'firozabad'],
        'bihar': ['patna', 'sitamarhi', 'madhubani', 'purbi champaran'],
        'haryana': ['faridabad', 'gurugram', 'panipat'],
        'meghalaya': ['east khasi hills', 'west garo hills']
    }
    
    reason_code_options = [
        'Low_Child_Enrolment',
        'High_Address_Churn',
        'Low_Biometric_Update_5to15',
        'Anomalous_Data_Entry'
    ]
    
    records = []
    for state, districts in districts_by_state.items():
        for district in districts:
            # Generate random sub-scores
            coverage = random.uniform(0.1, 0.9)
            instability = random.uniform(0.1, 0.9)
            biometric = random.uniform(0.1, 0.9)
            anomaly = random.uniform(0.0, 0.5)
            
            # Calculate ALRI score
            alri_score = (0.30 * coverage + 0.30 * instability + 0.30 * biometric + 0.10 * anomaly) * 100
            
            # Select reason codes based on highest sub-scores
            sub_scores_list = [
                ('Low_Child_Enrolment', coverage),
                ('High_Address_Churn', instability),
                ('Low_Biometric_Update_5to15', biometric),
                ('Anomalous_Data_Entry', anomaly)
            ]
            sub_scores_list.sort(key=lambda x: x[1], reverse=True)
            reason_codes = [code for code, _ in sub_scores_list[:2]]
            
            record = ALRIRecord(
                district=district,
                state=state,
                alri_score=alri_score,
                sub_scores={
                    'coverage': coverage,
                    'instability': instability,
                    'biometric': biometric,
                    'anomaly': anomaly
                },
                reason_codes=reason_codes,
                computed_at=datetime.now().isoformat()
            )
            records.append(record)
    
    return records


def main():
    """Main entry point for the dashboard."""
    dashboard = AadhaarSentinelDashboard()
    
    # Try to load data from storage, or use sample data
    data_path = "data/alri_storage/alri_results.json"
    
    try:
        dashboard.load_data(data_path)
    except FileNotFoundError:
        # Use sample data for demonstration
        st.sidebar.warning("Using sample data. Load actual ALRI results for production use.")
        sample_data = create_sample_data()
        dashboard.set_data(sample_data)
    
    dashboard.run()


if __name__ == "__main__":
    main()
