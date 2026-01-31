"""
ML Model Profiler - Streamlit Dashboard
Main UI entry point for model profiling
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
from pathlib import Path
import time


MCP_SERVER_URL = "http://localhost:8000"


st.set_page_config(
    page_title="ML Model Profiler",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .recommendation-card {
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
        border-radius: 5px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

if 'model_uploaded' not in st.session_state:
    st.session_state.model_uploaded = False
if 'profiling_done' not in st.session_state:
    st.session_state.profiling_done = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'model_info' not in st.session_state:
    st.session_state.model_info = None

def check_server_health():
    """Check if MCP server is running"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def upload_model_to_server(file, model_type, input_shape):
    """Upload model to MCP server"""
    try:
        files = {'model_file': file}
        data = {
            'model_type': model_type,
            'input_shape': input_shape
        }
        response = requests.post(
            f"{MCP_SERVER_URL}/upload-model",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Upload failed')
    except Exception as e:
        return False, str(e)

def run_profiling(config):
    """Trigger profiling on MCP server"""
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/run-profile",
            json=config,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Profiling failed')
    except Exception as e:
        return False, str(e)

def get_results():
    """Fetch profiling results from MCP server"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/results", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, None
    except Exception as e:
        return False, None

def get_recommendations():
    """Fetch recommendations from MCP server"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/recommendations", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, None
    except Exception as e:
        return False, None


def render_header():
    """Render header"""
    st.markdown('<h1 class="main-header">ML Model Profiler</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Optimize your models with intelligent profiling and recommendations</p>', unsafe_allow_html=True)

def render_server_status():
    """Render server connection status"""
    with st.sidebar:
        st.subheader("🔌 Server Status")
        if check_server_health():
            st.success("✅ Connected to MCP Server")
        else:
            st.error("❌ MCP Server not reachable")
            st.info(f"Please start the server at {MCP_SERVER_URL}")

def render_upload_section():
    """Render model upload section"""
    st.header("📤 Step 1: Upload Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a model file",
            type=['pt', 'pth', 'onnx'],
            help="Upload your PyTorch (.pt, .pth) or ONNX (.onnx) model"
        )
    
    with col2:
        model_type = st.selectbox(
            "Model Type",
            ["classification", "detection", "segmentation", "nlp", "custom"],
            help="Select the type of your model"
        )
        
        input_shape = st.text_input(
            "Input Shape",
            value="1,3,224,224",
            help="Enter input tensor shape (e.g., 1,3,224,224)"
        )
    
    if uploaded_file is not None:
        st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")
        
        if st.button("⬆️ Upload Model", key="upload_btn"):
            with st.spinner("Uploading model to server..."):
                success, result = upload_model_to_server(uploaded_file, model_type, input_shape)
                
                if success:
                    st.session_state.model_uploaded = True
                    st.session_state.model_info = result
                    st.success("✅ Model uploaded successfully!")
                    st.json(result)
                    st.rerun()
                else:
                    st.error(f"❌ Upload failed: {result}")

def render_profiling_section():
    """Render profiling configuration section"""
    st.header("⚙️ Step 2: Configure & Run Profiling")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Precision Modes")
        fp32 = st.checkbox("FP32", value=True)
        fp16 = st.checkbox("FP16", value=True)
        int8 = st.checkbox("INT8", value=False)
    
    with col2:
        st.subheader("Batch Sizes")
        batch_1 = st.checkbox("Batch 1", value=True)
        batch_4 = st.checkbox("Batch 4", value=True)
        batch_8 = st.checkbox("Batch 8", value=False)
        batch_16 = st.checkbox("Batch 16", value=False)
    
    with col3:
        st.subheader("Runtimes")
        pytorch = st.checkbox("PyTorch", value=True)
        onnx = st.checkbox("ONNX", value=False)
        tensorrt = st.checkbox("TensorRT", value=False)
        
        iterations = st.number_input(
            "Iterations",
            min_value=10,
            max_value=1000,
            value=100,
            help="Number of inference iterations for averaging"
        )
    
    # Build config
    precision_modes = []
    if fp32: precision_modes.append("fp32")
    if fp16: precision_modes.append("fp16")
    if int8: precision_modes.append("int8")
    
    batch_sizes = []
    if batch_1: batch_sizes.append(1)
    if batch_4: batch_sizes.append(4)
    if batch_8: batch_sizes.append(8)
    if batch_16: batch_sizes.append(16)
    
    runtimes = []
    if pytorch: runtimes.append("pytorch")
    if onnx: runtimes.append("onnx")
    if tensorrt: runtimes.append("tensorrt")
    
    config = {
        "precision_modes": precision_modes,
        "batch_sizes": batch_sizes,
        "runtimes": runtimes,
        "iterations": iterations
    }
    
    st.divider()
    
    if st.button("🚀 Start Profiling", key="profile_btn", type="primary"):
        if not precision_modes or not batch_sizes or not runtimes:
            st.error("⚠️ Please select at least one option from each category")
        else:
            with st.spinner("Running profiling... This may take a few minutes."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress (in production, poll server for real progress)
                for i in range(100):
                    time.sleep(0.1)
                    progress_bar.progress(i + 1)
                    status_text.text(f"Profiling... {i+1}%")
                
                success, result = run_profiling(config)
                
                if success:
                    st.success("✅ Profiling completed!")
                    
                    # Fetch results
                    success_res, results = get_results()
                    success_rec, recommendations = get_recommendations()
                    
                    if success_res and success_rec:
                        st.session_state.profiling_done = True
                        st.session_state.results = results
                        st.session_state.recommendations = recommendations
                        st.rerun()
                    else:
                        st.error("Failed to fetch results")
                else:
                    st.error(f"❌ Profiling failed: {result}")

def render_results_section():
    """Render profiling results"""
    st.header("📊 Step 3: Profiling Results")
    
    results = st.session_state.results
    
    if not results or 'configurations' not in results:
        st.warning("No results available")
        return
    
    # Summary Metrics
    st.subheader("📈 Performance Overview")
    
    summary = results.get('summary', {})
    best_latency = summary.get('best_latency', {})
    best_memory = summary.get('best_memory', {})
    best_gpu = summary.get('best_gpu_util', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "⚡ Best Latency",
            f"{best_latency.get('latency_mean', 0):.2f} ms",
            f"{best_latency.get('runtime', 'N/A')} | {best_latency.get('precision', 'N/A')}"
        )
    
    with col2:
        st.metric(
            "💾 Best Memory",
            f"{best_memory.get('memory_mean', 0):.0f} MB",
            f"Batch {best_memory.get('batch_size', 'N/A')}"
        )
    
    with col3:
        st.metric(
            "🔥 Best GPU Util",
            f"{best_gpu.get('gpu_util_mean', 0):.1f}%",
            f"{best_gpu.get('runtime', 'N/A')}"
        )
    
    with col4:
        throughput = (1000 * best_latency.get('batch_size', 1)) / best_latency.get('latency_mean', 1) if best_latency.get('latency_mean', 0) > 0 else 0
        st.metric(
            "📈 Throughput",
            f"{throughput:.1f} samples/s",
            f"Batch {best_latency.get('batch_size', 1)}"
        )
    
    st.divider()
    
    # Charts
    configs = results.get('configurations', [])
    valid_configs = [c for c in configs if 'error' not in c]
    
    if not valid_configs:
        st.warning("No valid configurations to display")
        return
    
    # Create charts
    col1, col2 = st.columns(2)
    
    with col1:
        render_latency_chart(valid_configs)
    
    with col2:
        render_memory_chart(valid_configs)
    
    col3, col4 = st.columns(2)
    
    with col3:
        render_batch_performance_chart(valid_configs)
    
    with col4:
        render_utilization_chart(valid_configs)
    
    # Detailed table
    st.subheader("📋 Detailed Configuration Results")
    render_results_table(valid_configs)

def render_latency_chart(configs):
    """Render latency comparison chart"""
    st.subheader("⚡ Latency Comparison")
    
    # Prepare data
    labels = [f"{c['runtime']}\n{c['precision']}\nB{c['batch_size']}" for c in configs]
    latencies = [c.get('latency_mean', 0) for c in configs]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=latencies,
            marker_color='#667eea',
            text=[f"{l:.2f} ms" for l in latencies],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        xaxis_title="Configuration",
        yaxis_title="Latency (ms)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_memory_chart(configs):
    """Render memory usage chart"""
    st.subheader("💾 Memory Usage")
    
    # Group by precision
    precisions = {}
    for c in configs:
        prec = c.get('precision', 'unknown')
        if prec not in precisions:
            precisions[prec] = []
        precisions[prec].append(c.get('memory_mean', 0))
    
    fig = go.Figure()
    
    for prec, mems in precisions.items():
        fig.add_trace(go.Box(
            y=mems,
            name=prec.upper(),
            boxmean='sd'
        ))
    
    fig.update_layout(
        yaxis_title="Memory Usage (MB)",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_batch_performance_chart(configs):
    """Render batch size performance chart"""
    st.subheader("📦 Batch Size Performance")
    
    # Group by batch size
    batch_data = {}
    for c in configs:
        batch = c.get('batch_size', 0)
        if batch not in batch_data:
            batch_data[batch] = {'latencies': [], 'throughputs': []}
        
        latency = c.get('latency_mean', 0)
        throughput = (1000 * batch / latency) if latency > 0 else 0
        
        batch_data[batch]['latencies'].append(latency)
        batch_data[batch]['throughputs'].append(throughput)
    
    # Create subplot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    batches = sorted(batch_data.keys())
    avg_latencies = [sum(batch_data[b]['latencies']) / len(batch_data[b]['latencies']) for b in batches]
    avg_throughputs = [sum(batch_data[b]['throughputs']) / len(batch_data[b]['throughputs']) for b in batches]
    
    fig.add_trace(
        go.Scatter(x=batches, y=avg_latencies, name="Latency", mode='lines+markers', line=dict(color='#667eea')),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=batches, y=avg_throughputs, name="Throughput", mode='lines+markers', line=dict(color='#764ba2')),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Batch Size")
    fig.update_yaxes(title_text="Latency (ms)", secondary_y=False)
    fig.update_yaxes(title_text="Throughput (samples/s)", secondary_y=True)
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)

def render_utilization_chart(configs):
    """Render resource utilization chart"""
    st.subheader("🔥 Resource Utilization")
    
    # Average by runtime
    runtime_data = {}
    for c in configs:
        runtime = c.get('runtime', 'unknown')
        if runtime not in runtime_data:
            runtime_data[runtime] = {'gpu': [], 'cpu': [], 'memory': []}
        
        runtime_data[runtime]['gpu'].append(c.get('gpu_util_mean', 0))
        runtime_data[runtime]['cpu'].append(c.get('cpu_util_mean', 0))
        runtime_data[runtime]['memory'].append(c.get('memory_percent', 0))
    
    # Create radar chart
    categories = list(runtime_data.keys())
    
    fig = go.Figure()
    
    metrics = ['GPU Util', 'CPU Util', 'Memory %']
    for metric, key in zip(metrics, ['gpu', 'cpu', 'memory']):
        values = [sum(runtime_data[r][key]) / len(runtime_data[r][key]) if runtime_data[r][key] else 0 for r in categories]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=metric
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_results_table(configs):
    """Render detailed results table"""
    df = pd.DataFrame(configs)
    
    # Select and rename columns
    display_cols = {
        'runtime': 'Runtime',
        'precision': 'Precision',
        'batch_size': 'Batch',
        'latency_mean': 'Latency (ms)',
        'memory_mean': 'Memory (MB)',
        'gpu_util_mean': 'GPU %',
        'cpu_util_mean': 'CPU %'
    }
    
    df_display = df[[col for col in display_cols.keys() if col in df.columns]].rename(columns=display_cols)
    
    # Format numbers
    if 'Latency (ms)' in df_display.columns:
        df_display['Latency (ms)'] = df_display['Latency (ms)'].apply(lambda x: f"{x:.2f}")
    if 'Memory (MB)' in df_display.columns:
        df_display['Memory (MB)'] = df_display['Memory (MB)'].apply(lambda x: f"{x:.0f}")
    if 'GPU %' in df_display.columns:
        df_display['GPU %'] = df_display['GPU %'].apply(lambda x: f"{x:.1f}")
    if 'CPU %' in df_display.columns:
        df_display['CPU %'] = df_display['CPU %'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(df_display, use_container_width=True)

def render_recommendations_section():
    """Render recommendations"""
    st.header("💡 Step 4: Optimization Recommendations")
    
    recs = st.session_state.recommendations
    
    if not recs:
        st.warning("No recommendations available")
        return
    
    recommendations = recs.get('recommendations', [])
    bottlenecks = recs.get('bottlenecks', [])
    
    # Bottleneck Analysis
    if bottlenecks:
        st.subheader("🔍 Bottleneck Analysis")
        
        for bottleneck in bottlenecks[:5]:  # Show top 5
            severity = bottleneck.get('severity', 'medium')
            
            if severity == 'high':
                alert_type = 'error'
                icon = '🔴'
            elif severity == 'medium':
                alert_type = 'warning'
                icon = '🟡'
            else:
                alert_type = 'info'
                icon = '🟢'
            
            with st.expander(f"{icon} {bottleneck.get('title', 'Issue')}", expanded=(severity=='high')):
                st.write(bottleneck.get('description', ''))
                if 'suggestions' in bottleneck:
                    st.write("**Suggestions:**")
                    for suggestion in bottleneck['suggestions']:
                        st.write(f"- {suggestion}")
    
    st.divider()
    
    # Recommendations
    if recommendations:
        st.subheader("✨ Actionable Recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            priority = rec.get('priority', 'medium')
            
            if priority == 'high':
                priority_color = '#dc3545'
                priority_badge = '🔴 HIGH'
            elif priority == 'medium':
                priority_color = '#ffc107'
                priority_badge = '🟡 MEDIUM'
            else:
                priority_color = '#28a745'
                priority_badge = '🟢 LOW'
            
            st.markdown(f"""
            <div style="border-left: 4px solid {priority_color}; padding: 1rem; margin: 1rem 0; background: #f8f9fa; border-radius: 5px;">
                <h4>{i}. {rec.get('title', 'Recommendation')}</h4>
                <span style="background: {priority_color}; color: white; padding: 0.25rem 0.5rem; border-radius: 3px; font-size: 0.8rem; font-weight: bold;">{priority_badge}</span>
                <p style="margin-top: 1rem;">{rec.get('description', '')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if 'impact' in rec:
                impact = rec['impact']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Latency Impact", impact.get('latency', 'N/A'))
                with col2:
                    st.metric("Memory Impact", impact.get('memory', 'N/A'))
                with col3:
                    st.metric("Throughput Impact", impact.get('throughput', 'N/A'))
            
            st.write("")

# ============================================================================
# MAIN APP FLOW
# ============================================================================

def main():
    """Main application flow"""
    
    # Header
    render_header()
    
    # Server status in sidebar
    render_server_status()
    
    # Step 1: Upload
    if not st.session_state.model_uploaded:
        render_upload_section()
    else:
        with st.expander("✅ Model Uploaded - Click to view details"):
            st.json(st.session_state.model_info)
        
        # Step 2: Profiling
        if not st.session_state.profiling_done:
            render_profiling_section()
        else:
            st.success("✅ Profiling Completed")
            
            # Step 3: Results
            render_results_section()
            
            st.divider()
            
            # Step 4: Recommendations
            render_recommendations_section()
            
            st.divider()
            
            # Actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Export Results"):
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(st.session_state.results, indent=2),
                        file_name="profiling_results.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("🔄 New Profiling"):
                    st.session_state.profiling_done = False
                    st.rerun()
            
            with col3:
                if st.button("🆕 Upload New Model"):
                    st.session_state.model_uploaded = False
                    st.session_state.profiling_done = False
                    st.session_state.results = None
                    st.session_state.recommendations = None
                    st.rerun()

if __name__ == "__main__":
    main()