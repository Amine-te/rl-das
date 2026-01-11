"""
Interactive RL-DAS Evaluation Dashboard.

A comprehensive Streamlit dashboard for evaluating RL-DAS agents on TSP problems.
Provides interactive parameter tuning, decision-making visualization, and baseline comparisons.
"""

import streamlit as st
import os
import time
import tempfile
import numpy as np
import pandas as pd
import altair as alt
import torch
from pathlib import Path
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Core imports
from problems import TSPProblem
from utils.tsp_loader import load_tsp_instance
from algorithms import GeneticAlgorithm, TabuSearch, SimulatedAnnealing, IteratedLocalSearch
from core import DASGymEnv

# Page Config
st.set_page_config(
    page_title="RL-DAS Interactive Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .stMetric label {
        color: rgba(255,255,255,0.8) !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
    }
    .step-table {
        font-size: 12px;
    }
    div[data-testid="stExpander"] {
        background: #f8f9fa;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
ALGO_NAMES = ['GA', 'TS', 'SA', 'ILS']
ALGO_COLORS = {'GA': '#FF6B6B', 'TS': '#4ECDC4', 'SA': '#45B7D1', 'ILS': '#96CEB4', 'RL-DAS': '#667eea'}
DEFAULT_MODEL_PATH = "checkpoints/restored_ppo/best_model.zip"


# --- Helper Functions ---

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temp file and return the path."""
    try:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded_file.getbuffer())
        return f.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def load_model_from_path(model_path):
    """Load model, inferring type."""
    try:
        model = PPO.load(model_path)
        return model, "PPO"
    except:
        try:
            model = DQN.load(model_path)
            return model, "DQN"
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None, None


def create_algorithms(problem):
    """Create algorithm instances."""
    return [
        GeneticAlgorithm(problem, population_size=min(50, problem.size), tournament_size=3),
        TabuSearch(problem, tabu_tenure=min(20, problem.size // 2), neighborhood_size=min(50, problem.size * 2), aspiration_enabled=True),
        SimulatedAnnealing(problem, initial_temperature=100.0, cooling_rate=0.995, min_temperature=0.01),
        IteratedLocalSearch(problem, perturbation_strength=max(2, problem.size // 10), local_search_max_iters=30)
    ]


def evaluate_single_baseline(problem, algo_name, algo, max_fes, interval_fes):
    """Run a single baseline algorithm."""
    start_time = time.time()
    algo.initialize()
    
    # Initial solution
    sol = problem.generate_random_solution()
    cost = problem.evaluate(sol)
    algo.inject_solution(sol, cost)
    
    current_fes = 0
    best_cost = cost
    history = [{'fes': 0, 'cost': cost}]
    
    while current_fes < max_fes:
        fes_allowed = min(interval_fes, max_fes - current_fes)
        _, step_cost = algo.step(fes_allowed)
        current_fes += fes_allowed
        
        if step_cost < best_cost:
            best_cost = step_cost
            
        history.append({'fes': current_fes, 'cost': best_cost})
        
    return {
        'final_cost': best_cost,
        'time': time.time() - start_time,
        'history': history,
        'best_tour': algo.get_best()[0]
    }


def evaluate_baselines(problem, max_fes, interval_fes, progress_callback=None):
    """Run all baseline algorithms."""
    results = {}
    algorithms = create_algorithms(problem)
    
    for idx, (name, algo) in enumerate(zip(ALGO_NAMES, algorithms)):
        if progress_callback:
            progress_callback(idx, len(algorithms), name)
        results[name] = evaluate_single_baseline(problem, name, algo, max_fes, interval_fes)
        
    return results


def evaluate_rl_agent(model, model_type, problem, max_fes, interval_fes, population_size, deterministic=True):
    """Evaluate RL agent with detailed step tracking."""
    start_time = time.time()
    
    # Create environment
    def make_env():
        algorithms = create_algorithms(problem)
        return DASGymEnv(
            problem=problem, 
            algorithms=algorithms, 
            max_fes=max_fes, 
            interval_fes=interval_fes, 
            population_size=population_size
        )
    
    env = DummyVecEnv([make_env])
    obs = env.reset()
    done = False
    
    # Get inner environment
    unwrapped = env.envs[0]
    inner_env = unwrapped.das_env if hasattr(unwrapped, 'das_env') else unwrapped
    
    initial_cost = inner_env.last_best_cost
    
    episode_data = {
        'history': [{'fes': 0, 'cost': initial_cost}],
        'steps': [],
        'final_cost': 0,
        'total_steps': 0,
        'switches': 0,
        'best_tour': None
    }
    
    step_count = 0
    prev_action = None
    
    while not done:
        # Get action and probabilities
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # Get action probabilities for PPO
        probs = None
        if model_type == "PPO":
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs).to(model.device)
                dist = model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]
        elif model_type == "DQN":
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs).to(model.device)
                q_values = model.q_net(obs_tensor).cpu().numpy()[0]
                # Convert Q-values to pseudo-probabilities via softmax
                exp_q = np.exp(q_values - np.max(q_values))
                probs = exp_q / exp_q.sum()
        
        # Execute step
        obs, reward, done, info = env.step(action)
        
        current_cost = info[0]['best_cost']
        current_fes = inner_env.current_fes
        action_idx = int(action[0])
        
        # Record step data
        step_data = {
            'step': step_count,
            'action': action_idx,
            'algorithm': ALGO_NAMES[action_idx],
            'cost': current_cost,
            'reward': float(reward[0]),
            'probs': probs,
            'is_switch': prev_action is not None and prev_action != action_idx
        }
        episode_data['steps'].append(step_data)
        episode_data['history'].append({'fes': current_fes, 'cost': current_cost})
        
        prev_action = action_idx
        step_count += 1
    
    episode_data['final_cost'] = float(info[0]['best_cost'])
    episode_data['time'] = time.time() - start_time
    episode_data['total_steps'] = step_count
    episode_data['switches'] = inner_env.context_manager.get_switch_count()
    episode_data['best_tour'] = inner_env.context_manager.get_global_best()[0]
    episode_data['initial_cost'] = initial_cost
    
    return episode_data


def plot_convergence(rl_history, baseline_results):
    """Create convergence comparison plot using layered charts."""
    layers = []
    
    # Helper to create a single line chart for one algorithm
    def make_line(data, algo_name, color, is_solid=False):
        df = pd.DataFrame(data)
        df['Algorithm'] = algo_name  # Add algorithm name for legend
        df = df.sort_values('fes').reset_index(drop=True)
        
        line = alt.Chart(df).mark_line(
            strokeWidth=2.5 if is_solid else 2,
            strokeDash=[] if is_solid else [5, 5]
        ).encode(
            x=alt.X('fes:Q', title='Function Evaluations'),
            y=alt.Y('cost:Q', title='Cost (Tour Length)', scale=alt.Scale(zero=False)),
            color=alt.Color('Algorithm:N', scale=alt.Scale(
                domain=['RL-DAS'] + ALGO_NAMES,
                range=[ALGO_COLORS['RL-DAS']] + [ALGO_COLORS[n] for n in ALGO_NAMES]
            )),
            tooltip=[
                alt.Tooltip('Algorithm:N'),
                alt.Tooltip('fes:Q', title='FEs'),
                alt.Tooltip('cost:Q', title='Cost', format='.2f')
            ]
        )
        return line
    
    # Add RL-DAS line (solid, thicker)
    rl_data = [{'fes': int(p['fes']), 'cost': float(p['cost'])} for p in rl_history]
    layers.append(make_line(rl_data, 'RL-DAS', ALGO_COLORS['RL-DAS'], is_solid=True))
    
    # Add baseline lines (dashed)
    for name in ALGO_NAMES:
        if name in baseline_results:
            baseline_data = [{'fes': int(p['fes']), 'cost': float(p['cost'])} 
                           for p in baseline_results[name]['history']]
            layers.append(make_line(baseline_data, name, ALGO_COLORS[name], is_solid=False))
    
    # Combine all layers with proper sizing
    chart = alt.layer(*layers).properties(
        height=400,
        title='Convergence Comparison'
    ).interactive()
    
    return chart


def plot_final_costs_bar(rl_cost, baseline_results):
    """Create bar chart of final costs."""
    data = [{'Algorithm': 'RL-DAS', 'Cost': rl_cost}]
    for name, res in baseline_results.items():
        data.append({'Algorithm': name, 'Cost': res['final_cost']})
    
    df = pd.DataFrame(data)
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Algorithm:N', sort=['RL-DAS'] + ALGO_NAMES),
        y=alt.Y('Cost:Q', scale=alt.Scale(zero=False)),
        color=alt.Color('Algorithm:N', scale=alt.Scale(
            domain=['RL-DAS'] + ALGO_NAMES,
            range=[ALGO_COLORS['RL-DAS']] + [ALGO_COLORS[n] for n in ALGO_NAMES]
        )),
        tooltip=['Algorithm', 'Cost']
    ).properties(height=300)
    
    return chart


def plot_action_timeline(steps_data):
    """Create action timeline visualization."""
    df = pd.DataFrame([{
        'Step': s['step'],
        'Algorithm': s['algorithm'],
        'Cost': s['cost']
    } for s in steps_data])
    
    timeline = alt.Chart(df).mark_rect().encode(
        x=alt.X('Step:O', title='Decision Step'),
        color=alt.Color('Algorithm:N', scale=alt.Scale(
            domain=ALGO_NAMES,
            range=[ALGO_COLORS[n] for n in ALGO_NAMES]
        )),
        tooltip=['Step', 'Algorithm', 'Cost']
    ).properties(height=60)
    
    return timeline


def plot_probability_heatmap(steps_data):
    """Create probability heatmap over time."""
    heatmap_data = []
    for step in steps_data:
        if step['probs'] is not None:
            for i, algo in enumerate(ALGO_NAMES):
                heatmap_data.append({
                    'Step': step['step'],
                    'Algorithm': algo,
                    'Probability': step['probs'][i]
                })
    
    if not heatmap_data:
        return None
    
    df = pd.DataFrame(heatmap_data)
    
    heatmap = alt.Chart(df).mark_rect().encode(
        x=alt.X('Step:O', title='Decision Step'),
        y=alt.Y('Algorithm:N', sort=ALGO_NAMES),
        color=alt.Color('Probability:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Step', 'Algorithm', alt.Tooltip('Probability:Q', format='.3f')]
    ).properties(height=150)
    
    return heatmap


def plot_tour(problem, tour, title, color='blue'):
    """Plot a TSP tour."""
    if not hasattr(problem, 'coordinates') or tour is None:
        return None
    
    city_coords = problem.coordinates[tour]
    city_coords = np.vstack([city_coords, city_coords[0]])  # Close the loop
    
    df_tour = pd.DataFrame(city_coords, columns=['x', 'y'])
    df_tour['order'] = range(len(df_tour))
    df_cities = pd.DataFrame(problem.coordinates, columns=['x', 'y'])
    df_cities['city'] = range(len(df_cities))
    
    # City points
    points = alt.Chart(df_cities).mark_circle(size=60, color='gray').encode(
        x='x:Q', y='y:Q',
        tooltip=['city']
    )
    
    # Tour path
    path = alt.Chart(df_tour).mark_line(color=color, strokeWidth=2).encode(
        x='x:Q', y='y:Q',
        order='order:O'
    )
    
    return (points + path).properties(title=title, height=350)


def create_comparison_table(rl_results, baseline_results):
    """Create comparison dataframe."""
    rl_cost = rl_results['final_cost']
    
    data = [{
        'Algorithm': 'RL-DAS',
        'Final Cost': f"{rl_cost:.2f}",
        'Time (s)': f"{rl_results['time']:.2f}",
        'Steps': rl_results['total_steps'],
        'Switches': rl_results['switches'],
        'Gap vs Best': '-'
    }]
    
    best_baseline = min(baseline_results.values(), key=lambda x: x['final_cost'])['final_cost']
    
    for name, res in baseline_results.items():
        gap = ((res['final_cost'] - rl_cost) / rl_cost) * 100
        data.append({
            'Algorithm': name,
            'Final Cost': f"{res['final_cost']:.2f}",
            'Time (s)': f"{res['time']:.2f}",
            'Steps': '-',
            'Switches': '-',
            'Gap vs RL-DAS': f"{gap:+.2f}%"
        })
    
    return pd.DataFrame(data)


def create_steps_dataframe(steps_data):
    """Create detailed steps dataframe."""
    rows = []
    for s in steps_data:
        row = {
            'Step': s['step'],
            'Action': s['action'],
            'Algorithm': s['algorithm'],
            'Cost': f"{s['cost']:.4f}",
            'Reward': f"{s['reward']:.4f}",
            'Switch': '🔄' if s['is_switch'] else ''
        }
        # Add probabilities
        if s['probs'] is not None:
            for i, algo in enumerate(ALGO_NAMES):
                row[f'P({algo})'] = f"{s['probs'][i]:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


# --- Main UI ---

st.title("🧠 RL-DAS Interactive Dashboard")
st.markdown("Evaluate and visualize **Reinforcement Learning Dynamic Algorithm Selection** on TSP problems.")

# Initialize session state
if 'rl_results' not in st.session_state:
    st.session_state.rl_results = None
if 'baseline_results' not in st.session_state:
    st.session_state.baseline_results = None
if 'problem' not in st.session_state:
    st.session_state.problem = None

# --- Sidebar ---
with st.sidebar:
    st.header("📁 Problem Setup")
    
    problem_source = st.radio("Problem Source", ["Upload .tsp file", "Generate Synthetic"])
    
    if problem_source == "Upload .tsp file":
        uploaded_problem = st.file_uploader("Upload TSP Instance", type=["tsp"])
    else:
        num_cities = st.slider("Number of Cities", 10, 200, 50)
        distribution = st.selectbox("Distribution", ["uniform", "clustered", "mixed"])
        gen_seed = st.number_input("Seed", 0, 10000, 42)
    
    st.divider()
    st.header("🤖 Model")
    
    use_default = st.checkbox("Use Default Model", value=True)
    if use_default:
        model_path = DEFAULT_MODEL_PATH
        st.caption(f"📍 `{model_path}`")
    else:
        uploaded_model = st.file_uploader("Upload RL Model (.zip)", type=["zip"])
        model_path = None
    
    st.divider()
    st.header("⚙️ Parameters")
    
    max_fes = st.number_input("Max FEs", 1000, 1000000, 100000, step=1000)
    interval_fes = st.number_input("Interval FEs", 100, 5000, 1000, step=100)
    pop_size = st.number_input("Population Size", 10, 100, 30, step=5)
    deterministic = st.checkbox("Deterministic Policy", value=True)
    
    st.divider()
    
    col_run1, col_run2 = st.columns(2)
    with col_run1:
        run_rl = st.button("🚀 Run RL-DAS", type="primary", use_container_width=True)
    with col_run2:
        run_baselines = st.button("📊 Run Baselines", use_container_width=True)
    
    run_both = st.button("⚡ Run Both", use_container_width=True)


# --- Main Content ---

# Load problem if needed
problem = None
if problem_source == "Upload .tsp file":
    if 'uploaded_problem' in dir() and uploaded_problem is not None:
        problem_path = save_uploaded_file(uploaded_problem)
        if problem_path:
            try:
                problem = load_tsp_instance(problem_path)
                os.remove(problem_path)
            except Exception as e:
                st.error(f"Failed to load TSP file: {e}")
else:
    problem = TSPProblem(num_cities=num_cities, distribution=distribution, seed=gen_seed)

# Store problem in session
if problem is not None:
    st.session_state.problem = problem

# Display problem info
if st.session_state.problem is not None:
    prob = st.session_state.problem
    st.info(f"📍 **Problem:** {getattr(prob, 'name', 'Synthetic')} | **Cities:** {prob.size}")

# Handle Run buttons
if run_rl or run_both:
    if st.session_state.problem is None:
        st.warning("Please load a TSP problem first!")
    elif model_path is None and not use_default:
        st.warning("Please upload a model or use the default!")
    else:
        # Load model
        actual_path = model_path if use_default else save_uploaded_file(uploaded_model)
        if actual_path and os.path.exists(actual_path):
            with st.spinner("Loading model..."):
                model, model_type = load_model_from_path(actual_path)
            
            if model:
                with st.spinner(f"Running RL-DAS ({model_type})..."):
                    st.session_state.rl_results = evaluate_rl_agent(
                        model, model_type, st.session_state.problem,
                        max_fes, interval_fes, pop_size, deterministic
                    )
                st.success("✅ RL-DAS evaluation complete!")
        else:
            st.error(f"Model file not found: {actual_path}")

if run_baselines or run_both:
    if st.session_state.problem is None:
        st.warning("Please load a TSP problem first!")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(idx, total, name):
            progress_bar.progress((idx + 1) / total)
            status_text.text(f"Running {name}...")
        
        st.session_state.baseline_results = evaluate_baselines(
            st.session_state.problem, max_fes, interval_fes, update_progress
        )
        
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Baseline evaluations complete!")

# --- Results Display ---
if st.session_state.rl_results is not None or st.session_state.baseline_results is not None:
    st.divider()
    
    # KPI Metrics
    st.subheader("📊 Performance Summary")
    
    cols = st.columns(5)
    
    if st.session_state.rl_results:
        rl = st.session_state.rl_results
        improvement = ((rl['initial_cost'] - rl['final_cost']) / rl['initial_cost']) * 100
        
        cols[0].metric("RL-DAS Cost", f"{rl['final_cost']:.2f}")
        cols[1].metric("Improvement", f"{improvement:.1f}%")
        cols[2].metric("Time", f"{rl['time']:.2f}s")
        cols[3].metric("Steps", rl['total_steps'])
        cols[4].metric("Switches", rl['switches'])
    
    # Comparison Table
    if st.session_state.rl_results and st.session_state.baseline_results:
        st.subheader("📋 Comparison Table")
        comp_df = create_comparison_table(st.session_state.rl_results, st.session_state.baseline_results)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    # Convergence Plot
    if st.session_state.rl_results and st.session_state.baseline_results:
        st.subheader("📈 Convergence Comparison")
        conv_chart = plot_convergence(
            st.session_state.rl_results['history'],
            st.session_state.baseline_results
        )
        st.altair_chart(conv_chart, use_container_width=True)
    
    # RL Decision Process
    if st.session_state.rl_results:
        st.subheader("🤖 RL Decision-Making Process")
        
        tab1, tab2, tab3 = st.tabs(["📊 Timeline", "🔥 Probability Heatmap", "📝 Step Details"])
        
        with tab1:
            col_timeline, col_dist = st.columns([2, 1])
            
            with col_timeline:
                st.write("**Algorithm Selection Timeline**")
                timeline = plot_action_timeline(st.session_state.rl_results['steps'])
                st.altair_chart(timeline, use_container_width=True)
            
            with col_dist:
                st.write("**Selection Distribution**")
                action_counts = pd.Series([s['algorithm'] for s in st.session_state.rl_results['steps']]).value_counts()
                dist_df = action_counts.reset_index()
                dist_df.columns = ['Algorithm', 'Count']
                
                pie_chart = alt.Chart(dist_df).mark_arc(innerRadius=50).encode(
                    theta='Count:Q',
                    color=alt.Color('Algorithm:N', scale=alt.Scale(
                        domain=ALGO_NAMES,
                        range=[ALGO_COLORS[n] for n in ALGO_NAMES]
                    )),
                    tooltip=['Algorithm', 'Count']
                ).properties(height=200)
                st.altair_chart(pie_chart, use_container_width=True)
        
        with tab2:
            heatmap = plot_probability_heatmap(st.session_state.rl_results['steps'])
            if heatmap:
                st.altair_chart(heatmap, use_container_width=True)
            else:
                st.info("Probability data not available (DQN models show Q-values instead)")
        
        with tab3:
            steps_df = create_steps_dataframe(st.session_state.rl_results['steps'])
            st.dataframe(steps_df, use_container_width=True, hide_index=True, height=400)
    
    # Final Cost Comparison
    if st.session_state.rl_results and st.session_state.baseline_results:
        st.subheader("🏆 Final Cost Comparison")
        bar_chart = plot_final_costs_bar(
            st.session_state.rl_results['final_cost'],
            st.session_state.baseline_results
        )
        st.altair_chart(bar_chart, use_container_width=True)
    
    # Tour Visualization
    if st.session_state.problem and hasattr(st.session_state.problem, 'coordinates'):
        st.subheader("🗺️ Tour Visualization")
        
        tour_cols = st.columns(2)
        
        with tour_cols[0]:
            if st.session_state.rl_results:
                rl_tour_chart = plot_tour(
                    st.session_state.problem,
                    st.session_state.rl_results['best_tour'],
                    f"RL-DAS Tour (Cost: {st.session_state.rl_results['final_cost']:.2f})",
                    color=ALGO_COLORS['RL-DAS']
                )
                if rl_tour_chart:
                    st.altair_chart(rl_tour_chart, use_container_width=True)
        
        with tour_cols[1]:
            if st.session_state.baseline_results:
                best_baseline_name = min(
                    st.session_state.baseline_results,
                    key=lambda k: st.session_state.baseline_results[k]['final_cost']
                )
                best_baseline = st.session_state.baseline_results[best_baseline_name]
                baseline_tour_chart = plot_tour(
                    st.session_state.problem,
                    best_baseline['best_tour'],
                    f"Best Baseline ({best_baseline_name}) (Cost: {best_baseline['final_cost']:.2f})",
                    color=ALGO_COLORS[best_baseline_name]
                )
                if baseline_tour_chart:
                    st.altair_chart(baseline_tour_chart, use_container_width=True)

else:
    # Initial state - show instructions
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Load a TSP Problem** - Upload a `.tsp` file or generate a synthetic instance
    2. **Configure Parameters** - Adjust evaluation settings in the sidebar
    3. **Run Evaluation** - Click "Run RL-DAS" and/or "Run Baselines"
    4. **Analyze Results** - Explore the decision-making process and comparisons
    
    ---
    
    **Features:**
    - 📊 Convergence comparison plots
    - 🤖 Step-by-step RL decision visualization
    - 🔥 Action probability heatmaps
    - 🗺️ Tour visualizations
    - 📋 Detailed comparison tables
    """)
