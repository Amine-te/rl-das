import streamlit as st
import os
import time
import tempfile
import numpy as np
import pandas as pd
import altair as alt
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Core imports (assuming running from root)
from problems import TSPProblem
from utils.tsp_loader import load_tsp_instance
from algorithms import GeneticAlgorithm, TabuSearch, SimulatedAnnealing, IteratedLocalSearch
from core import DASGymEnv

# Page Config
st.set_page_config(
    page_title="RL-DAS Evaluation Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        # Try Loading as PPO first (most common)
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

def evaluate_baselines(problem, max_fes, interval_fes):
    """Run baseline algorithms."""
    results = {}
    algo_names = ['GA', 'TS', 'SA', 'ILS']
    algorithms = create_algorithms(problem)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = len(algorithms)
    
    for idx, (name, algo) in enumerate(zip(algo_names, algorithms)):
        status_text.text(f"Running Baseline: {name}...")
        
        start_time = time.time()
        algo.initialize()
        
        # Initial solution
        sol = problem.generate_random_solution()
        cost = problem.evaluate(sol)
        algo.inject_solution(sol, cost)
        
        current_fes = 0
        best_cost = cost
        history = [{'fes': 0, 'cost': cost, 'algo': name}]
        
        # Run loop to collect history
        while current_fes < max_fes:
            fes_allowed = min(interval_fes, max_fes - current_fes)
            _, step_cost = algo.step(fes_allowed)
            current_fes += fes_allowed
            
            if step_cost < best_cost:
                best_cost = step_cost
                
            history.append({'fes': current_fes, 'cost': best_cost, 'algo': name})
            
        results[name] = {
            'final_cost': best_cost,
            'time': time.time() - start_time,
            'history': history,
            'best_tour': algo.get_best()[0]
        }
        progress_bar.progress((idx + 1) / total_steps)
        
    status_text.empty()
    progress_bar.empty()
    return results

def evaluate_rl_agent(model, model_type, problem, max_fes, interval_fes, population_size, deterministic=True):
    """Evaluate RL agent."""
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
    # Note: We are skipping VecNormalize loading for simplicity in this dashboard version 
    # unless user uploads it separately. Usually fine for simple visual checks.
    
    obs = env.reset()
    done = False
    
    algo_names = ['GA', 'TS', 'SA', 'ILS']
    
    episode_data = {
        'history': [],
        'actions': [],
        'action_names': [],
        'final_cost': 0,
        'steps': 0,
        'switches': 0,
        'best_tour': None
    }
    
    # Get initial info
    unwrapped = env.envs[0]
    if hasattr(unwrapped, 'das_env'):
         inner_env = unwrapped.das_env
    else:
         inner_env = unwrapped
         
    initial_cost = inner_env.last_best_cost
    episode_data['history'].append({'fes': 0, 'cost': initial_cost, 'algo': 'RL-DAS'})
    
    step_count = 0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        
        current_cost = info[0]['best_cost']
        current_fes = inner_env.current_fes
        
        episode_data['history'].append({'fes': current_fes, 'cost': current_cost, 'algo': 'RL-DAS'})
        episode_data['actions'].append(int(action[0]))
        episode_data['action_names'].append(algo_names[int(action[0])])
        
        step_count += 1
    
    episode_data['final_cost'] = float(info[0]['best_cost'])
    episode_data['time'] = time.time() - start_time
    episode_data['steps'] = step_count
    episode_data['switches'] = inner_env.context_manager.get_switch_count()
    episode_data['best_tour'] = inner_env.context_manager.get_global_best()[0]
    
    return episode_data


# --- UI Layout ---

st.title("🧠 RL-DAS Evaluation Dashboard")
st.markdown("Compare your **Reinforcement Learning Agent** against standard metaheuristics on TSP problems.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("1. Load Model")
    uploaded_model = st.file_uploader("Upload RL Agent (.zip)", type="zip")
    
    st.subheader("2. Load Problem")
    uploaded_problem = st.file_uploader("Upload TSP Instance (.tsp)", type="tsp")
    
    st.subheader("3. Settings")
    max_fes = st.number_input("Max FEs (Budget)", min_value=1000, value=30000, step=1000)
    interval_fes = st.number_input("Interval FEs (Step Size)", min_value=100, value=1000, step=100)
    pop_size = st.number_input("Population Size", min_value=10, value=30, step=10)
    deterministic = st.checkbox("Deterministic Evaluation", value=True)
    
    run_btn = st.button("🚀 Run Evaluation", type="primary", disabled=(not uploaded_model or not uploaded_problem))


# Main Logic
if run_btn and uploaded_model and uploaded_problem:
    
    # 1. Setup
    with st.spinner("Initializing..."):
        # Save files
        model_path = save_uploaded_file(uploaded_model)
        problem_path = save_uploaded_file(uploaded_problem)
        
        # Load Model
        model, model_type = load_model_from_path(model_path)
        if not model:
            st.stop()
            
        # Load Problem
        try:
            problem = load_tsp_instance(problem_path)
        except Exception as e:
            st.error(f"Invalid TSP file: {e}")
            st.stop()
            
    # 2. Run RL Agent
    st.subheader(f"Analyzing {problem.name} ({problem.size} cities)...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"🤖 Running **RL Agent ({model_type})**...")
        rl_results = evaluate_rl_agent(model, model_type, problem, max_fes, interval_fes, pop_size, deterministic)
        st.success("RL Agent Finished!")
        
    # 3. Run Baselines
    with col2:
        st.info("🏃 Running **Baselines** (GA, TS, SA, ILS)...")
        baseline_results = evaluate_baselines(problem, max_fes, interval_fes)
        st.success("Baselines Finished!")
        
    # 4. Display Results
    st.divider()
    
    # --- KPIs ---
    best_baseline_cost = min([r['final_cost'] for r in baseline_results.values()])
    rl_cost = rl_results['final_cost']
    gap = ((rl_cost - best_baseline_cost) / best_baseline_cost) * 100
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("RL Agent Cost", f"{rl_cost:.2f}", delta=f"{gap:+.2f}% vs Best Baseline", delta_color="inverse")
    kpi2.metric("Best Baseline Cost", f"{best_baseline_cost:.2f}")
    kpi3.metric("RL Steps / Switches", f"{rl_results['steps']} / {rl_results['switches']}")
    kpi4.metric("RL Time", f"{rl_results['time']:.2f}s")
    
    # --- Convergence Plot ---
    st.subheader("📈 Convergence Comparison")
    
    # Combine data for plotting
    all_history = []
    all_history.extend(rl_results['history'])
    for name, res in baseline_results.items():
        all_history.extend(res['history'])
        
    df_chart = pd.DataFrame(all_history)
    
    chart = alt.Chart(df_chart).mark_line().encode(
        x=alt.X('fes', title='Function Evaluations'),
        y=alt.Y('cost', title='Cost (Tour Length)'),
        color='algo',
        tooltip=['algo', 'fes', 'cost']
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    # --- Action Distribution ---
    st.subheader("🤖 RL Agent Behavior")
    
    col_behavior1, col_behavior2 = st.columns([1, 2])
    
    with col_behavior1:
        # Pie/Bar of Action Distribution
        action_counts = pd.Series(rl_results['action_names']).value_counts().reset_index()
        action_counts.columns = ['Algorithm', 'Count']
        
        st.write("Algorithm Selection Frequency")
        bar_chart = alt.Chart(action_counts).mark_bar().encode(
            x='Count',
            y=alt.Y('Algorithm', sort='-x'),
            color='Algorithm'
        )
        st.altair_chart(bar_chart, use_container_width=True)
        
    with col_behavior2:
        # Step-by-step strip
        st.write("Step-by-Step Decision Sequence")
        
        # Create a dataframe for the timeline
        df_seq = pd.DataFrame({
            'Step': range(len(rl_results['action_names'])),
            'Algorithm': rl_results['action_names']
        })
        
        timeline = alt.Chart(df_seq).mark_rect().encode(
            x='Step',
            color='Algorithm',
            tooltip=['Step', 'Algorithm']
        ).properties(height=50)
        
        st.altair_chart(timeline, use_container_width=True)
    
    # --- Visualization ---
    st.subheader("🗺️ Best Tour Visualization")
    
    # We choose the best tour found by RL or Best Baseline
    
    viz_col1, viz_col2 = st.columns(2)
    
    def plot_tour(tour, title):
        if hasattr(problem, 'cities'):
            city_coords = problem.cities[tour]
            # Close the loop
            city_coords = np.vstack([city_coords, city_coords[0]])
            
            df_tour = pd.DataFrame(city_coords, columns=['x', 'y'])
            df_cities = pd.DataFrame(problem.cities, columns=['x', 'y'])
            
            # City points
            points = alt.Chart(df_cities).mark_circle(size=60, color='gray').encode(x='x', y='y')
            # Tour path
            path = alt.Chart(df_tour).mark_line(color='blue').encode(x='x', y='y')
            
            st.altair_chart((points + path).properties(title=title), use_container_width=True)
        else:
            st.warning("No city coordinates available for visualization.")

    with viz_col1:
        plot_tour(rl_results['best_tour'], f"RL Agent Tour (Cost: {rl_results['final_cost']:.2f})")
        
    with viz_col2:
        # Find best baseline
        best_name = min(baseline_results, key=lambda k: baseline_results[k]['final_cost'])
        best_res = baseline_results[best_name]
        plot_tour(best_res['best_tour'], f"Best Baseline ({best_name}) Tour (Cost: {best_res['final_cost']:.2f})")
        
    # Cleanup temps
    os.remove(model_path)
    os.remove(problem_path)

else:
    # Placeholder / Intro
    if not uploaded_model:
        st.info("👈 Please upload a trained **RL Model** (.zip) to begin.")
    elif not uploaded_problem:
        st.info("👈 Please upload a **TSP Problem** (.tsp) to begin.")

