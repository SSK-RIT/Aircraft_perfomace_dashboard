import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Aircraft Performance Analysis Dashboard",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Interactive Aircraft Performance Analysis Dashboard")
st.markdown("This application calculates and visualizes flight performance parameters (Thrust, Power, Rate of Climb, V-n Envelope, Range, Endurance, Glide, and Takeoff) for custom altitudes spanning from sea level up to 5 km and beyond.")

# --- Sidebar Inputs ---
st.sidebar.header("🎛️ Aircraft & Mission Parameters")

# Weights & Geometry
W = st.sidebar.number_input("Aircraft Weight / Mass (W in N)", value=103047.0, step=1000.0, format="%.1f")
S = st.sidebar.number_input("Wing Reference Area (S in m²)", value=47.0, step=1.0, format="%.1f")
AR = st.sidebar.number_input("Aspect Ratio (AR)", value=6.5, step=0.1, format="%.2f")
e = st.sidebar.number_input("Oswald Efficiency Factor (e)", value=0.87, step=0.01, format="%.2f")
CD0 = st.sidebar.number_input("Zero-Lift Drag Coefficient (CD0)", value=0.032, step=0.001, format="%.3f")

# Propulsion & Fuel
thrust_per_engine = st.sidebar.number_input("Thrust per Engine (T0 in N)", value=40298.0, step=500.0, format="%.1f")
num_engines = st.sidebar.number_input("Number of Engines", value=2, step=1)
fuel_gallons = st.sidebar.number_input("Fuel Capacity (US Gallons)", value=1900.0, step=50.0, format="%.1f")

# Altitudes
st.sidebar.subheader("🌍 Altitude & Atmospheric Settings")
alt_min = st.sidebar.number_input("Minimum Altitude (m)", value=0.0, step=500.0)
alt_max = st.sidebar.number_input("Maximum Altitude (m)", value=5000.0, step=500.0)
alt_step_val = st.sidebar.number_input("Altitude Step for Analysis (m)", value=1000.0, step=500.0)
selected_alt = st.sidebar.slider("Target Altitude for Detailed Plots (m)", min_value=float(alt_min), max_value=float(alt_max), value=2000.0, step=500.0)

# Velocity Array Configuration
st.sidebar.subheader("🚀 Velocity Scan Settings")
v_start = st.sidebar.number_input("Velocity Start (m/s)", value=10.0, step=5.0)
v_end = st.sidebar.number_input("Velocity End (m/s)", value=500.0, step=10.0)
v_step = st.sidebar.number_input("Velocity Step (m/s)", value=5.0, step=1.0)

# --- Derived Aerodynamic Constants ---
k = 1.0 / (np.pi * e * AR)
TA0 = thrust_per_engine * num_engines

# Fuel Weight Conversion
fuel_density_kg_L = 0.8037
liters_per_gal = 3.78541
g_acc = 9.80665
W_fuel = fuel_gallons * liters_per_gal * fuel_density_kg_L * g_acc
W_final = W - W_fuel
weight_ratio = W / W_final if W_final > 0 else 1.0
ct = 1.0 / 3600.0

# ISA Atmosphere Model (Troposphere up to 11km)
def get_isa_density(alt_m):
    rho0 = 1.225
    T0 = 288.15
    if alt_m <= 11000:
        T = T0 - 0.0065 * alt_m
        rho = rho0 * (T / T0)**4.25588
    else:
        rho = 0.3639
    return rho

# --- Calculations across Altitude Range ---
altitudes = np.arange(alt_min, alt_max + 1, alt_step_val)
if len(altitudes) == 0 or altitudes[-1] != alt_max:
    altitudes = np.append(altitudes, alt_max)

alt_results = []
for alt in altitudes:
    rho = get_isa_density(alt)
    Ta_alt = TA0 * (rho / 1.225)
    
    Vs = np.sqrt((2 * W) / (rho * S * 2.8))
    
    v_scan = np.linspace(10, 600, 5000)
    Tr_scan = 0.5 * rho * v_scan**2 * S * CD0 + (2 * k * W**2) / (rho * v_scan**2 * S)
    valid_v = v_scan[Tr_scan <= Ta_alt]
    v_max = valid_v[-1] if len(valid_v) > 0 else 0.0
    
    Pr_scan = Tr_scan * v_scan
    Pa_scan = Ta_alt * v_scan
    roc_scan = (Pa_scan - Pr_scan) / W
    max_roc = np.max(roc_scan)
    
    alt_results.append({
        "Altitude (m)": alt,
        "Density (kg/m³)": rho,
        "Ta (N)": Ta_alt,
        "Vs (m/s)": Vs,
        "Vmax (m/s)": v_max,
        "Max ROC (m/s)": max_roc
    })

# --- Main Dashboard Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Thrust & Power Curves", 
    "📈 Rate of Climb", 
    "🛩️ V-n Flight Envelope", 
    "⛽ Range & Endurance", 
    "🛫 Takeoff & Glide"
])

# Target Altitude Parameters for Detailed Plots
rho_target = get_isa_density(selected_alt)
Ta_target = TA0 * (rho_target / 1.225)
valpha = np.arange(v_start, v_end + v_step, v_step)

Tr_target = 0.5 * rho_target * valpha**2 * S * CD0 + (2 * k * W**2) / (rho_target * valpha**2 * S)
Ta_target_arr = np.full_like(valpha, Ta_target, dtype=float)
Pr_target = Tr_target * valpha
Pa_target = Ta_target_arr * valpha

with tab1:
    st.subheader(f"Thrust Required (Tr) & Available (Ta) at Selected Altitude: {selected_alt} m")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(valpha, Tr_target, label="Thrust Required (Tr)", color="blue", linewidth=2)
    ax.plot(valpha, Ta_target_arr, label=f"Thrust Available (Ta = {Ta_target:.1f} N)", color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("True Airspeed V (m/s)")
    ax.set_ylabel("Thrust (N)")
    ax.set_title(f"Altitude Analysis: Tr vs Ta at Density rho = {rho_target:.4f} kg/m³")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("Power Required (Pr) & Available (Pa) (MW)")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(valpha, Pr_target / 1e6, label="Power Required (Pr)", color="darkgreen", linewidth=2)
    ax2.plot(valpha, Pa_target / 1e6, label="Power Available (Pa)", color="orange", linestyle="--", linewidth=2)
    ax2.set_xlabel("True Airspeed V (m/s)")
    ax2.set_ylabel("Power (MW)")
    ax2.set_title(f"Power Curves at Altitude {selected_alt} m")
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend()
    st.pyplot(fig2)

with tab2:
    st.subheader(f"Rate of Climb (ROC) vs Velocity at {selected_alt} m")
    roc_target = (Pa_target - Pr_target) / W
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(valpha, roc_target, label="Rate of Climb (ROC)", color="purple", linewidth=2)
    ax3.axhline(0, color="black", linestyle="--", alpha=0.6, label="Level Flight (ROC = 0)")
    
    max_roc_idx = np.argmax(roc_target)
    ax3.scatter([valpha[max_roc_idx]], [roc_target[max_roc_idx]], color="red", s=80, zorder=5,
                label=f"Max ROC: {roc_target[max_roc_idx]:.2f} m/s at {valpha[max_roc_idx]:.1f} m/s")
    
    ax3.set_xlabel("True Airspeed V (m/s)")
    ax3.set_ylabel("Rate of Climb (m/s)")
    ax3.set_title(f"ROC Performance Curve ({selected_alt} m)")
    ax3.grid(True, linestyle=":", alpha=0.7)
    ax3.legend()
    st.pyplot(fig3)

with tab3:
    st.subheader(f"V-n Flight Envelope Diagram at {selected_alt} m")
    n_max = 5.0
    n_min = -3.0
    CL_max_up = 2.8
    CL_max_down = 1.2
    VD = 295.14
    
    Vs_pos = np.sqrt((2 * W) / (rho_target * S * CL_max_up))
    Vs_neg = np.sqrt((2 * W) / (rho_target * S * CL_max_down))
    VA_pos = Vs_pos * np.sqrt(n_max)
    
    v_stall = np.linspace(0, Vs_pos, 200)
    n_stall = 0.5 * rho_target * v_stall**2 * S * CL_max_up / W
    v_stall_neg = np.linspace(0, Vs_neg, 200)
    n_stall_neg = -0.5 * rho_target * v_stall_neg**2 * S * CL_max_down / W
    
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(v_stall, n_stall, color="teal", linestyle="--")
    ax4.plot(v_stall_neg, n_stall_neg, color="teal", linestyle="--")
    ax4.plot([Vs_pos, VA_pos], [n_stall[-1], n_max], color="teal", linewidth=2, label="Positive Operational Limit")
    ax4.plot([VA_pos, VD], [n_max, n_max], color="teal", linewidth=2)
    ax4.plot([VD, VD], [n_max, 0], color="teal", linewidth=2)
    ax4.plot([Vs_neg, VD], [n_stall_neg[-1], n_min], color="darkred", linewidth=2, label="Negative Operational Limit")
    ax4.plot([VD, VD], [n_min, 0], color="darkred", linewidth=2)
    
    ax4.axhline(0, color='black', linewidth=0.8)
    ax4.axvline(0, color='black', linewidth=0.8)
    ax4.set_xlabel("True Airspeed V (m/s)")
    ax4.set_ylabel("Load Factor n")
    ax4.set_title(f"V-n Flight Envelope ({selected_alt} m)")
    ax4.grid(True, linestyle=":", alpha=0.7)
    ax4.legend()
    st.pyplot(fig4)

with tab4:
    st.subheader("Fuel, Range & Endurance Performance (Breguet Analysis)")
    
    LD_max = 1.0 / (2.0 * np.sqrt(CD0 * k))
    CL_range = np.sqrt(CD0 / (3.0 * k))
    CD_range = CD0 + k * CL_range**2
    LD_range = CL_range / CD_range
    
    E_hours = (1.0 / ct) * LD_max * np.log(weight_ratio) / 3600.0
    
    factor_R = (2.0 / ct) * np.sqrt(2.0 / (rho_target * S)) * (np.sqrt(CL_range) / CD_range) * (np.sqrt(W) - np.sqrt(W_final))
    range_km = factor_R / 1000.0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Fuel Weight", f"{W_fuel:,.1f} N", f"{fuel_gallons} US gal")
    col2.metric("Max Endurance (E)", f"{E_hours:.2f} hours")
    col3.metric(f"Max Range (R at {selected_alt} m)", f"{range_km:.1f} km")
    
    v_cruise = np.linspace(38, 250, 300)
    CL_c = (2 * W) / (rho_target * v_cruise**2 * S)
    CD_c = CD0 + k * CL_c**2
    D_c = 0.5 * rho_target * v_cruise**2 * S * CD_c
    SR = (v_cruise / (ct * D_c)) / 1000.0 
    
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.plot(v_cruise, SR, color="royalblue", linewidth=2, label="Specific Range Curve")
    ax5.set_xlabel("Flight Velocity V (m/s)")
    ax5.set_ylabel("Specific Range (km / N)")
    ax5.set_title(f"Specific Range vs Velocity ({selected_alt} m)")
    ax5.grid(True, linestyle=":", alpha=0.7)
    ax5.legend()
    st.pyplot(fig5)

with tab5:
    st.subheader("Takeoff & Glide Performance")
    
    Cl_to = 2.8
    mu_friction = 0.04
    v_lof = 1.1 * np.sqrt((2 * W) / (1.225 * S * Cl_to))
    v_avg = 0.707 * v_lof
    cl_avg = (2 * W) / (1.225 * v_avg**2 * S)
    cd_avg = CD0 + k * cl_avg**2
    d_avg = 0.5 * 1.225 * v_avg**2 * S * cd_avg
    l_avg = 0.5 * 1.225 * v_avg**2 * S * cl_avg
    accel = (9.81 / W) * (TA0 - d_avg - mu_friction * (W - l_avg))
    s_g = (v_lof**2) / (2 * accel) if accel > 0 else 0.0
    
    gamma_min_deg = np.degrees(np.arctan(1.0 / LD_max))
    best_glide_dist_5km = 5000.0 * LD_max
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("### 🛫 Takeoff Estimates (Sea Level)")
        st.write(f"- **Lift-Off Speed (V_LOF):** `{v_lof:.2f} m/s`")
        st.write(f"- **Ground Roll Distance (s_g):** `{s_g:.2f} m`")
        st.write(f"- **Total 15m Obstacle Clearance:** `{(s_g + 47.4):.2f} m`")
    with col_t2:
        st.markdown("### 🪂 Power-Off Glide Performance")
        st.write(f"- **Maximum L/D Ratio:** `{LD_max:.2f}`")
        st.write(f"- **Minimum Glide Angle (gamma_min):** `{gamma_min_deg:.2f}°`")
        st.write(f"- **Glide Range from 5 km Altitude:** `{best_glide_dist_5km/1000:.2f} km`")

st.markdown("---")
st.subheader("📊 Multi-Altitude Performance Summary Table (1 km to 5 km span)")
st.dataframe(alt_results, use_container_width=True)
