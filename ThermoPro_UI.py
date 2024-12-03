import streamlit as st
import pandas as pd
import plotly.express as px

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-image: url('https://wallpaperaccess.com/full/4299503.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 10px;
        overflow: hidden;
        height: 100vh;
        display: flex;
        flex-direction: column;
    }
    .header-title {
        font-family: Arial, sans-serif;
        font-size: 30px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 20px;
    }
    .legend-box {
        background-color: #f9f9f9;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        width: fit-content;
        margin: auto;
        text-align: left;
    }
    .scroll-sidebar {
        max-height: 100vh;
        overflow-y: auto;
    }
    .fixed-content {
        max-height: 90vh;
        overflow-y: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Helper Functions
def create_material_group():
    """Allow the user to dynamically create material groups for comparison."""
    st.sidebar.header("Material Group Setup")
    num_materials = st.sidebar.number_input("Number of Materials to Define:", min_value=1, step=1, value=1)
    material_group = []

    for i in range(num_materials):
        st.sidebar.subheader(f"Material {i + 1}")
        material_name = st.sidebar.text_input(f"Name of Material {i + 1}:", f"Material {i + 1}")
        num_layers = st.sidebar.number_input(f"Number of Layers in {material_name}:", min_value=1, step=1, value=1)
        layers = []

        for j in range(num_layers):
            layer_thickness = st.sidebar.number_input(
                f"Layer {j + 1} Thickness (m) ({material_name}):", min_value=0.01, max_value=0.5, value=0.05, step=0.01
            )
            layer_conductivity = st.sidebar.number_input(
                f"Layer {j + 1} Conductivity (W/m·K) ({material_name}):", min_value=0.01, max_value=5.00, value=0.04, step=0.01
            )
            layers.append({"thickness": layer_thickness, "conductivity": layer_conductivity})

        material_group.append({"name": material_name, "layers": layers})

    return material_group

def calculate_heat_flux(layers, Delta_T, area):
    """Calculate heat flux and total flux for a material group."""
    R_total_per_unit = sum(layer["thickness"] / layer["conductivity"] for layer in layers)
    heat_flux = Delta_T / R_total_per_unit
    total_flux = heat_flux * area
    return heat_flux, total_flux, R_total_per_unit

def calculate_annual_energy_cost(total_flux, energy_cost_per_kWh, operating_hours):
    """Calculate the annual energy cost for total flux."""
    energy_kWh = total_flux * operating_hours / 1000  # Convert W to kWh
    annual_cost = energy_kWh * energy_cost_per_kWh
    return annual_cost

def create_heat_flux_graph(data, temp_column):
    """Create a graph showing heat flux for selected materials."""
    fig = px.line(
        data,
        x=temp_column,
        y="Heat Flux (W/m²)",
        color="Material",
        title="Heat Flux Comparison Across Materials",
        labels={"Heat Flux (W/m²)": "Heat Flux (W/m²)", temp_column: "Temperature (K)"},
        markers=True
    )
    fig.update_layout(
        title=dict(
            text="Heat Flux across the listed materials",
            x=0.275,  # Centers the title
            font=dict(size=20),  # Adjust the font size if needed
            pad=dict(t=70)  # Add padding to bring the title down
        ),
        xaxis=dict(
            title=dict(
                text="Temperature (K)",
                font=dict(size=17, color="black", family="Calibri"),  # Bold and customized font
            ),
            tickfont=dict(size=14, color="black", family="calibri"),  # Bold tick labels
        ),
        yaxis=dict(
            title=dict(
                text="Heat Flux (W/m²)",
                font=dict(size=16, color="black", family="Arial"),  # Bold and customized font
            ),
            tickfont=dict(size=14, color="black", family="Calibri"),  # Bold tick labels
        ),
        legend=dict(
            title="Material",
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="Black",
            borderwidth=2.25
        ),
        margin=dict(l=50, r=50, t=100, b=60),  # Increase top margin for better spacing
        plot_bgcolor="white",  # Background inside the graph area
        paper_bgcolor="white"  # Background outside the graph
    )
    # Add a black border
    fig.update_layout(
        margin=dict(l=80, r=60, t=100, b=70),  # Space around the graph for the border
        paper_bgcolor="white",  # Ensure the graph area is white
        shapes=[
            dict(
                type="rect",
                x0=0, x1=1, y0=0, y1=1,
                xref="paper",
                yref="paper",
                line=dict(color="Grey", width=2.5)  # Black border with 4px width
            )
        ]
    )
    return fig

def highlight_row(row):
    """Highlight row colors based on effectiveness."""
    if row["Effectiveness"] == "Most Effective":
        return ["background-color: lightgreen"] * len(row)
    elif row["Effectiveness"] == "Least Effective":
        return ["background-color: lightcoral"] * len(row)
    elif row["Material"] == baseline_material:
        return ["background-color: lightblue"] * len(row)
    else:
        return [""] * len(row)

# Main Streamlit App
st.markdown('<div class="header-title">ThermoPro: Advanced Insulation Analysis & Energy Optimization Tool</div>', unsafe_allow_html=True)

# Input Material Groups
material_group = create_material_group()

# Sidebar: Material Selection for Comparison
st.sidebar.header("Material Comparison")
baseline_material = st.sidebar.selectbox(
    "Select Baseline Material:",
    [material["name"] for material in material_group]
)
selected_comparison_materials = st.sidebar.multiselect(
    "Select Materials to Compare with Baseline:",
    [material["name"] for material in material_group if material["name"] != baseline_material]
)

# Sidebar: Simulation Parameters
st.sidebar.header("Simulation Parameters")
fixed_temp_option = st.sidebar.radio("Fix Temperature At:", ["Interior", "Exterior"])

if fixed_temp_option == "Interior":
    fixed_temp = st.sidebar.slider("Interior Temperature (K):", min_value=200, max_value=400, value=300)
    temp_range = st.sidebar.slider("Exterior Temperature Range (K):", min_value=200, max_value=400, value=(250, 300))
    actual_temp_values = [ext_temp for ext_temp in range(temp_range[0], temp_range[1] + 1)]
else:
    fixed_temp = st.sidebar.slider("Exterior Temperature (K):", min_value=200, max_value=400, value=300)
    temp_range = st.sidebar.slider("Interior Temperature Range (K):", min_value=200, max_value=400, value=(300, 350))
    actual_temp_values = [int_temp for int_temp in range(temp_range[0], temp_range[1] + 1)]

area = st.sidebar.number_input("Wall Area (m²):", min_value=1.0, value=10.0, step=0.1)
energy_cost_per_kWh = st.sidebar.number_input("Energy Cost ($/kWh):", min_value=0.01, value=0.12, step=0.01)
operating_hours = st.sidebar.number_input("Operating Hours per Year:", min_value=1, value=8760, step=1)

# Generate Data
simulation_data = []

temp_column_name = "Temperature (K)" if fixed_temp_option == "Interior" else "Temperature (K)"

for material in material_group:
    if material["name"] in [baseline_material] + selected_comparison_materials:
        for actual_temp in actual_temp_values:
            Delta_T = abs(fixed_temp - actual_temp)
            heat_flux, total_flux, R_total_per_unit = calculate_heat_flux(material["layers"], Delta_T, area)
            annual_cost = calculate_annual_energy_cost(total_flux, energy_cost_per_kWh, operating_hours)
            simulation_data.append({
                "Material": material["name"],
                temp_column_name: actual_temp,
                "Heat Flux (W/m²)": heat_flux,
                "Total Heat Flux (W)": total_flux,
                "Effective Thermal Resistance (K·m²/W)": R_total_per_unit,
                "Annual Energy Cost ($)": annual_cost
            })

# Convert data to DataFrame
simulation_df = pd.DataFrame(simulation_data)

if not simulation_df.empty:
    # Identify Most and Least Effective Materials
    most_effective = simulation_df.loc[simulation_df["Heat Flux (W/m²)"].idxmin()]
    least_effective = simulation_df.loc[simulation_df["Heat Flux (W/m²)"].idxmax()]

    simulation_df["Effectiveness"] = simulation_df["Material"].apply(
        lambda x: "Most Effective" if x == most_effective["Material"] else
        ("Least Effective" if x == least_effective["Material"] else "Comparison")
    )

    # Display Results
    st.subheader("Heat Flux Comparison Graph")
    selected_graph_materials = st.multiselect(
        "Select Materials to Visualize in the Graph:",
        simulation_df["Material"].unique(),
        default=simulation_df["Material"].unique()
    )
    filtered_df = simulation_df[simulation_df["Material"].isin(selected_graph_materials)]
    if not filtered_df.empty:
        graph = create_heat_flux_graph(filtered_df, temp_column_name)
        st.plotly_chart(graph)

    # Display the Results Table
    st.subheader("Detailed Results Table")
    range_description = f"If {fixed_temp_option.lower()} temperature is {fixed_temp}K, the results are for the range between {min(actual_temp_values)}K and {max(actual_temp_values)}K."
    st.markdown(f"**{range_description}**")
    styled_table = simulation_df.style.apply(highlight_row, axis=1)
    st.dataframe(styled_table)

    # Display Insights
    st.subheader("Material Insights")
    st.write(f"🔹 **Baseline Material**: {baseline_material}")
    st.write(f"✅ **The Most Effective Material**: {most_effective['Material']} with the least heat flux of {most_effective['Heat Flux (W/m²)']:.2f} W/m².")
    st.write(f"❌ **The Least Effective Material**: {least_effective['Material']} with the highest heat flux of {least_effective['Heat Flux (W/m²)']:.2f} W/m².")

    # Download Results
    csv = simulation_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="thermal_analysis_results.csv",
        mime="text/csv"
    )
else:
    st.warning("No data to display. Please configure materials in the sidebar.")
