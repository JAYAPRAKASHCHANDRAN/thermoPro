import streamlit as st
import os  # For handling directories
import pandas as pd
import plotly.express as px
import json  # For saving and loading configurations

# Custom CSS for styling, including sidebar width
st.markdown(
    """   
    <style>
    /* Increase sidebar width */
    [data-testid="stSidebar"] {
        min-width: 350px; /* Adjust this value for desired width */
        max-width: 350px; /* Adjust this value for desired width */
    }

    /* Background styling */
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
    }
    .header-title {
        font-family: Arial, sans-serif;
        font-size: 30px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar-section {
        background-color: #E8F8F5;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .stButton > button {
        background-color: #2E86C1;
        color: white;
        border-radius: 5px;
        padding: 10px;
        border: none;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #117864;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Set this flag to toggle between Admin and User Mode
is_admin = st.sidebar.radio("Select Mode", options=["Admin", "User"], index=1) == "Admin"

# Directory to store the user manual
manuals_dir = os.path.abspath("manuals")
os.makedirs(manuals_dir, exist_ok=True)

# Path to the manual file
manual_file_path = os.path.join(manuals_dir, "user_manual.pdf")

# Function for Admin Mode
def admin_mode():
    """Admin mode to upload the user manual."""
    st.sidebar.markdown('<div class="sidebar-section"><h3>Admin: Upload User Manual</h3></div>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload User Manual (PDF)", type=["pdf"], key="admin_upload")
    if uploaded_file:
        try:
            # Save the uploaded file
            with open(manual_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"'{uploaded_file.name}' uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Failed to upload the manual: {e}")

# Function for User Mode
def user_mode():
    """User mode to download the user manual."""
    st.sidebar.markdown(
        """
        <style>
        .centered-heading {
            text-align: center;
            font-weight: bold;
            font-size: 18px; /* Adjust font size as needed */
            margin-bottom: 50px;
        }
        </style>
        <div class="sidebar-section">
            <h3 class="centered-heading">Download User Manual</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    if os.path.exists(manual_file_path):
        # Provide the download button for the manual
        with open(manual_file_path, "rb") as file:
            st.sidebar.download_button(
                label="Download User Manual",
                data=file,
                file_name="user_manual.pdf",
                mime="application/pdf",
                key="user_download"
            )
    else:
        st.sidebar.warning("No user manual is available for download.")

# Helper Functions
def create_material_group():
    """Allow the user to dynamically create or modify material groups for comparison."""
    st.sidebar.markdown(
        """
        <style>
        .centered-heading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 50px; /* Adjust height as needed */
            font-weight: bold;
            font-size: 18px; /* Adjust font size as needed */
            text-align: center;
        }
        .sidebar-section {
            background-color: #E8F8F5;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        </style>
        <div class="sidebar-section">
            <h3 class="centered-heading">Material Group Setup</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load Configuration
    uploaded_file = st.sidebar.file_uploader("Load Configuration (JSON, Excel, or CSV)", type=["json", "xlsx", "csv"])

    material_group = []
    if uploaded_file:
        if uploaded_file.name.endswith(".json"):
            saved_config = json.load(uploaded_file)
            material_group = saved_config.get("materials", [])
            st.sidebar.success("JSON configuration loaded successfully!")
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
            st.sidebar.success("Excel configuration loaded successfully!")
            grouped = df.groupby("Material")
            for material_name, group in grouped:
                layers = [
                    {
                        "name": f"Layer {idx+1}",
                        "thickness": row["Layer Thickness (m)"],
                        "conductivity": row["Thermal Conductivity (W/m·K)"],
                    }
                    for idx, (_, row) in enumerate(group.iterrows())
                ]
                material_group.append({"name": material_name, "layers": layers})
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV configuration loaded successfully!")
            grouped = df.groupby("Material")
            for material_name, group in grouped:
                layers = [
                    {
                        "name": f"Layer {idx+1}",
                        "thickness": row["Layer Thickness (m)"],
                        "conductivity": row["Thermal Conductivity (W/m·K)"],
                    }
                    for idx, (_, row) in enumerate(group.iterrows())
                ]
                material_group.append({"name": material_name, "layers": layers})

    num_materials = st.sidebar.number_input(
        "Number of Materials to Define:", min_value=1, step=1, value=len(material_group) or 1
    )
    updated_material_group = []

    for i in range(num_materials):
        if i < len(material_group):
            material_name = st.sidebar.text_input(
                f"Name of Material {i + 1}:", material_group[i]["name"]
            )
            num_layers = st.sidebar.number_input(
                f"Number of Layers in {material_name}:",
                min_value=1,
                step=1,
                value=len(material_group[i]["layers"]),
            )
            layers = []
            for j in range(num_layers):
                if j < len(material_group[i]["layers"]):
                    layer_name = st.sidebar.text_input(
                        f"Layer {j + 1} Name ({material_name}):",
                        value=material_group[i]["layers"][j].get("name", f"Layer {j+1}"),
                    )
                    layer_thickness = st.sidebar.number_input(
                        f"{layer_name} Thickness (m) ({material_name}):",
                        min_value=0.01,
                        max_value=0.5,
                        value=material_group[i]["layers"][j]["thickness"],
                        step=0.01,
                    )
                    layer_conductivity = st.sidebar.number_input(
                        f"{layer_name} Conductivity (W/m·K) ({material_name}):",
                        min_value=0.01,
                        max_value=5.00,
                        value=material_group[i]["layers"][j]["conductivity"],
                        step=0.01,
                    )
                else:
                    layer_name = st.sidebar.text_input(
                        f"Layer {j + 1} Name ({material_name}):", value=f"Layer {j+1}"
                    )
                    layer_thickness = st.sidebar.number_input(
                        f"{layer_name} Thickness (m) ({material_name}):",
                        min_value=0.01,
                        max_value=0.5,
                        value=0.05,
                        step=0.01,
                    )
                    layer_conductivity = st.sidebar.number_input(
                        f"{layer_name} Conductivity (W/m·K) ({material_name}):",
                        min_value=0.01,
                        max_value=5.00,
                        value=0.04,
                        step=0.01,
                    )
                layers.append(
                    {
                        "name": layer_name,
                        "thickness": layer_thickness,
                        "conductivity": layer_conductivity,
                    }
                )
        else:
            material_name = st.sidebar.text_input(f"Name of Material {i + 1}:", f"Material {i + 1}")
            num_layers = st.sidebar.number_input(
                f"Number of Layers in {material_name}:", min_value=1, step=1, value=1
            )
            layers = []
            for j in range(num_layers):
                layer_name = st.sidebar.text_input(
                    f"Layer {j + 1} Name ({material_name}):", value=f"Layer {j+1}"
                )
                layer_thickness = st.sidebar.number_input(
                    f"{layer_name} Thickness (m) ({material_name}):",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.05,
                    step=0.01,
                )
                layer_conductivity = st.sidebar.number_input(
                    f"{layer_name} Conductivity (W/m·K) ({material_name}):",
                    min_value=0.01,
                    max_value=5.00,
                    value=0.04,
                    step=0.01,
                )
                layers.append(
                    {
                        "name": layer_name,
                        "thickness": layer_thickness,
                        "conductivity": layer_conductivity,
                    }
                )
        updated_material_group.append({"name": material_name, "layers": layers})

    if st.sidebar.button("Save Configuration"):
        config_data = {"materials": updated_material_group}
        config_json = json.dumps(config_data, indent=4)
        st.sidebar.download_button(
            label="Download Updated Configuration (JSON)",
            data=config_json,
            file_name="updated_material_configuration.json",
            mime="application/json",
        )
        st.sidebar.success("Updated configuration saved successfully!")

    return updated_material_group

def calculate_heat_flux(layers, Delta_T, area):
    """Calculate heat flux and total flux for a material group."""
    R_total_per_unit_area = sum(layer["thickness"] / layer["conductivity"] for layer in layers)
    heat_flux_per_unit_area = Delta_T / R_total_per_unit_area
    total_flux = heat_flux_per_unit_area * area
    return R_total_per_unit_area, heat_flux_per_unit_area, total_flux

def calculate_annual_energy_cost(total_flux, energy_cost_per_kWh, operating_hours):
    """Calculate the annual energy cost for total flux."""
    energy_kWh = total_flux * operating_hours / 1000  # Convert W to kWh
    annual_cost = energy_kWh * energy_cost_per_kWh
    return annual_cost

def validate_temperature_drop(layers, Delta_T_total, heat_flux_per_unit_area):
    """Validate temperature drop across each layer and cumulative drop."""
    cumulative_temp_drop = 0
    tolerance = 0.01  # Small tolerance for rounding errors
    temp_drop_details = []
    for layer in layers:
        RSI_layer = layer["thickness"] / layer["conductivity"]
        temp_drop_k = heat_flux_per_unit_area * RSI_layer
        cumulative_temp_drop += temp_drop_k
        temp_drop_details.append({
            "Layer": layer["name"],
            "Temperature Drop (K)": round(temp_drop_k, 2),
            "Thermal Resistance (K·m²/W)": round(RSI_layer, 4)
        })

    # Check if cumulative temperature drop matches the input Delta T
    is_valid = abs(cumulative_temp_drop - Delta_T_total) < tolerance
    difference = abs(cumulative_temp_drop - Delta_T_total)

    return temp_drop_details, cumulative_temp_drop, is_valid, difference

def create_heat_flux_graph(data, temp_column):
    """Create a graph showing heat flux for selected materials."""
    fig = px.line(
        data,
        x=temp_column,
        y="Heat Flux (W/m²)",
        color="Material",
        title="Heat Flux Comparison Across Materials",
        labels={"Heat Flux (W/m²)": "Heat Flux (W/m²)", temp_column: "Temperature (K)"},
        markers=True,
        template="plotly_dark",
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

# Main App
st.markdown(
    '<div class="header-title">ThermoPro: Advanced Insulation Analysis & Energy Optimization Tool</div>',
    unsafe_allow_html=True,
)

# Render the correct mode in the sidebar
if is_admin:
    admin_mode()
else:
    user_mode()

# Input Material Groups
material_group = create_material_group()

# Sidebar: Material Selection
st.sidebar.markdown(
    """
    <style>
    .centered-heading {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 50px; /* Adjust height as needed */
        font-weight: bold;
        font-size: 18px; /* Adjust font size as needed */
        text-align: center;
    }
    .sidebar-section {
        background-color: #E8F8F5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    <div class="sidebar-section">
        <h3 class="centered-heading">Material Comparison</h3>
    </div>
    """,
    unsafe_allow_html=True
)

baseline_material = st.sidebar.selectbox(
    "Select Baseline Material:", [material["name"] for material in material_group]
)
selected_comparison_materials = st.sidebar.multiselect(
    "Select Materials to Compare with Baseline:",
    [material["name"] for material in material_group if material["name"] != baseline_material],
)

# Sidebar: Simulation Parameters
st.sidebar.markdown(
    """
    <style>
    .centered-heading {
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    <div class="sidebar-section centered-heading">
        <h3>Simulation Parameters</h3>
    </div>
    """,
    unsafe_allow_html=True
)
fixed_temp_option = st.sidebar.radio("Fix Temperature At:", ["Interior", "Exterior"])

if fixed_temp_option == "Interior":
    fixed_temp = st.sidebar.slider("Interior Temperature (K):", min_value=200, max_value=400, value=300)
    temp_range = st.sidebar.slider(
        "Exterior Temperature Range (K):", min_value=200, max_value=400, value=(250, 300)
    )
    actual_temp_values = [ext_temp for ext_temp in range(temp_range[0], temp_range[1] + 1)]
else:
    fixed_temp = st.sidebar.slider("Exterior Temperature (K):", min_value=200, max_value=400, value=300)
    temp_range = st.sidebar.slider(
        "Interior Temperature Range (K):", min_value=200, max_value=400, value=(300, 350)
    )
    actual_temp_values = [int_temp for int_temp in range(temp_range[0], temp_range[1] + 1)]

area = st.sidebar.number_input("Wall Area (m²):", min_value=1.0, value=10.0, step=0.1)
energy_cost_per_kWh = st.sidebar.number_input(
    "Energy Cost ($/kWh):", min_value=0.01, value=0.12, step=0.01
)
operating_hours = st.sidebar.number_input("Operating Hours per Year:", min_value=1, value=8760, step=1)

# Generate Data
simulation_data = []

temp_column_name = "Temperature (K)"

for material in material_group:
    if material["name"] in [baseline_material] + selected_comparison_materials:
        for actual_temp in actual_temp_values:
            Delta_T = abs(fixed_temp - actual_temp)

            # Calculate heat flux and total flux
            (
                R_total_per_unit_area,
                heat_flux_per_unit_area,
                total_flux,
            ) = calculate_heat_flux(material["layers"], Delta_T, area)
            annual_cost = calculate_annual_energy_cost(total_flux, energy_cost_per_kWh, operating_hours)

            # Validate temperature drop across layers
            (
                temp_drop_details,
                cumulative_temp_drop,
                is_valid,
                difference,
            ) = validate_temperature_drop(material["layers"], Delta_T, heat_flux_per_unit_area)

            # Append the results, including validation status
            simulation_data.append(
                {
                    "Material": material["name"],
                    temp_column_name: actual_temp,
                    "Heat Flux (W/m²)": heat_flux_per_unit_area,
                    "Total Heat Flux (W)": total_flux,
                    "Effective Thermal Resistance (K·m²/W)": R_total_per_unit_area,
                    "Annual Energy Cost ($)": annual_cost,
                    "Validation Passed": is_valid,
                    "Temp Drop Difference (K)": difference,
                }
            )

            # Check validation status
            if not is_valid:
                # Display a warning if validation fails
                st.warning(
                    f"Validation Failed for Material '{material['name']}' at {actual_temp} K.\n"
                    f"- Cumulative Temperature Drop: {cumulative_temp_drop:.2f} K\n"
                    f"- Expected ΔT: {Delta_T:.2f} K\n"
                    f"- Difference: {difference:.2f} K\n\n"
                    "Suggestions for Improvement:\n"
                    "1. Increase layer thickness for higher resistance.\n"
                    "2. Use materials with lower thermal conductivity.\n"
                    "3. Verify input values for accuracy."
                )

                # Display layer-wise temperature drop details
                st.write("Layer-wise Temperature Drop Details:")
                for layer in temp_drop_details:
                    st.write(
                        f"- {layer['name']}: {layer['temperature_drop']:.2f} K "
                        f"(Thermal Resistance: {layer['thermal_resistance']:.4f} K·m²/W)"
                    )

# Convert data to DataFrame
simulation_df = pd.DataFrame(simulation_data)

if not simulation_df.empty:
    # Identify Most and Least Effective Materials
    most_effective = simulation_df.loc[simulation_df["Heat Flux (W/m²)"].idxmin()]
    least_effective = simulation_df.loc[simulation_df["Heat Flux (W/m²)"].idxmax()]

    simulation_df["Effectiveness"] = simulation_df["Material"].apply(
        lambda x: "Most Effective"
        if x == most_effective["Material"]
        else ("Least Effective" if x == least_effective["Material"] else "Comparison")
    )

    # Display Results
    st.subheader("Heat Flux Comparison Graph")
    selected_graph_materials = st.multiselect(
        "Select Materials to Visualize in the Graph:",
        simulation_df["Material"].unique(),
        default=simulation_df["Material"].unique(),
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
    st.write(
        f"✅ **The Most Effective Material**: {most_effective['Material']} with the least heat flux of {most_effective['Heat Flux (W/m²)']:.2f} W/m²."
    )
    st.write(
        f"❌ **The Least Effective Material**: {least_effective['Material']} with the highest heat flux of {least_effective['Heat Flux (W/m²)']:.2f} W/m²."
    )

# Display Heat Flux and Layer Reactions
st.subheader("Each Layer Reactions")

# Allow user to select material and adjust internal temperature
selected_material_for_analysis = st.selectbox(
    "Select a material to view its heat flux behavior:",
    simulation_df["Material"].unique(),
)

adjusted_temp = st.number_input(
    "Adjust the internal temperature (K):",
    min_value=float(200.0),  # Cast to float
    max_value=float(fixed_temp - 1),  # Cast to float
    value=float(actual_temp_values[0]),  # Cast to float
    step=0.1  # Already a float
)

# Filter data based on user selection
analysis_df = simulation_df[
    (simulation_df["Material"] == selected_material_for_analysis)
    & (simulation_df[temp_column_name] == adjusted_temp)
]

if not analysis_df.empty:
    index = analysis_df.index[0]
    material = next(
        (m for m in material_group if m["name"] == selected_material_for_analysis), None
    )
    if material:
        Delta_T_total = abs(fixed_temp - adjusted_temp)
        cumulative_temp_drop = 0
        tolerance = 0.01  # Tolerance for cumulative drop
        layer_reactions = []

        st.write(f"**External Temperature (Fixed):** {fixed_temp} K")
        st.write(f"**Adjusted Internal Temperature:** {adjusted_temp} K")
        st.write(f"**Total Temperature Drop Across Layers:** {Delta_T_total:.2f} K")

# Calculate layer reactions and validate cumulative drop
st.write("Layer-wise Temperature Drop (in Kelvin):")

layer_reactions = []  # Initialize list to store layer data
cumulative_temp_drop = 0  # Initialize cumulative temperature drop
total_RSI = sum([layer["thickness"] / layer["conductivity"] for layer in material["layers"]])
heat_flux_per_unit_area = Delta_T_total / total_RSI  # Ensure consistent heat flux calculation

for idx, layer in enumerate(material["layers"]):
    # Calculate layer-specific thermal resistance and temperature drop
    RSI_layer = layer["thickness"] / layer["conductivity"]
    temp_drop_k = heat_flux_per_unit_area * RSI_layer
    cumulative_temp_drop += temp_drop_k

    # Append results for the current layer
    layer_reactions.append({
        "S.No": idx + 1,
        "Layer": layer["name"],
        "Temperature Drop (K)": temp_drop_k,
        "Thermal Resistance (K·m²/W)": RSI_layer,
    })

    # Display layer details
    st.write(
        f"{layer['name']}: {temp_drop_k:.2f} K "
        f"(Thermal Resistance: {RSI_layer:.4f} K·m²/W)"
    )

# Tabular display of results
reaction_df = pd.DataFrame(layer_reactions)
st.table(reaction_df)

# Check cumulative temperature drop
if abs(cumulative_temp_drop - Delta_T_total) < tolerance:
    st.success(f"Cumulative Temperature Drop matches the specified ΔT ({Delta_T_total:.2f} K).")
else:
    difference = abs(cumulative_temp_drop - Delta_T_total)
    st.warning(
        f"Cumulative Temperature Drop does not match ΔT. Difference: {difference:.2f} K."
    )
    st.write("Suggested Adjustments:")
    st.write("- Increase layer thickness for better insulation.")
    st.write("- Use materials with lower thermal conductivity for more resistance.")
    st.write("- Ensure accurate input values for each layer.")

# Add download option for layer reactions
csv_data = reaction_df.to_csv(index=False)
st.download_button(
    label="Download Layer Reactions as CSV",
    data=csv_data,
    file_name="layer_reactions.csv",
    mime="text/csv",
)

# Download Results
csv = simulation_df.to_csv(index=False)
st.download_button(
    label="Download Results as CSV",
    data=csv,
    file_name="thermal_analysis_results.csv",
    mime="text/csv",
)

# Footer Section
st.sidebar.markdown(
    """
    <div style="text-align:center; background-color:#E8F8F5; padding:10px; border-radius:10px; margin-top:20px;">
        <b>Developed by Jayaprakash Chandran</b><br>
        <span style="font-size:12px; color:#117864;">© 2024 | All Rights Reserved</span>
    </div>
    """,
    unsafe_allow_html=True
)

#streamlit run "C:\Users\G S JAYAPRAKASH29\OneDrive - Syracuse University\2024 - R2 Resume\Caliber\thermoPro\V2.py" --server.runOnSave=false