import streamlit as st
import os
import pandas as pd
from fpdf import FPDF
import io
import spacy
import plotly.express as px
import plotly.graph_objects as go # Added for 3D Mesh
import json
import plotly.io as pio
import traceback # For more detailed error logging if needed

# Ensure kaleido is installed for Plotly image export (for PDF report)
# You might need to run: pip install kaleido

# Custom CSS for styling, including sidebar width
st.markdown(
    """
    <style>
    /* Increase sidebar width */
    [data-testid="stSidebar"] > div:first-child { /* More specific selector */
        min-width: 350px; /* Adjust this value for desired width */
        max-width: 350px; /* Adjust this value for desired width */
    }

    /* Background styling - Applied to .stApp */
    .stApp {
        background-image: url(https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.shutterstock.com%2Fimage-photo%2Fminimal-white-dotted-paper-texture-600nw-2533715245.jpg&f=1&nofb=1&ipt=7063b8828a94d0d31bc7033d3bae7ae61200a28b2eeebdfc7859e0087c63323c);
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        /* Consider adding a fallback background color */
        /* background-color: #f0f2f6; */
    }

    /* Optional: Ensure content blocks have a background if needed over the image */
      .main .block-container {
          background-color: rgba(255, 255, 255, 0.92); /* White background for content */
          padding: 2rem;
          border-radius: 10px;
          margin: 20px; /* Add margin back to the content block */
          box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      }

    .header-title {
        font-family: Arial, sans-serif;
        font-size: 30px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 20px;
        padding-top: 20px; /* Add padding at the top */
    }
    .sidebar-section {
        background-color: #E8F8F5;
        padding: 15px; /* Increased padding */
        border-radius: 10px;
        margin-bottom: 15px; /* Increased margin */
        border: 1px solid #D0E0E3; /* Add a light border */
    }
      .sidebar-section h3 {
        text-align: center;
        color: #117864; /* Darker green for sidebar headers */
      }
    .stButton > button {
        background-color: #2E86C1;
        color: white;
        border-radius: 5px;
        padding: 10px;
        border: none;
        font-size: 16px;
        width: 100%; /* Make buttons full width in sidebar */
        margin-bottom: 5px; /* Space between buttons */
    }
    .stButton > button:hover {
        background-color: #117864;
        color: white;
    }
    /* Style for the results table */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
    }
    .dataframe th, .dataframe td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center; /* Center align cells */
    }
    .dataframe th {
        text-align: center; /* Center align header */
        background-color: #f2f2f2;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Directory to store the user manual
manuals_dir = os.path.abspath("manuals")
os.makedirs(manuals_dir, exist_ok=True)

# Path to the manual file - we will generate this on the fly now
manual_file_name = "user_manual.pdf"

# --- User Manual Generation Function ---
def generate_user_manual_pdf():
    """Generates a simple user manual PDF on the fly."""
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "ThermoPro User Manual", 0, 1, "C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "1. Introduction", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "ThermoPro is a tool for analyzing heat transfer through layered materials, comparing different insulation configurations, and optimizing energy performance.", border=0, align="J")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "2. Material Group Setup", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "Use the 'Material Group Setup' section in the sidebar to define your materials. You can add multiple materials, each with multiple layers. For each layer, specify its name, thickness (in meters), and thermal conductivity (in W/m·K). You can also load existing configurations from JSON, Excel (.xlsx), or CSV files.", border=0, align="J")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "3. Material Comparison Selection", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "Select one material as the 'Baseline Material' for comparison. Choose other materials from the 'Select Materials to Compare' list. Only the selected materials will be included in the simulation results and graphs.", border=0, align="J")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "4. Simulation Parameters", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "Select your preferred Temperature Unit (K, °C, °F). Define the temperature conditions by fixing either the interior or exterior temperature and setting a range for the other temperature. Specify the Wall Area (in m²), Energy Cost ($/kWh), and annual Operating Hours. You can also choose to use specific Convective Heat Transfer Coefficients (h-values) for more advanced boundary conditions.", border=0, align="J")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "5. Simulation Results", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "The results section displays a graph comparing heat flux for the selected materials across the temperature range. A detailed table shows calculated values (Heat Flux, Total Flux, Thermal Resistance, Annual Cost) for each material at each temperature point. Validation messages will alert you if temperature drop calculations across layers do not sum up correctly.", border=0, align="J")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "6. Detailed Layer Analysis", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "In this section, you can select a specific material and temperature point from your simulation results to view the temperature drop and thermal resistance for each individual layer within that material stack. A bar chart visualizes the temperature drop per layer.", border=0, align="J")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "7. 3D Material Visualization", 0, 1) # New Section
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "Visualize the layered structure of a selected material in 3D. This section provides a simple 3D representation of the material stack, showing each layer's thickness.", border=0, align="J")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "8. Reports", 0, 1) # Old 7 is now 8
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "Download comprehensive simulation results in Excel (.xlsx) or PDF format. The PDF report includes the main results table and the heat flux comparison graph.", border=0, align="J")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "9. Chatbot", 0, 1) # Old 8 is now 9
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "Use the 'Chat with ThermoBot' in the sidebar for quick answers to common questions about the app and heat transfer concepts.", border=0, align="J")
    pdf.ln(5)

    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer) 
    pdf_buffer.seek(0) 
    return pdf_buffer.getvalue()
 
# Add Download Manual button (at top level, called when script runs)
manual_pdf_buffer = generate_user_manual_pdf()
st.sidebar.download_button(
    label="Download User Manual",
    data=manual_pdf_buffer,
    file_name=manual_file_name,
    mime="application/pdf",
    key="user_download_main" # Changed key
)


# --- Chatbot Setup ---
# Check if spacy model is downloaded, provide instructions if not
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Spacy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm' in your terminal.")
    st.stop() # Stop execution if model is missing

# Chatbot response function using NLP
def chatbot_response(user_input_bot): # Renamed parameter
    doc = nlp(user_input_bot.lower())

    # Define keyword categories and responses
    keyword_responses = {
        "hello": "Hi there! How can I assist you today?",
        "hi": "Hello! How can I help?",
        "help": "I can assist you with defining materials, setting up simulations, interpreting results, generating reports, or viewing 3D models. What do you need help with?",
        "material": "You can upload or manually define material configurations in the sidebar. Supported formats are JSON, Excel, or CSV.",
        "temperature": "Temperature settings are adjusted in the sidebar under 'Simulation Parameters'. You can fix one temperature and set a range for the other. You can also choose K, °C, or °F.",
        "energy": "The app calculates annual energy cost based on the total heat flux, operating hours, and energy cost per kWh.",
        "simulation": "To run a simulation, define your materials, select baseline and comparison materials, and set the simulation parameters (temperatures, area, costs, boundary conditions). The app will automatically display results.",
        "project": "Great! Tell me about your project. I can help with material selection and performance analysis.",
        "report": "You can download simulation results as an Excel or PDF report from the sidebar after running the simulation.",
        "heat flux": "Heat flux is the rate of heat energy transfer per unit area. It's calculated based on the temperature difference and the material's thermal resistance.",
        "resistance": "Thermal Resistance (R-value) is a measure of a material's ability to resist heat flow. Higher resistance means less heat flow for a given temperature difference.",
        "conductivity": "Thermal Conductivity (k-value) is a material property indicating how well it conducts heat. Lower conductivity means better insulation (higher resistance per unit thickness).",
        "cost": "Annual energy cost is calculated from the total heat flux, operating hours, and your specified energy cost per kWh.",
        "convective": "You can choose to use fixed surface resistances or input specific convective heat transfer coefficients (h-values) for interior and exterior surfaces under 'Simulation Parameters'.",
        "3d": "You can visualize a selected material's layers in 3D in the '3D Material Visualization' section after defining materials.", # New
        "visualize": "You can visualize heat flux graphs, layer temperature drops, or a 3D model of material layers. Which visualization are you interested in?", # New
        "model": "The '3D Material Visualization' section shows a 3D model of the selected material's layers.", # New
        "stack": "The 3D visualization shows the material layers stacked on top of each other." # New
    }

    # Check if any keyword is present
    for token in doc:
        if token.lemma_ in keyword_responses:
            return keyword_responses[token.lemma_]

    # Basic fallback
    if any(word in user_input_bot.lower() for word in ["what is", "explain", "tell me about"]):
        return "I can explain concepts like heat flux, thermal resistance, or conductivity. What topic are you curious about?"
    if "how to" in user_input_bot.lower():
        return "I can guide you on how to use features like defining materials, loading configurations, or interpreting graphs. What task are you trying to perform?"

    return "I'm not sure I understand. Could you rephrase or ask about a specific feature like 'material', 'temperature', 'report', '3D model', or 'help'?"

# Streamlit chatbot UI in sidebar (at top level, runs when script runs)
st.sidebar.markdown('<div class="sidebar-section"><h3 class="centered-heading">Chat with ThermoBot 🤖</h3></div>', unsafe_allow_html=True)

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input_for_chatbot = st.sidebar.text_input("Type your message here:", key="chat_input_main_ui") # Changed key

if user_input_for_chatbot:
    response = chatbot_response(user_input_for_chatbot)
    st.session_state['chat_history'].append({"user": user_input_for_chatbot, "bot": response})

chat_display_limit = 10
for chat in reversed(st.session_state['chat_history'][-chat_display_limit:]):
    st.sidebar.markdown(f"**You:** {chat['user']}")
    st.sidebar.markdown(f"**ThermoBot:** {chat['bot']}")
    st.sidebar.markdown("---")


# --- Helper Functions ---
def convert_temperature_value(value, from_unit, to_unit):
    """Converts temperature between K, °C, °F."""
    if from_unit == to_unit:
        return float(value) # Ensure it's a float
    k_value = 0.0
    try:
        val_float = float(value)
    except ValueError:
        raise ValueError(f"Invalid temperature value for conversion: {value}")

    if from_unit == "°C": k_value = val_float + 273.15
    elif from_unit == "°F": k_value = (val_float - 32) * 5/9 + 273.15
    elif from_unit == "K": k_value = val_float
    else: raise ValueError(f"Invalid 'from_unit' for temperature: {from_unit}")

    if to_unit == "°C": return k_value - 273.15
    elif to_unit == "°F": return (k_value - 273.15) * 9/5 + 32
    elif to_unit == "K": return k_value
    else: raise ValueError(f"Invalid 'to_unit' for temperature: {to_unit}")


def create_material_group():
    st.sidebar.markdown("""<div class="sidebar-section"><h3 class="centered-heading">Material Group Setup</h3></div>""", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Load Configuration (JSON, Excel, or CSV)", type=["json", "xlsx", "csv"], key="material_uploader")
    initial_material_group = []
    load_error = False
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".json"):
                saved_config = json.load(uploaded_file)
                if "materials" in saved_config and isinstance(saved_config["materials"], list):
                    is_valid_json = True
                    for mat in saved_config["materials"]:
                        if not isinstance(mat, dict) or "name" not in mat or "layers" not in mat or not isinstance(mat["layers"], list):
                            is_valid_json = False; break
                        for layer in mat["layers"]:
                            if not isinstance(layer, dict) or not all(k in layer for k in ["name", "thickness", "conductivity"]):
                                is_valid_json = False; break
                        if not is_valid_json: break
                    if is_valid_json:
                        initial_material_group = saved_config["materials"]
                        st.sidebar.success("JSON configuration loaded successfully!")
                    else:
                        st.sidebar.error("Invalid JSON structure."); load_error = True
                else:
                    st.sidebar.error("Invalid JSON format."); load_error = True
            elif uploaded_file.name.endswith((".xlsx", ".csv")):
                df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
                required_cols = ["Material", "Layer Thickness (m)", "Thermal Conductivity (W/m·K)"]
                if not all(col in df.columns for col in required_cols):
                    st.sidebar.error(f"Missing required columns. Need: {', '.join(required_cols)}"); load_error = True
                else:
                    df_process = df[required_cols + ["Layer Name"] if "Layer Name" in df.columns else required_cols].copy()
                    if df_process[required_cols].isnull().any().any():
                        st.sidebar.error(f"Missing values in required columns. Please fill them."); load_error = True
                    else:
                        numeric_cols = ["Layer Thickness (m)", "Thermal Conductivity (W/m·K)"]
                        df_process[numeric_cols] = df_process[numeric_cols].apply(pd.to_numeric, errors='coerce')
                        if df_process[numeric_cols].isnull().any().any(): st.sidebar.error("Non-numeric values in Thickness/Conductivity."); load_error = True
                        elif (df_process["Layer Thickness (m)"] <= 0).any(): st.sidebar.error("Thickness must be > 0."); load_error = True
                        elif (df_process["Thermal Conductivity (W/m·K)"] <= 0).any(): st.sidebar.error("Conductivity must be > 0."); load_error = True
                        else:
                            st.sidebar.success(f"{uploaded_file.type.split('/')[1].upper()} loaded!")
                            grouped = df_process.groupby("Material")
                            for material_name_df, group_df in grouped: # Renamed loop variables
                                layers_df = []
                                for idx, row_df in group_df.iterrows():
                                    layers_df.append({"name": row_df.get("Layer Name", f"Layer {len(layers_df)+1}"), "thickness": row_df["Layer Thickness (m)"], "conductivity": row_df["Thermal Conductivity (W/m·K)"]})
                                initial_material_group.append({"name": str(material_name_df), "layers": layers_df}) # Ensure material_name is string
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}"); load_error = True

    if 'material_group' not in st.session_state or not st.session_state['material_group'] or (initial_material_group and not load_error):
        if initial_material_group and not load_error: st.session_state['material_group'] = initial_material_group
        elif 'material_group' not in st.session_state or not st.session_state['material_group']:
            st.session_state['material_group'] = [{"name": "Default Material", "layers": [{"name": "Layer 1", "thickness": 0.05, "conductivity": 0.04}]}]

    num_materials = st.sidebar.number_input("Number of Materials to Define:", min_value=1, step=1, value=len(st.session_state['material_group']), key='num_materials_setup')
    current_len = len(st.session_state['material_group'])
    if num_materials > current_len:
        for i in range(current_len, num_materials): st.session_state['material_group'].append({"name": f"Material {i + 1}", "layers": [{"name": "Layer 1", "thickness": 0.05, "conductivity": 0.04}]})
    elif num_materials < current_len: st.session_state['material_group'] = st.session_state['material_group'][:num_materials]

    updated_material_group = []
    validation_passed = True
    for i in range(num_materials):
        st.sidebar.markdown("---")
        material_name_input = st.sidebar.text_input(f"Name of Material {i + 1}:", st.session_state['material_group'][i]["name"], key=f'material_name_setup_{i}')
        if not material_name_input: st.sidebar.warning(f"Material {i+1} needs a name.", icon="⚠️"); validation_passed = False
        num_layers_input = st.sidebar.number_input(f"Layers in '{material_name_input or f'Material {i+1}'}':", min_value=1, step=1, value=len(st.session_state['material_group'][i]["layers"]), key=f'num_layers_setup_{i}')
        current_layer_len = len(st.session_state['material_group'][i]['layers'])
        if num_layers_input > current_layer_len:
            for j in range(current_layer_len, num_layers_input): st.session_state['material_group'][i]['layers'].append({"name": f"Layer {j + 1}", "thickness": 0.05, "conductivity": 0.04})
        elif num_layers_input < current_layer_len: st.session_state['material_group'][i]['layers'] = st.session_state['material_group'][i]['layers'][:num_layers_input]
        
        layers_data = [] # Renamed variable
        for j in range(num_layers_input):
            layer_name_input = st.sidebar.text_input(f"Layer {j + 1} Name ('{material_name_input or f'Material {i+1}'}'):", value=st.session_state['material_group'][i]['layers'][j].get("name", f"Layer {j+1}"), key=f'layer_name_setup_{i}_{j}')
            if not layer_name_input: st.sidebar.warning(f"Layer {j+1} in '{material_name_input or f'Material {i+1}'}' needs a name.", icon="⚠️"); validation_passed = False
            layer_thickness_input = st.sidebar.number_input(f"'{layer_name_input or f'Layer {j+1}'}' Thickness (m):", min_value=0.0001, max_value=5.0, value=st.session_state['material_group'][i]['layers'][j]["thickness"], step=0.001, format="%.4f", key=f'layer_thickness_setup_{i}_{j}')
            layer_conductivity_input = st.sidebar.number_input(f"'{layer_name_input or f'Layer {j+1}'}' Conductivity (W/m·K):", min_value=0.0001, max_value=500.0, value=st.session_state['material_group'][i]['layers'][j]["conductivity"], step=0.001, format="%.4f", key=f'layer_conductivity_setup_{i}_{j}')
            layers_data.append({"name": layer_name_input, "thickness": layer_thickness_input, "conductivity": layer_conductivity_input})
        updated_material_group.append({"name": material_name_input, "layers": layers_data})
    
    if validation_passed: st.session_state['material_group'] = updated_material_group
    else: st.sidebar.error("Please fix warnings before saving/running.", icon="❌")
    
    if st.sidebar.button("Save Current Configuration (JSON)", disabled=not validation_passed, key="save_config_button_main"):
        config_data = {"materials": st.session_state['material_group']}
        st.sidebar.download_button(label="Click to Download JSON", data=json.dumps(config_data, indent=4), file_name="material_configuration.json", mime="application/json", key="download_config_json_button_setup")
    return st.session_state['material_group']


def calculate_heat_flux_advanced(layers, Delta_T_kelvin, area, use_convective_coeffs=False, h_interior_val=None, h_exterior_val=None): # Renamed to avoid conflict
    R_total_per_unit_area = 0.0
    layer_RSI_details = []
    if use_convective_coeffs:
        if h_interior_val is None or not isinstance(h_interior_val, (int,float)) or h_interior_val <= 0: raise ValueError("Invalid interior h-value.")
        if h_exterior_val is None or not isinstance(h_exterior_val, (int,float)) or  h_exterior_val <= 0: raise ValueError("Invalid exterior h-value.")
        R_surface_interior, R_surface_exterior = 1.0 / h_interior_val, 1.0 / h_exterior_val
    else:
        R_surface_interior, R_surface_exterior = 0.12, 0.03 # Default fixed RSI
    
    layer_RSI_details.append({"Layer": "Interior Surface", "RSI (K·m²/W)": R_surface_interior, "Thickness (m)": 0.0, "Conductivity (W/m·K)": float('inf')})
    R_total_per_unit_area += R_surface_interior
    
    for layer in layers:
        layer_name, thickness, conductivity = layer.get('name', 'N/A'), layer.get("thickness"), layer.get("conductivity")
        if thickness is None or conductivity is None: raise ValueError(f"Missing data for layer '{layer_name}'.")
        if not isinstance(thickness, (int,float)) or thickness <= 0 : raise ValueError(f"Invalid thickness for '{layer_name}'. Must be positive number.")
        if not isinstance(conductivity, (int,float)) or conductivity <= 0: raise ValueError(f"Invalid conductivity for '{layer_name}'. Must be positive number.")
        RSI_layer = thickness / conductivity
        R_total_per_unit_area += RSI_layer
        layer_RSI_details.append({"Layer": layer_name, "RSI (K·m²/W)": RSI_layer, "Thickness (m)": thickness, "Conductivity (W/m·K)": conductivity})
    
    layer_RSI_details.append({"Layer": "Exterior Surface", "RSI (K·m²/W)": R_surface_exterior, "Thickness (m)": 0.0, "Conductivity (W/m·K)": float('inf')})
    R_total_per_unit_area += R_surface_exterior
    
    heat_flux_per_unit_area = 0.0
    if Delta_T_kelvin != 0: # Check for actual difference
        if R_total_per_unit_area <= 1e-9: raise ValueError(f"Total R-value near zero ({R_total_per_unit_area:.4f}). Cannot divide by zero.")
        heat_flux_per_unit_area = Delta_T_kelvin / R_total_per_unit_area
    
    total_flux = heat_flux_per_unit_area * area
    return R_total_per_unit_area, heat_flux_per_unit_area, total_flux, layer_RSI_details


def calculate_annual_energy_cost_adv(total_flux, energy_cost_per_kWh, operating_hours): # Renamed
    op_hours = max(0.0, float(operating_hours) if operating_hours is not None else 0.0)
    cost_kwh = max(0.0, float(energy_cost_per_kWh) if energy_cost_per_kWh is not None else 0.0)
    energy_kWh_per_year = abs(float(total_flux)) * op_hours / 1000.0
    annual_cost = energy_kWh_per_year * cost_kwh
    return annual_cost, energy_kWh_per_year


def validate_temperature_drop_adv(Delta_T_total_kelvin, heat_flux_per_unit_area, layer_RSI_details): # Renamed
    cumulative_temp_drop_calculated = 0.0
    temp_drop_details_for_table = []
    if heat_flux_per_unit_area is None or pd.isna(heat_flux_per_unit_area) or not layer_RSI_details:
        return temp_drop_details_for_table, 0.0, False, abs(float(Delta_T_total_kelvin))
    
    for detail in layer_RSI_details:
        try:
            rsi = detail.get('RSI (K·m²/W)')
            if rsi is None or pd.isna(rsi): raise ValueError("RSI missing.")
            drop = float(heat_flux_per_unit_area) * float(rsi)
            cumulative_temp_drop_calculated += drop
            if 'Surface' not in detail.get('Layer', ''):
                temp_drop_details_for_table.append({
                    "S.No": len(temp_drop_details_for_table) + 1, "Layer": detail.get("Layer", "N/A"),
                    "Thickness (m)": detail.get("Thickness (m)", 0.0), "Conductivity (W/m·K)": detail.get("Conductivity (W/m·K)", 0.0),
                    "Thermal Resistance (K·m²/W)": float(rsi), "Temperature Drop (K)": drop
                })
        except Exception as e:
            print(f"Validation error for {detail.get('Layer', 'N/A')}: {e}")
            return temp_drop_details_for_table, cumulative_temp_drop_calculated, False, abs(float(Delta_T_total_kelvin))
    
    tolerance = 0.05 # K
    is_valid = abs(cumulative_temp_drop_calculated - float(Delta_T_total_kelvin)) < tolerance
    difference = abs(cumulative_temp_drop_calculated - float(Delta_T_total_kelvin))
    return temp_drop_details_for_table, cumulative_temp_drop_calculated, is_valid, difference


def create_heat_flux_graph_adv(data, temp_column_display_name, current_temp_unit_graph): # Renamed and added unit
    if data is None or data.empty: return None
    required_cols = [temp_column_display_name, "Heat Flux (W/m²)", "Material"]
    if not all(col in data.columns for col in required_cols):
        print(f"Graphing data missing. Need: {required_cols}"); return None
    
    graph_df = data.copy() # Work on a copy
    graph_df["Heat Flux (W/m²)"] = pd.to_numeric(graph_df["Heat Flux (W/m²)"], errors='coerce')
    graph_df[temp_column_display_name] = pd.to_numeric(graph_df[temp_column_display_name], errors='coerce')
    graph_df = graph_df.dropna(subset=["Heat Flux (W/m²)", temp_column_display_name])
    if graph_df.empty: print("No valid numeric data for graph."); return None
    
    fig = px.line(graph_df.sort_values(by=temp_column_display_name), x=temp_column_display_name, y="Heat Flux (W/m²)", color="Material",
                  title="Heat Flux Comparison Across Materials", labels={"Heat Flux (W/m²)": "Heat Flux (W/m²)", temp_column_display_name: f"{temp_column_display_name}"}, # Unit already in name
                  markers=True, template="plotly_white")
    fig.update_layout(title=dict(x=0.5, xanchor='center', font=dict(size=20)),
                      xaxis=dict(title_font=dict(size=17), tickfont=dict(size=14), showgrid=True, gridwidth=1, gridcolor='lightgrey'),
                      yaxis=dict(title_font=dict(size=17), tickfont=dict(size=14), showgrid=True, gridwidth=1, gridcolor='lightgrey'),
                      legend=dict(title="Material", bgcolor="rgba(255,255,255,0.7)", bordercolor="grey", borderwidth=1), margin=dict(l=60, r=40, t=80, b=60))
    return fig


def highlight_row_adv(row): # Renamed
    styles = [''] * len(row)
    try:
        if "Effectiveness" in row.index:
            if row["Effectiveness"] == "Most Effective": styles = ['background-color: #C8E6C9'] * len(row)
            elif row["Effectiveness"] == "Least Effective": styles = ['background-color: #FFCCBC'] * len(row)
            elif "Material" in row.index and row["Material"] == st.session_state.get('sidebar_baseline_material_select'): styles = ['background-color: #BBDEFB'] * len(row)
    except Exception as e: print(f"Error styling row: {e}")
    return styles

# --- NEW: 3D Material Visualization Function ---
def create_3d_material_visualization(material_data):
    """Creates a 3D visualization of material layers using Plotly Mesh3d."""
    if not material_data or 'layers' not in material_data or not material_data['layers']:
        return go.Figure(layout_title_text="No material data provided for 3D visualization.")

    layers = material_data['layers']
    material_name_viz = material_data.get('name', 'Unnamed Material') # Renamed

    # Define fixed width and depth for the cuboids
    width = 1.0  # x-dimension
    depth = 0.5  # z-dimension

    # Define colors for layers (cycle through if more layers than colors)
    colors = px.colors.qualitative.Plotly
    
    mesh_traces = []
    current_y_offset = 0.0
    annotations = []

    for i, layer in enumerate(layers):
        thickness = layer.get('thickness', 0.1) # Default thickness if not specified
        layer_name_viz = layer.get('name', f'Layer {i+1}') # Renamed
        color = colors[i % len(colors)]

        # Vertices of the cuboid for this layer
        x_coords = [0, width, width, 0, 0, width, width, 0]
        y_coords = [current_y_offset, current_y_offset, current_y_offset + thickness, current_y_offset + thickness,
                    current_y_offset, current_y_offset, current_y_offset + thickness, current_y_offset + thickness]
        z_coords = [0, 0, 0, 0, depth, depth, depth, depth]

        mesh_traces.append(go.Mesh3d(
            x=x_coords, y=y_coords, z=z_coords,
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], # Indices for faces
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            name=f'{layer_name_viz} (Thickness: {thickness:.3f}m)',
            color=color,
            opacity=0.8,
            hoverinfo='name'
        ))
        
        # Add annotation for layer name (position at the center of the front face)
        annotations.append(dict(
            showarrow=False,
            x=width / 2,
            y=current_y_offset + thickness / 2,
            z=0, # On the front face
            text=f"{layer_name_viz}",
            xanchor="center",
            yanchor="middle",
            font=dict(color="black", size=10)
        ))
        current_y_offset += thickness

    fig_3d = go.Figure(data=mesh_traces) # Renamed
    fig_3d.update_layout(
        title=f'3D Layered Model: {material_name_viz}',
        scene=dict(
            xaxis_title='Width (m)',
            yaxis_title='Cumulative Thickness (m)',
            zaxis_title='Depth (m)',
            aspectmode='data', # Makes axes scale according to data ranges
            # annotations=annotations # Annotations can be tricky with Mesh3d, might overlap
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig_3d


# --- Main App Layout ---
st.markdown(
    '<div class="header-title">ThermoPro: Advanced Insulation Analysis & Energy Optimization Tool</div>',
    unsafe_allow_html=True,
)
st.write("Define your layered materials, set environmental conditions, and analyze heat transfer and energy costs.")

# --- Sidebar Content ---
with st.sidebar:
    material_group_main = create_material_group() # Renamed variable

    st.markdown("""<div class="sidebar-section"><h3 class="centered-heading">Material Comparison</h3></div>""", unsafe_allow_html=True)
    material_names_main = [material["name"] for material in st.session_state.get('material_group', []) if material.get("name")] # Renamed variable
    st.session_state.setdefault('sidebar_baseline_material_main_select', None) # Renamed key
    st.session_state.setdefault('sidebar_comparison_materials_main_select', []) # Renamed key

    baseline_material_main = None # Renamed variable
    selected_comparison_materials_main = [] # Renamed variable

    if not material_names_main:
        st.warning("Please define at least one named material above.", icon="⚠️")
        st.session_state['sidebar_baseline_material_main_select'] = None
        st.session_state['sidebar_comparison_materials_main_select'] = []
    else:
        if st.session_state['sidebar_baseline_material_main_select'] not in material_names_main:
            st.session_state['sidebar_baseline_material_main_select'] = material_names_main[0]
        baseline_material_main = st.selectbox("Select Baseline Material:", material_names_main, key='sidebar_baseline_material_main_select')
        comparison_options_main = [name for name in material_names_main if name != baseline_material_main] # Renamed variable
        st.session_state['sidebar_comparison_materials_main_select'] = [name for name in st.session_state.get('sidebar_comparison_materials_main_select', []) if name in comparison_options_main]
        selected_comparison_materials_main = st.multiselect("Select Materials to Compare:", comparison_options_main, key='sidebar_comparison_materials_main_select')

    st.markdown("""<div class="sidebar-section"><h3 class="centered-heading">Simulation Parameters</h3></div>""", unsafe_allow_html=True)
    
    st.session_state.setdefault('temp_unit_sidebar_select', "K") # Renamed key
    temp_unit_display_sidebar = st.selectbox("Temperature Unit for Input/Display:", ["K", "°C", "°F"], key='temp_unit_sidebar_select')

    st.session_state.setdefault('fixed_temp_option_sidebar_select', "Interior") # Renamed key
    st.session_state.setdefault('fixed_interior_temp_k_sidebar_slider', 294.15) # Renamed key
    st.session_state.setdefault('fixed_exterior_temp_k_sidebar_slider', 273.15) # Renamed key
    st.session_state.setdefault('ext_temp_min_k_sidebar_slider', 255.15) # Renamed key
    st.session_state.setdefault('ext_temp_max_k_sidebar_slider', 294.15) # Renamed key
    st.session_state.setdefault('int_temp_min_k_sidebar_slider', 294.15) # Renamed key
    st.session_state.setdefault('int_temp_max_k_sidebar_slider', 310.15) # Renamed key
    st.session_state.setdefault('wall_area_sidebar_input', 10.0) # Renamed key
    st.session_state.setdefault('energy_cost_sidebar_input', 0.12) # Renamed key
    st.session_state.setdefault('operating_hours_sidebar_input', 8760) # Renamed key

    fixed_temp_option_sidebar = st.radio("Fix Temperature At:", ["Interior", "Exterior"], key="fixed_temp_option_sidebar_select") # Renamed variable
    
    st.markdown("---"); st.markdown("<h5>Boundary Conditions</h5>", unsafe_allow_html=True)
    st.session_state.setdefault('use_convective_coeffs_sidebar_check', False) # Renamed key
    use_convective_coeffs_sidebar = st.checkbox("Use Convective Heat Transfer Coefficients (h-values)", key='use_convective_coeffs_sidebar_check') # Renamed variable
    h_interior_val_sidebar, h_exterior_val_sidebar = None, None # Renamed variables
    if use_convective_coeffs_sidebar:
        st.session_state.setdefault('h_interior_sidebar_input', 5.0); st.session_state.setdefault('h_exterior_sidebar_input', 25.0) # Renamed keys
        h_interior_val_sidebar = st.number_input("Interior h-value (W/m²K):", min_value=0.1, value=st.session_state.h_interior_sidebar_input, step=0.1, key='h_interior_sidebar_input')
        h_exterior_val_sidebar = st.number_input("Exterior h-value (W/m²K):", min_value=0.1, value=st.session_state.h_exterior_sidebar_input, step=0.1, key='h_exterior_sidebar_input')
    else:
        st.info("Using default fixed surface resistances (R_int=0.12, R_ext=0.03 K·m²/W)")

    actual_temp_values_kelvin_sidebar = [] # Renamed variable
    temp_column_name_base_sidebar = "" # Renamed variable
    fixed_temp_column_name_base_sidebar = "" # Renamed variable
    fixed_temp_kelvin_sidebar = 0.0 # Renamed variable

    def get_slider_params_sidebar(k_val, unit_display_param, slider_key_suffix_param): # Renamed parameters
        slider_k_min, slider_k_max = 100.0, 500.0
        val_disp = round(convert_temperature_value(k_val, "K", unit_display_param))
        min_disp = round(convert_temperature_value(slider_k_min, "K", unit_display_param))
        max_disp = round(convert_temperature_value(slider_k_max, "K", unit_display_param))
        return val_disp, min_disp, max_disp

    if fixed_temp_option_sidebar == "Interior":
        val_disp_fixed_int, min_disp_fixed_int, max_disp_fixed_int = get_slider_params_sidebar(st.session_state.fixed_interior_temp_k_sidebar_slider, temp_unit_display_sidebar, "fixed_int")
        fixed_temp_input_disp_sidebar = st.slider(f"Fixed Interior Temp ({temp_unit_display_sidebar}):", min_disp_fixed_int, max_disp_fixed_int, val_disp_fixed_int, key="fixed_interior_temp_sidebar_slider_disp") # Renamed key
        st.session_state.fixed_interior_temp_k_sidebar_slider = convert_temperature_value(fixed_temp_input_disp_sidebar, temp_unit_display_sidebar, "K")
        fixed_temp_kelvin_sidebar = st.session_state.fixed_interior_temp_k_sidebar_slider

        val_min_disp_ext, min_r_min_disp_ext, max_r_min_disp_ext = get_slider_params_sidebar(st.session_state.ext_temp_min_k_sidebar_slider, temp_unit_display_sidebar, "ext_min")
        val_max_disp_ext, min_r_max_disp_ext, max_r_max_disp_ext = get_slider_params_sidebar(st.session_state.ext_temp_max_k_sidebar_slider, temp_unit_display_sidebar, "ext_max")
        temp_range_min_input_disp_sidebar = st.slider(f"Exterior Temp Range (Min {temp_unit_display_sidebar}):", min_r_min_disp_ext, max_r_min_disp_ext, val_min_disp_ext, key="ext_temp_min_sidebar_slider_disp") # Renamed key
        temp_range_max_input_disp_sidebar = st.slider(f"Exterior Temp Range (Max {temp_unit_display_sidebar}):", min_r_max_disp_ext, max_r_max_disp_ext, val_max_disp_ext, key="ext_temp_max_sidebar_slider_disp") # Renamed key
        st.session_state.ext_temp_min_k_sidebar_slider = convert_temperature_value(temp_range_min_input_disp_sidebar, temp_unit_display_sidebar, "K")
        st.session_state.ext_temp_max_k_sidebar_slider = convert_temperature_value(temp_range_max_input_disp_sidebar, temp_unit_display_sidebar, "K")

        if st.session_state.ext_temp_min_k_sidebar_slider > st.session_state.ext_temp_max_k_sidebar_slider: st.warning(f"Min Exterior Temp > Max Exterior Temp", icon="⚠️")
        actual_temp_values_kelvin_sidebar = list(range(int(round(min(st.session_state.ext_temp_min_k_sidebar_slider, st.session_state.ext_temp_max_k_sidebar_slider))), int(round(max(st.session_state.ext_temp_min_k_sidebar_slider, st.session_state.ext_temp_max_k_sidebar_slider))) + 1))
        temp_column_name_base_sidebar = "Exterior Temperature"
        fixed_temp_column_name_base_sidebar = "Interior Temperature"
    else: # Exterior
        val_disp_fixed_ext, min_disp_fixed_ext, max_disp_fixed_ext = get_slider_params_sidebar(st.session_state.fixed_exterior_temp_k_sidebar_slider, temp_unit_display_sidebar, "fixed_ext")
        fixed_temp_input_disp_sidebar = st.slider(f"Fixed Exterior Temp ({temp_unit_display_sidebar}):", min_disp_fixed_ext, max_disp_fixed_ext, val_disp_fixed_ext, key="fixed_exterior_temp_sidebar_slider_disp") # Renamed key
        st.session_state.fixed_exterior_temp_k_sidebar_slider = convert_temperature_value(fixed_temp_input_disp_sidebar, temp_unit_display_sidebar, "K")
        fixed_temp_kelvin_sidebar = st.session_state.fixed_exterior_temp_k_sidebar_slider

        val_min_disp_int, min_r_min_disp_int, max_r_min_disp_int = get_slider_params_sidebar(st.session_state.int_temp_min_k_sidebar_slider, temp_unit_display_sidebar, "int_min")
        val_max_disp_int, min_r_max_disp_int, max_r_max_disp_int = get_slider_params_sidebar(st.session_state.int_temp_max_k_sidebar_slider, temp_unit_display_sidebar, "int_max")
        temp_range_min_input_disp_sidebar = st.slider(f"Interior Temp Range (Min {temp_unit_display_sidebar}):", min_r_min_disp_int, max_r_min_disp_int, val_min_disp_int, key="int_temp_min_sidebar_slider_disp") # Renamed key
        temp_range_max_input_disp_sidebar = st.slider(f"Interior Temp Range (Max {temp_unit_display_sidebar}):", min_r_max_disp_int, max_r_max_disp_int, val_max_disp_int, key="int_temp_max_sidebar_slider_disp") # Renamed key
        st.session_state.int_temp_min_k_sidebar_slider = convert_temperature_value(temp_range_min_input_disp_sidebar, temp_unit_display_sidebar, "K")
        st.session_state.int_temp_max_k_sidebar_slider = convert_temperature_value(temp_range_max_input_disp_sidebar, temp_unit_display_sidebar, "K")

        if st.session_state.int_temp_min_k_sidebar_slider > st.session_state.int_temp_max_k_sidebar_slider: st.warning(f"Min Interior Temp > Max Interior Temp", icon="⚠️")
        actual_temp_values_kelvin_sidebar = list(range(int(round(min(st.session_state.int_temp_min_k_sidebar_slider, st.session_state.int_temp_max_k_sidebar_slider))), int(round(max(st.session_state.int_temp_min_k_sidebar_slider, st.session_state.int_temp_max_k_sidebar_slider))) + 1))
        temp_column_name_base_sidebar = "Interior Temperature"
        fixed_temp_column_name_base_sidebar = "Exterior Temperature"
    
    temp_column_name_display_sidebar = f"{temp_column_name_base_sidebar} ({temp_unit_display_sidebar})" # Renamed variable
    fixed_temp_column_name_display_sidebar = f"{fixed_temp_column_name_base_sidebar} ({temp_unit_display_sidebar})" # Renamed variable

    area_sidebar = st.number_input("Wall Area (m²):", min_value=0.01, key="wall_area_sidebar_input") # Renamed variable
    energy_cost_per_kWh_sidebar = st.number_input("Energy Cost ($/kWh):", min_value=0.001, format="%.3f", key="energy_cost_sidebar_input") # Renamed variable
    operating_hours_sidebar = st.number_input("Operating Hours per Year:", min_value=0, step=1, key="operating_hours_sidebar_input") # Renamed variable

    st.sidebar.markdown("""<div style="text-align:center; background-color:#E8F8F5; padding:10px; border-radius:10px; margin-top:20px; font-size:12px;"><b>Developed by Jayaprakash Chandran</b><br>© 2025 | All Rights Reserved</div>""", unsafe_allow_html=True)

# --- Simulation Logic (Main Area) ---
current_material_group_main = st.session_state.get('material_group', []) # Renamed variable
current_baseline_main = st.session_state.get('sidebar_baseline_material_main_select')
current_comparisons_main = st.session_state.get('sidebar_comparison_materials_main_select', [])

simulation_ready_main = (current_material_group_main and current_baseline_main is not None and
                    current_baseline_main in [m.get("name") for m in current_material_group_main] and actual_temp_values_kelvin_sidebar) # Renamed variable

if simulation_ready_main:
    valid_comparison_materials_main = [name for name in current_comparisons_main if name != current_baseline_main and name in [m.get("name") for m in current_material_group_main]] # Renamed variable
    materials_to_simulate_info_main = [] # Renamed variable
    baseline_dict_main = next((m for m in current_material_group_main if m.get("name") == current_baseline_main), None) # Renamed variable
    if baseline_dict_main: materials_to_simulate_info_main.append(baseline_dict_main)
    for comp_name_main in valid_comparison_materials_main: # Renamed variable
        comp_dict_main = next((m for m in current_material_group_main if m.get("name") == comp_name_main), None) # Renamed variable
        if comp_dict_main: materials_to_simulate_info_main.append(comp_dict_main)

    if not materials_to_simulate_info_main:
        st.warning("No valid materials selected for simulation (check names and selections).", icon="⚠️")
        st.session_state.pop('simulation_df_full', None); st.session_state.pop('validation_messages', None)
    else:
        st.subheader("Running Simulation...")
        all_simulation_results_main = []; validation_messages_main = [] # Renamed variables
        progress_bar_main = st.progress(0) # Renamed variable
        total_steps_main = len(materials_to_simulate_info_main) * len(actual_temp_values_kelvin_sidebar) # Renamed variable
        step_count_main = 0 # Renamed variable

        current_use_convective_main = st.session_state.get('use_convective_coeffs_sidebar_check', False) # Renamed variable
        current_h_int_main = st.session_state.get('h_interior_sidebar_input', None) # Renamed variable
        current_h_ext_main = st.session_state.get('h_exterior_sidebar_input', None) # Renamed variable

        for material_info_main in materials_to_simulate_info_main: # Renamed variable
            material_name_sim = material_info_main.get("name", "Unnamed Material") # Renamed variable
            layers_sim = material_info_main.get("layers", []) # Renamed variable
            if not material_name_sim or not layers_sim:
                validation_messages_main.append(f"Skipping invalid material entry: {material_info_main}"); total_steps_main = max(1, total_steps_main - len(actual_temp_values_kelvin_sidebar)); continue
            for actual_temp_k_sim in actual_temp_values_kelvin_sidebar: # Renamed variable
                step_count_main += 1; progress_bar_main.progress(min(step_count_main / max(total_steps_main, 1), 1.0))
                try:
                    numeric_fixed_temp_k_sim = float(fixed_temp_kelvin_sidebar) # Renamed variable
                    numeric_actual_temp_k_sim = float(actual_temp_k_sim) # Renamed variable
                    Delta_T_k_sim = numeric_fixed_temp_k_sim - numeric_actual_temp_k_sim # Renamed variable
                    
                    (R_total_sim, flux_per_area_sim, total_flux_sim, rsi_details_sim) = calculate_heat_flux_advanced(layers_sim, Delta_T_k_sim, area_sidebar, current_use_convective_main, current_h_int_main, current_h_ext_main) # Renamed variables
                    (cost_sim, kwh_sim) = calculate_annual_energy_cost_adv(total_flux_sim, energy_cost_per_kWh_sidebar, operating_hours_sidebar) # Renamed variables
                    (drop_details_sim, sum_drops_sim, is_valid_sim, diff_sim) = validate_temperature_drop_adv(Delta_T_k_sim, flux_per_area_sim, rsi_details_sim) # Renamed variables
                    
                    all_simulation_results_main.append({
                        "Material": material_name_sim,
                        fixed_temp_column_name_display_sidebar: convert_temperature_value(numeric_fixed_temp_k_sim, "K", temp_unit_display_sidebar),
                        temp_column_name_display_sidebar: convert_temperature_value(numeric_actual_temp_k_sim, "K", temp_unit_display_sidebar),
                        "Temperature Difference (K)": Delta_T_k_sim, "Heat Flux (W/m²)": flux_per_area_sim,
                        "Total Heat Flux (W)": total_flux_sim, "Effective Thermal Resistance (K·m²/W)": R_total_sim,
                        "Annual Energy Cost ($)": cost_sim, "Annual Energy Use (kWh)": kwh_sim,
                        "Validation Passed": is_valid_sim, "Temp Drop Difference (K)": diff_sim, "Layer_Details": drop_details_sim,
                        f"{fixed_temp_column_name_base_sidebar} (K)": numeric_fixed_temp_k_sim,
                        f"{temp_column_name_base_sidebar} (K)": numeric_actual_temp_k_sim
                    })
                    if not is_valid_sim: validation_messages_main.append(f"Warning: '{material_name_sim}' @ {temp_column_name_display_sidebar}={convert_temperature_value(numeric_actual_temp_k_sim, 'K', temp_unit_display_sidebar)}{temp_unit_display_sidebar} (ΔT={Delta_T_k_sim:.2f}K): Sum drops ({sum_drops_sim:.2f}K) vs Expected ({Delta_T_k_sim:.2f}K) -> Diff={diff_sim:.2f}K")
                except ValueError as e_val: validation_messages_main.append(f"ERROR: Calc failed for '{material_name_sim}' @ {temp_column_name_display_sidebar}={convert_temperature_value(numeric_actual_temp_k_sim, 'K', temp_unit_display_sidebar)}{temp_unit_display_sidebar}: {e_val}") # Renamed variable
                except Exception as e_exc: validation_messages_main.append(f"CRITICAL ERROR for '{material_name_sim}' @ {temp_column_name_display_sidebar}={convert_temperature_value(numeric_actual_temp_k_sim, 'K', temp_unit_display_sidebar)}{temp_unit_display_sidebar}: {e_exc}\n{traceback.format_exc()}") # Renamed variable
        progress_bar_main.empty()
        if all_simulation_results_main: st.session_state['simulation_df_full'] = pd.DataFrame(all_simulation_results_main)
        else: st.session_state.pop('simulation_df_full', None)
        st.session_state['validation_messages'] = validation_messages_main
        st.session_state['temp_column_name_display_actual_run'] = temp_column_name_display_sidebar # Store the actual display name used in this run
        st.session_state['fixed_temp_column_name_display_actual_run'] = fixed_temp_column_name_display_sidebar
        st.session_state['temp_unit_display_actual_run'] = temp_unit_display_sidebar # Store the unit used for this run

# --- Display Results (Reads from session state) ---
if 'simulation_df_full' in st.session_state and not st.session_state['simulation_df_full'].empty:
    simulation_df_full_results = st.session_state['simulation_df_full'] # Renamed variable
    validation_messages_results = st.session_state.get('validation_messages', []) # Renamed variable
    temp_col_name_disp_results = st.session_state.get('temp_column_name_display_actual_run', f"Varying Temp ({st.session_state.get('temp_unit_display_actual_run', 'K')})") # Renamed variable
    fixed_col_name_disp_results = st.session_state.get('fixed_temp_column_name_display_actual_run', f"Fixed Temp ({st.session_state.get('temp_unit_display_actual_run', 'K')})") # Renamed variable
    current_run_temp_unit_results = st.session_state.get('temp_unit_display_actual_run', 'K') # Renamed variable

    display_columns_results = [ # Renamed variable
        "Material", fixed_col_name_disp_results, temp_col_name_disp_results, "Temperature Difference (K)",
        "Heat Flux (W/m²)", "Total Heat Flux (W)", "Effective Thermal Resistance (K·m²/W)",
        "Annual Energy Cost ($)", "Annual Energy Use (kWh)", "Validation Passed", "Temp Drop Difference (K)"
    ]
    display_columns_exist_results = [col for col in display_columns_results if col in simulation_df_full_results.columns] # Renamed variable
    simulation_df_display_results = simulation_df_full_results[display_columns_exist_results].copy() # Renamed variable

    st.header("Simulation Results")
    avg_heat_flux_results = pd.Series(dtype=float); most_effective_material_results, least_effective_material_results = "N/A", "N/A" # Renamed variables
    flux_col_name_results = "Heat Flux (W/m²)" # Renamed variable
    if flux_col_name_results in simulation_df_display_results.columns:
        numeric_flux_results = pd.to_numeric(simulation_df_display_results[flux_col_name_results], errors='coerce') # Renamed variable
        valid_flux_df_results = simulation_df_display_results.loc[numeric_flux_results.notna()].copy() # Renamed variable
        valid_flux_df_results[flux_col_name_results] = numeric_flux_results.dropna()
        if not valid_flux_df_results.empty:
            avg_heat_flux_results = valid_flux_df_results.groupby("Material")[flux_col_name_results].apply(lambda x: x.abs().mean())
            if not avg_heat_flux_results.empty:
                most_effective_material_results, least_effective_material_results = avg_heat_flux_results.idxmin(), avg_heat_flux_results.idxmax()
                simulation_df_display_results["Effectiveness"] = simulation_df_display_results["Material"].map({most_effective_material_results: "Most Effective", least_effective_material_results: "Least Effective"}).fillna("Comparison")

    st.subheader("Heat Flux Comparison Graph")
    graph_materials_options_results = simulation_df_display_results["Material"].unique().tolist() # Renamed variable
    graph_results = None # Renamed variable
    if graph_materials_options_results:
        st.session_state.setdefault('selected_graph_materials_chart_results', graph_materials_options_results) # Renamed key
        current_default_graph_selection_results = [m for m in st.session_state['selected_graph_materials_chart_results'] if m in graph_materials_options_results] # Renamed variable
        selected_graph_materials_chart_results = st.multiselect("Select Materials to Visualize:", graph_materials_options_results, default=current_default_graph_selection_results, key='selected_graph_materials_chart_results') # Renamed variable
        if selected_graph_materials_chart_results:
            filtered_graph_df_results = simulation_df_display_results[simulation_df_display_results["Material"].isin(selected_graph_materials_chart_results)] # Renamed variable
            if not filtered_graph_df_results.empty:
                graph_results = create_heat_flux_graph_adv(filtered_graph_df_results, temp_col_name_disp_results, current_run_temp_unit_results)
                if graph_results: st.plotly_chart(graph_results, use_container_width=True)
                else: st.info("Could not generate graph for selected data.")
            else: st.info("Selected materials have no simulation data to plot.")
        else: st.info("Select materials above to visualize the graph.")
    else: st.info("Run the simulation to generate data for the graph.")

    st.subheader("Detailed Results Table")
    simulated_temps_display_vals_results = simulation_df_display_results[temp_col_name_disp_results].unique().tolist() # Renamed variable
    current_fixed_temp_disp_val_results = simulation_df_display_results[fixed_col_name_disp_results].iloc[0] if not simulation_df_display_results.empty else 'N/A' # Renamed variable
    if simulated_temps_display_vals_results:
        range_desc_results = (f"Results for {fixed_col_name_disp_results}={current_fixed_temp_disp_val_results}, "
                      f"{temp_col_name_disp_results} varying from {min(simulated_temps_display_vals_results):.0f} to {max(simulated_temps_display_vals_results):.0f}.") # Renamed variable
        st.markdown(f"**{range_desc_results}**")
    formatting_results = { # Renamed variable
        "Temperature Difference (K)": "{:.2f}", "Heat Flux (W/m²)": "{:.2f}", "Total Heat Flux (W)": "{:.2f}",
        "Effective Thermal Resistance (K·m²/W)": "{:.3f}", "Annual Energy Cost ($)": "${:.2f}",
        "Annual Energy Use (kWh)": "{:.2f}", "Temp Drop Difference (K)": "{:.3f}",
        fixed_col_name_disp_results: "{:.0f}", temp_col_name_disp_results: "{:.0f}"
    }
    if "Effectiveness" in simulation_df_display_results.columns: st.dataframe(simulation_df_display_results.style.apply(highlight_row_adv, axis=1).format(formatting_results, na_rep="N/A"))
    else: st.dataframe(simulation_df_display_results.style.format(formatting_results, na_rep="N/A"))

    st.subheader("Validation Summary")
    if validation_messages_results:
        errors_results = [m for m in validation_messages_results if "ERROR:" in m] # Renamed variable
        warnings_val_results = [m for m in validation_messages_results if "Warning:" in m] # Renamed variable
        if errors_results: st.error("Errors occurred during calculation:", icon="❌"); [st.code(msg, language=None) for msg in errors_results]
        if warnings_val_results: st.warning("Validation warnings (sum of temp drops may not exactly match total ΔT due to precision):", icon="⚠️"); [st.code(msg, language=None) for msg in warnings_val_results]
    else: st.success("All calculations and temperature drop validations passed successfully.", icon="✅")

    st.subheader("Material Insights")
    current_baseline_material_results = st.session_state.get('sidebar_baseline_material_main_select', 'N/A') # Renamed variable
    st.write(f"🔹 **Baseline Material**: **{current_baseline_material_results}**")
    if most_effective_material_results != "N/A" and most_effective_material_results in avg_heat_flux_results.index: st.write(f"✅ **Most Effective (lowest avg. |Heat Flux|)**: **{most_effective_material_results}** ({avg_heat_flux_results[most_effective_material_results]:.2f} W/m²)")
    if least_effective_material_results != "N/A" and least_effective_material_results in avg_heat_flux_results.index: st.write(f"❌ **Least Effective (highest avg. |Heat Flux|)**: **{least_effective_material_results}** ({avg_heat_flux_results[least_effective_material_results]:.2f} W/m²)")
    if current_baseline_material_results in avg_heat_flux_results.index and len(avg_heat_flux_results) > 1:
        st.markdown("#### Performance vs Baseline (Average)")
        baseline_avg_flux_mag_results = avg_heat_flux_results[current_baseline_material_results] # Renamed variable
        baseline_avg_cost_results = simulation_df_display_results[simulation_df_display_results["Material"] == current_baseline_material_results]["Annual Energy Cost ($)"].mean() # Renamed variable
        if pd.notna(baseline_avg_flux_mag_results) and pd.notna(baseline_avg_cost_results):
            insights_results = [] # Renamed variable
            for material_name_insight_results in simulation_df_display_results["Material"].unique(): # Renamed variable
                if material_name_insight_results != current_baseline_material_results and material_name_insight_results in avg_heat_flux_results.index:
                    comp_avg_flux_mag_results = avg_heat_flux_results[material_name_insight_results] # Renamed variable
                    comp_avg_cost_results = simulation_df_display_results[simulation_df_display_results["Material"] == material_name_insight_results]["Annual Energy Cost ($)"].mean() # Renamed variable
                    flux_comp_results, cost_comp_results = "N/A", "N/A" # Renamed variables
                    if pd.notna(comp_avg_flux_mag_results) and baseline_avg_flux_mag_results > 1e-9: diff_pct_results = ((baseline_avg_flux_mag_results - comp_avg_flux_mag_results) / baseline_avg_flux_mag_results) * 100; flux_comp_results = f"{abs(diff_pct_results):.1f}% {'lower' if diff_pct_results > 0 else 'higher'} |flux|" # Renamed variable
                    elif pd.notna(comp_avg_flux_mag_results): flux_comp_results = f"{'lower' if comp_avg_flux_mag_results < baseline_avg_flux_mag_results else 'higher'} |flux|"
                    if pd.notna(comp_avg_cost_results) and baseline_avg_cost_results > 1e-9: diff_pct_results = ((baseline_avg_cost_results - comp_avg_cost_results) / baseline_avg_cost_results) * 100; cost_comp_results = f"{abs(diff_pct_results):.1f}% {'lower' if diff_pct_results > 0 else 'higher'} cost"
                    elif pd.notna(comp_avg_cost_results): cost_comp_results = f"{'lower' if comp_avg_cost_results < baseline_avg_cost_results else 'higher'} cost"
                    insights_results.append(f"• **{material_name_insight_results}**: {flux_comp_results}, {cost_comp_results}")
            if insights_results: [st.write(insight) for insight in insights_results]
            else: st.info("No comparison materials with valid data.")
        else: st.warning("Baseline material has missing average flux/cost data.")

    st.header("Detailed Layer Analysis")
    detail_material_options_results = simulation_df_full_results["Material"].unique().tolist() # Renamed variable
    if detail_material_options_results:
        default_detail_material_results = current_baseline_material_results if current_baseline_material_results in detail_material_options_results else detail_material_options_results[0] # Renamed variable
        st.session_state.setdefault('selected_detail_material_results_select', default_detail_material_results) # Renamed key
        if st.session_state['selected_detail_material_results_select'] not in detail_material_options_results: st.session_state['selected_detail_material_results_select'] = detail_material_options_results[0]
        selected_detail_material_results = st.selectbox("Select Material:", detail_material_options_results, key='selected_detail_material_results_select') # Renamed variable
        if selected_detail_material_results:
            material_specific_results_detail = simulation_df_full_results[simulation_df_full_results["Material"] == selected_detail_material_results] # Renamed variable
            if not material_specific_results_detail.empty:
                detail_temp_options_results = sorted(material_specific_results_detail[temp_col_name_disp_results].unique().tolist()) # Renamed variable
                if detail_temp_options_results:
                    st.session_state.setdefault('selected_detail_temp_point_results_select', detail_temp_options_results[0]) # Renamed key
                    if st.session_state['selected_detail_temp_point_results_select'] not in detail_temp_options_results: st.session_state['selected_detail_temp_point_results_select'] = detail_temp_options_results[0]
                    selected_detail_temp_point_disp_results = st.selectbox(f"Select {temp_col_name_disp_results} point:", detail_temp_options_results, key='selected_detail_temp_point_results_select') # Renamed variable
                    
                    selected_row_detail_candidates_results = material_specific_results_detail[abs(material_specific_results_detail[temp_col_name_disp_results].astype(float) - float(selected_detail_temp_point_disp_results)) < 1e-6] # Renamed variable
                    if not selected_row_detail_candidates_results.empty:
                        selected_row_detail_results = selected_row_detail_candidates_results.iloc[0] # Renamed variable
                        layer_details_for_display_results = selected_row_detail_results.get('Layer_Details', []) # Renamed variable
                        if layer_details_for_display_results:
                            delta_t_k_for_row_results = selected_row_detail_results['Temperature Difference (K)'] # Renamed variable
                            st.markdown(f"#### Layer Details for **{selected_detail_material_results}** at **{temp_col_name_disp_results} = {selected_detail_temp_point_disp_results}** (ΔT = {delta_t_k_for_row_results:.2f} K)")
                            layer_detail_df_results = pd.DataFrame(layer_details_for_display_results) # Renamed variable
                            st.dataframe(layer_detail_df_results.style.format({"Thickness (m)": "{:.4f}", "Conductivity (W/m·K)": "{:.4f}", "Thermal Resistance (K·m²/W)": "{:.4f}", "Temperature Drop (K)": "{:.2f}"}))
                            st.markdown("#### Temperature Drop Across Layers")
                            fig_layer_drop_results = px.bar(layer_detail_df_results, x="Layer", y="Temperature Drop (K)", title=f"Temp Drop per Layer ({selected_detail_material_results} @ {selected_detail_temp_point_disp_results})", template="plotly_white", text="Temperature Drop (K)") # Renamed variable
                            fig_layer_drop_results.update_layout(xaxis={'categoryorder':'array', 'categoryarray': layer_detail_df_results["Layer"].tolist()})
                            fig_layer_drop_results.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                            st.plotly_chart(fig_layer_drop_results, use_container_width=True)
                        else: st.info("No layer details available for this point.")
                    else: st.info(f"Could not find exact data for temp point {selected_detail_temp_point_disp_results}. Try adjusting the range slightly.")
                else: st.info("No temperature points found for this material.")
            else: st.info(f"No simulation results found for material '{selected_detail_material_results}'.")
        else: st.info("Select a material.")
    else: st.info("Run simulation to generate data for detailed layer analysis.")

    # --- NEW: 3D Material Visualization Section ---
    st.header("3D Material Visualization")
    if current_material_group_main: # Check if materials are defined
        material_names_for_3d_viz = [m['name'] for m in current_material_group_main if m.get('name')] # Renamed variable
        if material_names_for_3d_viz:
            st.session_state.setdefault('selected_material_for_3d_viz', material_names_for_3d_viz[0]) # Renamed key
            if st.session_state['selected_material_for_3d_viz'] not in material_names_for_3d_viz:
                 st.session_state['selected_material_for_3d_viz'] = material_names_for_3d_viz[0]

            selected_material_name_3d = st.selectbox( # Renamed variable
                "Select Material to Visualize in 3D:",
                material_names_for_3d_viz,
                key='selected_material_for_3d_viz'
            )
            selected_material_data_3d = next((m for m in current_material_group_main if m.get("name") == selected_material_name_3d), None) # Renamed variable
            if selected_material_data_3d:
                fig_3d_viz = create_3d_material_visualization(selected_material_data_3d) # Renamed variable
                st.plotly_chart(fig_3d_viz, use_container_width=True)
            else:
                st.info("Selected material data not found.")
        else:
            st.info("No named materials available for 3D visualization. Please define materials with names.")
    else:
        st.info("Define materials in the sidebar to visualize them in 3D.")


    st.header("Download Reports")
    @st.cache_data(persist="disk")
    def get_excel_report_cached_final(data_bytes_param, temp_col_disp_name_param, fixed_temp_col_disp_name_param): # Renamed parameters
        data_df = pd.read_feather(io.BytesIO(data_bytes_param)) # Renamed variable
        excel_buffer_obj = io.BytesIO() # Renamed variable
        report_columns_excel = [ # Renamed variable
            "Material", fixed_temp_col_disp_name_param, temp_col_disp_name_param, "Temperature Difference (K)",
            "Heat Flux (W/m²)", "Total Heat Flux (W)", "Effective Thermal Resistance (K·m²/W)",
            "Annual Energy Cost ($)", "Annual Energy Use (kWh)", "Validation Passed", "Temp Drop Difference (K)"
        ]
        report_columns_exist_excel = [col for col in report_columns_excel if col in data_df.columns] # Renamed variable
        df_report_excel = data_df[report_columns_exist_excel] # Renamed variable
        if df_report_excel.empty: print("No data for Excel report."); return None
        try:
            with pd.ExcelWriter(excel_buffer_obj, engine='xlsxwriter') as writer_excel: # Renamed variable
                df_report_excel.to_excel(writer_excel, sheet_name="Simulation Results", index=False)
                workbook_excel, worksheet_excel = writer_excel.book, writer_excel.sheets["Simulation Results"] # Renamed variables
                header_format_excel = workbook_excel.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1, 'align': 'center', 'valign': 'vcenter'}) # Renamed variable
                data_format_excel, num_format_2dp_excel, num_format_3dp_excel, cost_format_excel = \
                    workbook_excel.add_format({'border': 1, 'align': 'center'}), \
                    workbook_excel.add_format({'border': 1, 'align': 'center', 'num_format': '0.00'}), \
                    workbook_excel.add_format({'border': 1, 'align': 'center', 'num_format': '0.000'}), \
                    workbook_excel.add_format({'border': 1, 'align': 'center', 'num_format': '$0.00'}) # Renamed variables
                for col_num_excel, value_excel in enumerate(df_report_excel.columns): # Renamed variables
                    worksheet_excel.write(0, col_num_excel, value_excel, header_format_excel)
                    col_len_excel = max(df_report_excel[value_excel].astype(str).map(len).max(), len(str(value_excel))) + 2 # Ensure value is string for len # Renamed variable
                    fmt_excel = data_format_excel # Renamed variable
                    if value_excel in ["Temperature Difference (K)", "Heat Flux (W/m²)", "Total Heat Flux (W)", "Annual Energy Use (kWh)"]: fmt_excel = num_format_2dp_excel
                    elif value_excel in ["Effective Thermal Resistance (K·m²/W)", "Temp Drop Difference (K)"]: fmt_excel = num_format_3dp_excel
                    elif value_excel == "Annual Energy Cost ($)": fmt_excel = cost_format_excel
                    elif value_excel == fixed_temp_col_disp_name_param or value_excel == temp_col_disp_name_param: fmt_excel = workbook_excel.add_format({'border': 1, 'align': 'center', 'num_format': '0'})
                    worksheet_excel.set_column(col_num_excel, col_num_excel, col_len_excel, fmt_excel)
            excel_buffer_obj.seek(0)
            return excel_buffer_obj
        except Exception as e_excel_report: print(f"Error generating Excel report: {e_excel_report}"); return None # Renamed variable

    @st.cache_data(persist="disk")
    def get_pdf_report_cached_final(data_bytes_param, fig_json_str_param, temp_col_disp_name_param, fixed_temp_col_disp_name_param, fixed_temp_val_disp_param, temp_unit_disp_param, fixed_temp_option_param, area_param, energy_cost_param, operating_hours_param, validation_msgs_param): # Renamed parameters
        data_df = pd.read_feather(io.BytesIO(data_bytes_param)) # Renamed variable
        plotly_fig_pdf = pio.from_json(fig_json_str_param) if fig_json_str_param else None # Renamed variable
        pdf_obj = FPDF() # Renamed variable
        pdf_obj.add_page(); pdf_obj.set_font("Arial", "B", 14); pdf_obj.cell(0, 10, "ThermoPro Simulation Results Report", 0, 1, "C"); pdf_obj.ln(10)
        pdf_obj.set_font("Arial", "B", 10); pdf_obj.cell(0, 10, "Simulation Parameters:", 0, 1)
        pdf_obj.set_font("Arial", "", 10)
        pdf_obj.cell(0, 7, f"Fixed Temperature ({fixed_temp_option_param}): {fixed_temp_val_disp_param} {temp_unit_disp_param}", 0, 1)
        sim_temps_disp_pdf = data_df[temp_col_disp_name_param].unique().tolist() if not data_df.empty and temp_col_disp_name_param in data_df.columns else [] # Renamed variable
        pdf_obj.cell(0, 7, f"Varying Temp Range ({temp_col_disp_name_param}): {min(sim_temps_disp_pdf):.0f} to {max(sim_temps_disp_pdf):.0f}" if sim_temps_disp_pdf else 'N/A', 0, 1)
        pdf_obj.cell(0, 7, f"Wall Area: {area_param} m²", 0, 1); pdf_obj.cell(0, 7, f"Energy Cost: ${energy_cost_param:.3f}/kWh", 0, 1); pdf_obj.cell(0, 7, f"Operating Hours/Year: {operating_hours_param}", 0, 1); pdf_obj.ln(10)
        if plotly_fig_pdf:
            img_buffer_pdf = io.BytesIO() # Renamed variable
            try:
                plotly_fig_pdf.write_image(img_buffer_pdf, format="png", engine="kaleido", width=800, height=450); img_buffer_pdf.seek(0)
                pdf_obj.set_font("Arial", "B", 12); pdf_obj.cell(0, 10, "Heat Flux Comparison Graph", 0, 1, "C"); pdf_obj.ln(5)
                pdf_obj.image(img_buffer_pdf, x=(pdf_obj.w - 180) / 2, w=180)
                pdf_obj.ln( (180 / (800/450)) + 5 )
            except Exception as e_pdf_graph: print(f"Error PDF graph: {e_pdf_graph}"); pdf_obj.cell(0, 10, "[Graph gen error]", 0, 1, "C") # Renamed variable
        pdf_obj.set_font("Arial", "B", 12); pdf_obj.cell(0, 10, "Detailed Results Table", 0, 1, "C"); pdf_obj.ln(5)
        # Simplified PDF table for brevity
        pdf_table_cols_pdf = [fixed_temp_col_disp_name_param, temp_col_disp_name_param, "Heat Flux (W/m²)", "Annual Energy Cost ($)"] # Renamed variable
        pdf_table_cols_exist_pdf = [col for col in pdf_table_cols_pdf if col in data_df.columns] # Renamed variable
        if pdf_table_cols_exist_pdf:
            for material_name_pdf_table in data_df["Material"].unique(): # Renamed variable
                 pdf_obj.set_font("Arial", "B", 8); pdf_obj.cell(0,7, f"Material: {material_name_pdf_table}",0,1)
                 pdf_obj.set_font("Arial", "", 7)
                 for col_pdf_table in pdf_table_cols_exist_pdf: pdf_obj.cell(45, 5, col_pdf_table, 1, 0, "C") # Renamed variable
                 pdf_obj.ln()
                 for _, row_pdf_table in data_df[data_df["Material"] == material_name_pdf_table][pdf_table_cols_exist_pdf].iterrows(): # Renamed variable
                     for item_pdf_table in row_pdf_table: pdf_obj.cell(45, 5, str(round(item_pdf_table,2) if isinstance(item_pdf_table, float) else item_pdf_table), 1, 0, "C") # Renamed variable
                     pdf_obj.ln()
                 pdf_obj.ln(3)

        pdf_obj.set_font("Arial", "B", 12); pdf_obj.cell(0, 10, "Validation Summary", 0, 1, "C"); pdf_obj.ln(5)
        pdf_obj.set_font("Arial","",8)
        for msg_pdf_val in validation_msgs_param: pdf_obj.multi_cell(0,5,msg_pdf_val,0,"J") # Renamed variable

        try:
            pdf_buffer = io.BytesIO()
            pdf_obj.output(pdf_buffer)
            pdf_buffer.seek(0)
            return pdf_buffer.getvalue()
        except Exception as e_pdf_output: print(f"PDF Output Error: {e_pdf_output}"); return None # Renamed variable

    df_bytes_buffer_dl = io.BytesIO() # Renamed variable
    simulation_df_full_results.to_feather(df_bytes_buffer_dl)
    df_bytes_buffer_dl.seek(0)
    fig_json_str_report_dl = graph_results.to_json() if graph_results else None # Renamed variable

    excel_buffer_download = get_excel_report_cached_final(df_bytes_buffer_dl.getvalue(), temp_col_name_disp_results, fixed_col_name_disp_results) # Renamed variable
    if excel_buffer_download: st.download_button("Download Excel Report 📊", excel_buffer_download, "thermoPro_simulation_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key='excel_download_button_final') # Changed key
    else: st.warning("Excel report could not be generated.", icon="⚠️")

    pdf_fixed_temp_val_disp_dl = convert_temperature_value(fixed_temp_kelvin_sidebar, "K", current_run_temp_unit_results) # Renamed variable
    pdf_buffer_download = get_pdf_report_cached_final( # Renamed variable
        df_bytes_buffer_dl.getvalue(), fig_json_str_report_dl, temp_col_name_disp_results, fixed_col_name_disp_results,
        pdf_fixed_temp_val_disp_dl, current_run_temp_unit_results, st.session_state.get('fixed_temp_option_sidebar_select', 'N/A'),
        area_sidebar, energy_cost_per_kWh_sidebar, operating_hours_sidebar, validation_messages_results
    )
    if pdf_buffer_download: st.download_button("Download PDF Report 📄", pdf_buffer_download, "thermoPro_simulation_report.pdf", "application/pdf", key='pdf_download_button_final') # Changed key
    else: st.warning("PDF report could not be generated (check logs for details).", icon="⚠️")

elif not simulation_ready_main:
    st.info("Configure materials and simulation parameters in the sidebar to run the analysis.")
    current_material_group_fallback = st.session_state.get('material_group', []) # Renamed variable
    current_baseline_fallback = st.session_state.get('sidebar_baseline_material_main_select') # Renamed variable
    actual_temp_values_fallback = actual_temp_values_kelvin_sidebar # Renamed variable

    if not current_material_group_fallback or not any(m.get("name") for m in current_material_group_fallback): st.info("➡️ Define at least one named material.")
    elif not current_baseline_fallback or current_baseline_fallback not in [m.get("name") for m in current_material_group_fallback]: st.info("➡️ Select a valid Baseline Material.")
    elif not actual_temp_values_fallback: st.info("➡️ Ensure the temperature range includes at least one point.")