# Import necessary libraries
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the trained model and scaler
model = joblib.load('House_price_predictor.pkl')
scaler = joblib.load('features_standard_scaling.pkl')

# Function to make predictions
def make_prediction(features):
    return model.predict(np.array(features).reshape(1, -1))[0]

# Function to visualize feature values
def visualize_features(features):
    fig, ax = plt.subplots()
    feature_names = ['No of Bedrooms', 'Living Area (sqft)', 'Lot Size (sqft)', 
                     'House Area Above (sqft)', 'Area of Basement (sqft)', 
                     'Sqft Living 15', 'Sqft Lot 15', 'View Quality', 
                     'House Condition', 'House Grade']
    
    filtered_features = [features[i] for i in [1, 3, 4, 7, 10, 11, 12, 13, 14, 15]]
    
    ax.bar(feature_names, filtered_features)
    ax.set_title('Selected Feature Values')
    ax.set_ylabel('Value')
    ax.set_xlabel('Feature')
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Clear the previous plot
    if hasattr(visualize_features, 'canvas'):
        visualize_features.canvas.get_tk_widget().destroy()

    visualize_features.canvas = FigureCanvasTkAgg(fig, master=window)
    visualize_features.canvas.get_tk_widget().pack()
    visualize_features.canvas.draw()

# Function to predict and visualize
def predict_and_visualize():
    try:
        features = [float(entry.get()) for entry in entries]
        
        # Scale first 13 features and keep the last 4 unchanged
        scaled_features = scaler.transform([features[:13]])[0].tolist() + features[13:]
        
        predicted_price = make_prediction(scaled_features)
        result_var.set(f"Predicted House Price: ${predicted_price:.2f}")

        visualize_features(features)  # Visualize the original features
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all features.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
window = tk.Tk()
window.title("House Price Prediction")
window.geometry("1632x900")
window.configure(bg="#e8f4f8")

# Define colors and fonts
label_bg = "#d1e0e5"
button_bg = "#6699cc"
button_fg = "white"
result_bg = "#f2d7d9"
font_title = ("Helvetica", 14, "bold")
font_label = ("Helvetica", 10)

# Title
tk.Label(window, text="House Price Prediction Tool", font=font_title, bg=label_bg, padx=10, pady=10).pack(fill=tk.X)

# Frame for input fields
input_frame = tk.Frame(window, bg=label_bg, padx=20, pady=10)
input_frame.pack(pady=10)

# Feature labels and entry fields
labels = ["Year Sold", "No of Bedrooms", "No of Bathrooms", "Living Area (sqft)", "Lot Size (sqft)", 
          "No of Floors", "House Area Above (sqft)", "Area of Basement (sqft)", "Latitude", "Longitude",
          "Sqft Living 15", "Sqft Lot 15", "Renovated (yrs ago)", "View Quality", 
          "Waterfront", "House Condition", "House Grade"]

# Create labels and entry fields
entries = []
for i, label_text in enumerate(labels):
    tk.Label(input_frame, text=label_text + ":", font=font_label, bg=label_bg).grid(row=i//2, column=(i%2)*2, sticky="e", padx=(5, 10), pady=5)
    entry = tk.Entry(input_frame)
    entry.grid(row=i//2, column=(i%2)*2+1, padx=(5, 10), pady=5)
    entries.append(entry)

# Frame for buttons
button_frame = tk.Frame(window, bg="#e8f4f8")
button_frame.pack(pady=10)

# Prediction and reset buttons
tk.Button(button_frame, text="Predict and Visualize", command=predict_and_visualize, bg=button_bg, fg=button_fg, padx=10, pady=5).grid(row=0, column=0, padx=5, pady=5)
tk.Button(button_frame, text="Reset", command=lambda: [entry.delete(0, tk.END) for entry in entries], bg=button_bg, fg=button_fg, padx=10, pady=5).grid(row=0, column=1, padx=5, pady=5)

# Label to display prediction result
result_var = tk.StringVar()
tk.Label(window, textvariable=result_var, bg=result_bg, font=("Helvetica", 12), pady=10).pack(fill=tk.X, padx=10, pady=10)

# Run the application
window.mainloop()
# Run the application by running python app.py on your terminal