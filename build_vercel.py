"""
build_vercel.py
Build script that bundles the Iris EDA Streamlit Dashboard into an ultra-fast
Stlite (Streamlit on WebAssembly) static package for 1-click Vercel deployment.
"""

import os
import json

def build_vercel_bundle():
    print("Building Vercel bundle...")
    
    # Read core files
    files_to_bundle = {
        "streamlit_app.py": "streamlit_app.py",
        "app.py": "streamlit_app.py",
        "data_engine.py": "data_engine.py",
        "visualizations.py": "visualizations.py",
        "styles.py": "styles.py",
        "database.py": "database.py",
        "data/iris.csv": "data/iris.csv",
        ".streamlit/config.toml": ".streamlit/config.toml",
        "assets/iris_logo.svg": "assets/iris_logo.svg",
    }
    
    bundled_files = {}
    for virtual_path, local_path in files_to_bundle.items():
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                bundled_files[virtual_path] = f.read()
        else:
            print(f"Warning: {local_path} not found")
            
    files_json = json.dumps(bundled_files)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, shrink-to-fit=no" />
    <title>Iris Botanical Analytics Lab | Deep Tech Obsidian</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><circle cx=%2250%22 cy=%2250%22 r=%2240%22 fill=%22%2338BDF8%22/></svg>">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.63.1/build/stlite.css" />
    <style>
      body, html {{
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        background-color: #0B0F17;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      }}
      #root {{
        width: 100%;
        height: 100%;
      }}
      .loading-screen {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100vh;
        color: #F8FAFC;
        background: #0B0F17;
        text-align: center;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      .spinner {{
        width: 48px;
        height: 48px;
        border: 4px solid #1E293B;
        border-top: 4px solid #38BDF8;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 16px;
      }}
      @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
      }}
    </style>
  </head>
  <body>
    <div id="root">
      <div class="loading-screen" id="loading">
        <div class="spinner"></div>
        <h2 style="margin: 0 0 8px 0; font-size: 1.3rem; color: #F8FAFC;">Loading Iris Botanical Analytics Lab...</h2>
        <p style="margin: 0; color: #94A3B8; font-size: 0.95rem;">
          Initializing NumPy Linear Algebra &amp; Matplotlib Engine via WebAssembly
        </p>
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.63.1/build/stlite.js"></script>
    <script>
      const bundledFiles = {files_json};
      
      stlite.mount(
        {{
          requirements: ["numpy", "pandas", "matplotlib"],
          entrypoint: "streamlit_app.py",
          files: bundledFiles,
          streamlitConfig: {{
            "theme.base": "dark",
            "theme.primaryColor": "#38BDF8",
            "theme.backgroundColor": "#0B0F17",
            "theme.secondaryBackgroundColor": "#131B2E",
            "theme.textColor": "#F8FAFC"
          }}
        }},
        document.getElementById("root")
      ).then(() => {{
        const loader = document.getElementById("loading");
        if (loader) loader.style.display = "none";
      }}).catch((err) => {{
        const loader = document.getElementById("loading");
        if (loader) {{
          loader.innerHTML = `
            <div style="max-width: 500px; padding: 2rem; background: #fff; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
              <h2 style="color: #dc2626; margin-bottom: 0.5rem;">Application Failed to Load</h2>
              <p style="color: #475569; font-size: 0.9rem;">${{err.message || err}}</p>
              <button onclick="location.reload()" style="margin-top: 1rem; padding: 0.6rem 1.2rem; background: #2563eb; color: #fff; border: none; border-radius: 6px; font-weight: 600; cursor: pointer;">
                Reload Page
              </button>
            </div>
          `;
        }}
      }});
    </script>
  </body>
</html>
"""

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print("Vercel deployment bundle generated successfully: index.html")

if __name__ == "__main__":
    build_vercel_bundle()
