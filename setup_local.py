import os
import subprocess
import sys
import platform

def run_cmd(cmd, cwd=None):
    print(f">> Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False
    return True

def main():
    print("========================================")
    print("   🎓 Scholar AI - Local Setup Utility")
    print("========================================")
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend-functions")
    frontend_dir = os.path.join(root_dir, "frontend-angular")
    
    # 1. Backend Setup
    print("\n[1/3] Setting up Backend...")
    if not os.path.exists(os.path.join(backend_dir, "venv")):
        run_cmd(f"{sys.executable} -m venv venv", cwd=backend_dir)
    
    # Determine pip path
    if platform.system() == "Windows":
        pip_path = os.path.join(backend_dir, "venv", "Scripts", "pip")
    else:
        pip_path = os.path.join(backend_dir, "venv", "bin", "pip")
        
    run_cmd(f"{pip_path} install -r requirements.txt", cwd=backend_dir)
    
    # 2. Frontend Setup
    print("\n[2/3] Setting up Frontend...")
    print("Note: This requires Node.js and npm to be installed.")
    run_cmd("npm install --legacy-peer-deps", cwd=frontend_dir)
    
    # 3. Environment Check
    print("\n[3/3] Finalizing Environment...")
    if not os.path.exists(os.path.join(backend_dir, ".env")):
        print("Creating default .env for backend...")
        with open(os.path.join(backend_dir, ".env"), "w") as f:
            f.write("GEMINI_API_KEY=AIzaSyCMbDpybbi30oHf1sLHGD0dO2DIt0fagrI\n")
            f.write("DATABASE_DIR=database\n")
            f.write("FLASK_DEBUG=1\n")

    print("\n" + "="*40)
    print(" 🎉 SETUP COMPLETE!")
    print("="*40)
    print("\nTo start the application, run these in two separate terminals:")
    print("\nBACKEND:")
    if platform.system() == "Windows":
        print(f"  cd backend-functions && venv\\Scripts\\activate && python main.py")
    else:
        print(f"  cd backend-functions && source venv/bin/activate && python main.py")
        
    print("\nFRONTEND:")
    print(f"  cd frontend-angular && npm start")
    print("\nAccess the app at: http://localhost:4200")
    print("="*40)

if __name__ == "__main__":
    main()
