## Requirements
- **macOS** (or Linux/Windows with minor adjustments)
- **Python 3.10+**
- **Node.js 18+** and **npm**
- **Git** (optional, for version control)

---

## Project Structure
- `research-backend/` — Python backend (API server)
- `src/` — React frontend (user interface)
- `public/` — Static files for frontend
- `requirements.txt` — Python dependencies
- `package.json` — Node.js dependencies

---

## Initial Setup

### 1. Open Terminal
- Press `Cmd + Space`, type `Terminal`, and press `Enter`.

### 2. Navigate to the Project Folder
Replace `/path/to/research-main` with the actual path if different.
```sh
cd ~/Downloads/research-main
```

### 1. Install Python Dependencies
```sh
cd research-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Node.js Dependencies
Open a **new terminal tab** (press `Cmd + T`), then:
```sh
cd ~/Downloads/research-main
npm install
```

---

## Running the Backend (Python)
1. In the terminal (with the virtual environment activated):
```sh
cd research-backend
source venv/bin/activate
python app.py
```
- The backend server should start, usually on `http://127.0.0.1:5000` or similar.

---

## Running the Frontend (React)
1. In a **new terminal tab**:
```sh
cd ~/Downloads/research-main
npm start
```
