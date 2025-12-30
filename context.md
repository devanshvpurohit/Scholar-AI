# 🎓 Scholar AI - Project Context

## 📖 Overview
**Scholar AI** is an intelligent study assistant web application. It allows users to upload study materials (PDFs, Docs, Audio files) and automatically generates:
- 📝 **Summaries**
- 🗂️ **Flashcards**
- ❓ **Quizzes**

The application is designed for modern students, featuring a sleek "glassmorphism" UI and leveraging Google's **Gemini 1.5 Flash** for rapid content generation.

---

## 🛠️ Tech Stack

### **Frontend**
- **Framework:** Angular 18+ (Standalone Components)
- **Styling:** SCSS, Modern Dark Mode, Inter Font
- **Authentication:** Firebase Authentication (Email/Password & Google Sign-In)
- **Hosting:** Vercel (preferred) or Firebase Hosting

### **Backend**
- **Framework:** Python Flask (Standard WSGI)
- **AI Model:** Google Gemini 1.5 Flash (Optimized for speed)
- **Speech-to-Text:** Google Cloud Speech-to-Text (with fallback/direct audio handling)
- **Processing:** `pypdf` (PDF), `python-docx` (Word), `mutagen` (Audio duration)
- **Deployment:** Vercel Serverless Functions (`@vercel/python`) or Google Cloud Run

---

## 📂 Project Structure

```text
Scholar-AI/
├── frontend-angular/          # Angular Single Page Application
│   ├── src/
│   │   ├── app/
│   │   │   ├── components/    # (Home, Login, Guide)
│   │   │   ├── services/      # (Auth, API)
│   │   │   ├── environments/  # (Firebase Config)
│   │   └── ...
│   ├── firebase.json          # Firebase Hosting Config
│   └── package.json
│
├── backend-functions/         # Flask API
│   ├── main.py                # Main Application Logic (Endpoints)
│   ├── requirements.txt       # Python Dependencies
│   └── Dockerfile             # Container config for Cloud Run
│
├── deploy.sh                  # Script for Firebase+Cloud Run Deployment
├── vercel.json                # Vercel Deployment Config
└── context.md                 # Project Documentation (This file)
```

---

## 🔑 Key Features & Implementation Details

1.  **Single-Shot AI Generation:**
    - To support Vercel's 10-second timeout on free tier, the backend uses a **unified prompt** to generate the Title, Summary, Flashcards, and Quiz in a SINGLE API call to Gemini.
    - **Optimization:** Switched to `gemini-1.5-flash` for lowest latency.

2.  **Authentication:**
    - Frontend: Uses `@angular/fire` to handle login.
    - Backend: Verifies Firebase ID tokens using `firebase-admin`.
    - **Local Dev:** Includes a "Demo User" fallback if local Google Credentials aren't set up, preventing auth errors during development.

3.  **File Handling:**
    - Uploads are processed in-memory (or temp storage) for immediate extraction.
    - Large audio files (>55s) trigger robust GCS-based transcription (requires GCP credentials), while shorter ones use synchronous recognition.

---

## 🚀 Environment Variables

### **Backend (.env)**
```bash
GEMINI_API_KEY=AIzaSy...           # Required: Google Gemini API Key
GOOGLE_APPLICATION_CREDENTIALS=... # Optional: Path to GCP Service Account (for Speech-to-Text)
```

### **Frontend (src/environments/environment.ts)**
User-specific Firebase configuration:
```typescript
firebase: {
  apiKey: "...",
  authDomain: "...",
  projectId: "...",
  ...
}
```

---

## 📜 Deployment

### **Option 1: Vercel (Recommended)**
- **Config:** `vercel.json` handles routing.
- **Frontend:** Deploys as a static site.
- **Backend:** Deploys `main.py` as a Serverless Function.
- **Setup:** Import repo, set `GEMINI_API_KEY` env var, deploy.

### **Option 2: Firebase + Cloud Run**
- **Script:** Run `./deploy.sh`
- **Requires:** Google Cloud Billing enabled (for Cloud Build/Run).
- **Architecture:** Frontend on Firebase Hosting -> Rewrites `/api` -> Cloud Run Service.

---

## 📝 Recent Updates
- **Refactor:** Removed `functions-framework` dependency; standard Flask app now.
- **Fixes:** Added local auth fallback, updated deprecated Gemini model names, and fixed Angular peer dependency issues.
