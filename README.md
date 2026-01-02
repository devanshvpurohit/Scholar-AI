# 🎓 AI Learning Assistant

A full-stack AI-powered study tool that transforms various content formats into personalized learning materials including summaries, flashcards, and quizzes.

## 🏗️ Architecture

| Layer | Technology | Deployment |
|-------|------------|------------|
| **Frontend** | Angular 21 | Firebase Hosting |
| **Backend API** | Google Functions Framework (Python) | Cloud Run |
| **Authentication** | Firebase Auth | - |
| **AI/ML** | Google Gemini Pro | - |
| **Speech-to-Text** | Google Cloud Speech API | - |
| **Storage** | Google Cloud Storage | - |

## 📁 Project Structure

```
Scholar-AI/
├── frontend-angular/           # Angular SPA
│   ├── src/
│   │   ├── app/
│   │   │   ├── components/     # UI Components
│   │   │   │   ├── auth/       # Login/Signup
│   │   │   │   ├── home/       # File upload page
│   │   │   │   └── guide/      # Study guide view
│   │   │   ├── services/       # API & Auth services
│   │   │   ├── app.config.ts   # App configuration
│   │   │   ├── app.routes.ts   # Routing
│   │   │   └── app.ts          # Root component
│   │   ├── environments/       # Environment configs
│   │   └── index.html
│   ├── firebase.json           # Firebase Hosting config
│   ├── angular.json
│   └── package.json
│
├── backend-functions/          # Cloud Run Backend
│   ├── main.py                 # API endpoints
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Container config
│   └── .env.example            # Environment template
│
└── README.md
```

## 🛠️ Easy Setup (Recommended)

To set up the entire project (backend venv, dependencies, and environment) automatically on any machine, run:

```bash
python setup_local.py
```

This script will:
- Create a Python virtual environment.
- Install all backend dependencies.
- Install all frontend dependencies (requires Node.js/npm).
- Generate a default `.env` if one doesn't exist.

---

## 🚀 Quick Run (Local Development)


1. **Start Backend**:
   ```bash
   cd backend-functions
   source venv/bin/activate
   python main.py
   ```
2. **Start Frontend**:
   ```bash
   cd frontend-angular
   npm start
   ```
3. **Access**: Open `http://localhost:4200`


### 2. Configure Firebase

1. Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable Authentication (Email/Password and Google Sign-in)
3. Get your Firebase config from Project Settings

### 3. Frontend Setup

```bash
cd frontend-angular

# Install dependencies
npm install --legacy-peer-deps

# Update environment files with your Firebase config
# Edit: src/environments/environment.ts
# Edit: src/environments/environment.prod.ts

# Run locally
npm start
```

### 4. Backend Setup

```bash
cd backend-functions

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Run locally
python main.py
```

---

## 🦾 Agentic Study Features

Scholar AI is now equipped with an **Agentic Study System**:

- ✅ **Study Goal Personalization**: Define specific goals for your material (e.g., "Explain like I'm 5").
- ✅ **Dynamic Progress Tracking**: Mark sessions as completed to keep track of your learning.
- ✅ **AI Motivational Nudges**: Get personalized encouragement based on your completion status.
- ✅ **Intelligent Replanning**: Missed a day? The AI agent can automatically restructure your remaining schedule based on your progress and reasoning.
- ✅ **Difficulty Tagging**: Key topics are tagged (Easy/Medium/Hard) for prioritizing your study.
- ✅ **Revision Slots**: Automatic insertion of revision sessions for harder topics to ensure retention.

---

## 🎨 Branding & UX

- **Modern SVG Branding**: A new stylized "Scholar AI" logo replaced the generic emoji.
- **Guest Access**: For rapid local development, use the "Continue as Guest" feature to bypass Firebase Login.
- **Enhanced UI**: Refined study guide view with collapsible sections and interactive timelines.


---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/upload` | Upload file + goals and generate guide |
| GET | `/api/guide/:id` | Get study guide data |
| PUT | `/api/guide/:id/progress` | Update session completion status |
| POST | `/api/guide/:id/replan` | Re-generate schedule based on missed tasks |
| POST | `/api/motivation` | Get AI-generated encouragement |
| GET | `/api/guides` | List all user guides |
| DELETE | `/api/guide/:id` | Delete specific guide |

---

## 🔐 Firebase Authentication

The app supports:
- **Email/Password** authentication
- **Google Sign-in**

Firebase ID tokens are verified in the backend using Firebase Admin SDK.

---

## 📦 Supported File Formats

| Format | Extension | Processing |
|--------|-----------|------------|
| PDF | .pdf | Text extraction |
| Word Document | .docx | Text extraction |
| Text | .txt, .md, .html | Direct read |
| Audio | .mp3, .wav | Speech-to-text |
| Video | .mp4 | Audio extraction → Speech-to-text |

---

## 🛠️ Development

### Frontend Development

```bash
cd frontend-angular
npm start
# Opens at http://localhost:4200
```

### Backend Development

```bash
cd backend-functions
python main.py
# Runs at http://localhost:8080
```

### Testing the API

```bash
# Health check
curl http://localhost:8080/api/health

# Upload a file
curl -X POST http://localhost:8080/api/upload \
  -F "file=@document.pdf"
```

---

## 📝 Key Features

- ✅ **Agentic Autoplan**: AI-driven schedule restructuring.
- ✅ **Multi-format upload** with goal-based generation.
- ✅ **Interactive flashcards** & scored quizzes.
- ✅ **Export to DOCX** (Quiz, Summary, Flashcards).
- ✅ **Firebase Auth** (Legacy) + **Guest Mode** (New).
- ✅ **Responsive design** (Mobile optimized).
- ✅ **Personalized Study Tips** section.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments
- Developed by **Devansh V Purohit**
- Google Gemini for AI capabilities
- Firebase for authentication and hosting
- Angular team for the framework

---
```
