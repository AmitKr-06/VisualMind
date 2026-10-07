# VisualMind

VisualMind is a multimodal RAG (Retrieval-Augmented Generation) API for studying from PDF textbooks. Ask a question in text, or upload a photo or sketch of a diagram, and VisualMind answers from your study material, shows the matching figures, and politely declines questions that fall outside the documents.

The service runs as a FastAPI app in a Docker container and is deployed on AWS EC2 with an automated CI/CD pipeline through GitHub Actions.

## Features

- **Text question answering** grounded in the indexed PDF content, with the source chunks used for each answer.
- **Image queries**: upload a phone photo or hand-drawn sketch of a diagram and VisualMind finds the matching figure in the textbooks.
- **Hybrid retrieval**: dense embeddings plus BM25 keyword search, followed by a cross-encoder reranker for higher precision.
- **Figure matching** with CLIP so answers can come with the relevant diagrams.
- **Off-topic handling**: questions outside the study material are detected and declined instead of answered with made-up content.
- **Reflection step**: the generated answer is checked against the retrieved context before it is returned.
- **Model fallback chain**: several Gemini models are tried in order, with Groq as a fallback provider.
- **Frozen pipeline configuration**: retrieval and routing settings are loaded from versioned config files, so results are reproducible.
- **Production-ready serving**: health check endpoint, models loaded once at startup, and automatic container restarts.

## How it works

Each request flows through a LangGraph workflow made of five nodes:

```
router -> retrieval -> answer -> reflection -> responder
```

| Node | Role |
|---|---|
| Router | Decides whether the request is a text question, an image query, or off-topic |
| Retrieval | Finds the best text chunks (hybrid search + reranker) and matching figures (CLIP) |
| Answer | Generates a grounded answer with the Gemini model chain, falling back to Groq |
| Reflection | Verifies the answer is supported by the retrieved context |
| Responder | Formats the final response with sources and figures |

## Tech stack

| Area | Technology |
|---|---|
| API | FastAPI, Uvicorn |
| Orchestration | LangGraph, LangChain |
| Embeddings | `BAAI/bge-base-en-v1.5` (sentence-transformers) |
| Reranker | `BAAI/bge-reranker-base` |
| Keyword search | rank-bm25 |
| Vector search | FAISS |
| Figure matching | CLIP |
| PDF processing | PyMuPDF |
| LLMs | Google Gemini (primary), Groq (fallback) |
| Container | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Cloud | AWS EC2, S3, IAM |

## Quick start (local)

### Prerequisites

- Docker and Docker Compose
- API keys for Gemini and Groq
- The `data/` folder (processed chunks, embeddings, extracted figures, results)

### 1. Clone the repository

```bash
git clone https://github.com/AmitKr-06/VisualMind.git
cd VisualMind
```

### 2. Create the `.env` file

Create a `.env` file in the project root with your keys. Use `NAME=value` with no spaces around `=`.

```env
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
```

`.env` is read at runtime by Docker Compose. It is never copied into the image and must never be committed.

### 3. Add the data

Place your prepared `data/` folder in the project root. The compose file mounts these subfolders into the container:

| Folder | Purpose | Access |
|---|---|---|
| `data/raw/documents` | Source PDFs | read-only |
| `data/processed` | Cleaned pages and text chunks | read-only |
| `data/extracted` | Extracted figures and manifests | read-only |
| `data/embeddings` | Precomputed embedding files | read-only |
| `data/eval` | Evaluation questions and images | read-only |
| `data/results` | Frozen configs and experiment outputs | read-write |
| `data/vision_cache` | Cached image analysis results | read-write |
| `data/uploads` | User-uploaded images | read-write |

### 4. Build and run

```bash
docker compose up --build
```

The first start downloads the embedding and reranker models, so allow a few minutes. When you see `Application startup complete`, the API is ready.

### 5. Try it

| URL | What it is |
|---|---|
| `http://localhost:8000/docs` | Interactive Swagger UI to try every endpoint |
| `http://localhost:8000/health` | Health check used by Docker and the deploy pipeline |
| `http://localhost:8000/config` | The active pipeline configuration |

## Configuration

At startup the app builds its configuration from the frozen config file in `data/results` and merges in the latest pipeline handoff files. The active values are visible at `/config`:

| Setting | Meaning |
|---|---|
| `chunk_file` | Which chunked text file the retriever uses |
| `k_chunks` | Number of text chunks passed to the answer step |
| `n_figures` | Number of matching figures returned |
| `tau_text` | Relevance threshold for answering from text |
| `dense_model` / `reranker_model` | Embedding and reranking models |
| `answer_model_chain` | Gemini models tried in order |
| `fallback_provider` | Provider used if the Gemini chain fails |

The container runs on CPU only (`CUDA_VISIBLE_DEVICES` is empty), so no GPU is needed.

## Deployment on AWS

VisualMind runs on a single EC2 instance. Code ships as a Docker image through Docker Hub, while large data files and secrets stay out of the image.

### Architecture

```
Developer  --git push-->  GitHub  --Actions-->  Docker Hub  --pull-->  EC2 (Docker)
                                                                          |
                                                              S3 (data) --+-- IAM role
```

| Component | Purpose |
|---|---|
| GitHub Actions | Lints, builds the image, pushes it to Docker Hub, then deploys |
| Docker Hub | Stores the `amiitkr/visualmind` image |
| EC2 | Runs the container with Docker Compose |
| S3 | Stores the `data/` folder |
| IAM role | Gives the EC2 instance read-only access to S3, so no AWS keys live on the server |
| Elastic IP | Gives the server a fixed public address |

### AWS setup

1. **IAM user**: create an admin user for console and CLI work, enable MFA, and stop using the root account.
2. **S3 bucket**: create a private bucket with "Block all public access" turned on, then upload the data:
   ```bash
   aws s3 sync ./data s3://<your-bucket>/data
   ```
3. **IAM role for EC2**: create a role for the EC2 service with `AmazonS3ReadOnlyAccess`, plus `AmazonSSMManagedInstanceCore` if you want Session Manager access.
4. **EC2 instance**:
   - Ubuntu Server 24.04 LTS
   - An instance type with **8 GiB RAM** (for example `m7i-flex.large`). Smaller instances cannot hold the embedding model, reranker and CLIP model at once.
   - 30 GiB gp3 storage
   - The IAM role attached as the instance profile
   - Security group: SSH (22) and Custom TCP 8000 (restrict 8000 to your own IP until the API is protected)
5. **Elastic IP**: allocate one and associate it with the instance.

### Server setup (one time)

Connect to the instance (SSH, EC2 Instance Connect, or Session Manager), then:

```bash
# Install Docker and the AWS CLI
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
sudo snap install aws-cli --classic
sudo su - ubuntu

# Download the data from S3 (uses the instance role, no keys needed)
mkdir -p ~/visualmind && cd ~/visualmind
aws s3 sync s3://<your-bucket>/data ./data --region us-east-1

# Add secrets and the compose file
nano .env                 # your API keys
nano docker-compose.yml   # same compose file as local, with: image: amiitkr/visualmind:latest

# Start the app
docker compose pull
docker compose up -d
```

Open `http://<elastic-ip>:8000/docs` to confirm the service is running.

### CI/CD pipeline

Every push to `main` runs `.github/workflows/ci.yaml`:

1. **Lint**: `ruff` checks `src/` for serious errors.
2. **Build and push**: builds the Docker image and pushes `latest` and a commit-SHA tag to Docker Hub.
3. **Deploy**: connects to the EC2 instance over SSH, runs `docker compose pull` and `docker compose up -d`, then waits for `/health` to pass.

Pull requests run lint and a test build only. They never push or deploy.

Add these under **Settings > Secrets and variables > Actions** in the GitHub repository:

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | A Docker Hub access token with read/write permission |
| `EC2_HOST` | The Elastic IP of the instance |
| `EC2_USER` | `ubuntu` |
| `EC2_SSH_KEY` | The full contents of the EC2 `.pem` key file |

The deploy job connects from GitHub's servers, so the SSH rule in the security group must allow GitHub's addresses (in practice `0.0.0.0/0`, protected by the key pair, since password login is disabled).

### Updating the data

Data lives in S3 and is pulled onto the server, not baked into the image. To publish new data:

```bash
aws s3 sync ./data s3://<your-bucket>/data          # from your computer
aws s3 sync s3://<your-bucket>/data ~/visualmind/data   # on the server
docker compose restart visualmind-api
```

## Security notes

- `.env`, `data/` and `*.pem` must stay out of Git (keep them in `.gitignore`).
- API keys are supplied at runtime and are not stored in the Docker image.
- The S3 bucket is private, and the server reads it through an IAM role instead of stored keys.
- Restrict port 8000 to trusted IPs, or add API-key authentication and rate limiting before exposing the service publicly. Otherwise anyone with the URL can use your Gemini and Groq quota.
- Rotate any key that was ever exposed, and use Docker Hub access tokens instead of your password.

## Cost tips

- Stop the EC2 instance when it is not in use. A stopped instance does not incur compute charges.
- An Elastic IP that is not attached to a running instance can incur a small hourly charge.
- Set an AWS Budget alert so unexpected charges are caught early.

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| Container exits at startup with a config `KeyError` | The frozen config or handoff files in `data/results` are missing or empty. Check them with `GET /config` locally. |
| Container is killed or restarts repeatedly | The instance has too little memory. Use an instance with at least 8 GiB RAM. |
| `permission denied` writing to `results`, `uploads` or `vision_cache` | Make the folders writable on the host: `sudo chmod -R 777 data/results data/uploads data/vision_cache` |
| `aws s3 sync` returns `AccessDenied` on EC2 | The IAM role is not attached to the instance. Attach it under **Actions > Security > Modify IAM role**. |
| SSH from your computer times out | Your network may block port 22, or your IP changed. Update the security group rule, or use EC2 Instance Connect / Session Manager. |
| `/docs` does not load from the browser | Port 8000 only allows the IP set in the security group. Update the rule to your current IP. |
| Deploy job fails with `no configuration file provided` | `docker-compose.yml` is missing in `~/visualmind` on the server. |

## License

Add your license here.