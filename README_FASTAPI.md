# PrivacyGuard API - FastAPI Version

This is the FastAPI migration of the PrivacyGuard API, designed for deployment on Railway hosting platform.

## 🚀 Features

- **FastAPI Framework**: High-performance async API framework
- **Google Custom Search API**: Primary search engine
- **APIFY Scraper**: Advanced web scraping with fallbacks
- **PII Extraction**: AI-powered personal information detection
- **Social Media Search**: Multi-platform social media discovery
- **Risk Assessment**: Comprehensive privacy risk analysis
- **Railway Ready**: Optimized for Railway deployment

## 📁 Project Structure

```
PrivacyGuard/
├── main.py                 # FastAPI application entry point
├── models.py              # Pydantic models for request/response validation
├── utils.py               # Utility functions and helper classes
├── core.py                # Core business logic and classes
├── routes/                # Modular route structure
│   ├── __init__.py
│   ├── search.py          # /search endpoint
│   ├── extract.py         # /extract endpoint
│   ├── health.py          # /apify/health, /apify/test endpoints
│   └── performance.py     # /performance/stats endpoint
├── apify_scraper.py       # APIFY scraper module (preserved)
├── requirements.txt       # Python dependencies
├── railway.json          # Railway deployment configuration
├── Procfile              # Alternative start command
└── env.example           # Environment variables template
```

## 🛠️ Installation & Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PrivacyGuard
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your actual API keys
   ```

5. **Run the application**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Railway Deployment

1. **Connect to Railway**
   - Go to [Railway.app](https://railway.app)
   - Connect your GitHub repository
   - Select this project

2. **Set Environment Variables**
   - In Railway dashboard, go to Variables tab
   - Add all variables from `env.example`

3. **Deploy**
   - Railway will automatically deploy on git push
   - Monitor deployment in Railway dashboard

## 🔧 API Endpoints

### Search Endpoints
- `GET /api/v1/search` - Search for person information
- `POST /api/v1/extract` - Extract PII from selected sources

### Health & Monitoring
- `GET /health` - Basic health check
- `GET /api/v1/apify/health` - APIFY-specific health check
- `GET /api/v1/apify/test` - Test APIFY configuration
- `GET /api/v1/performance/stats` - Performance statistics

### Documentation
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## 📊 API Usage Examples

### Search Request
```bash
curl "https://your-app.railway.app/api/v1/search?searchName=John%20Doe&includeSocial=true&maxResults=20"
```

### Extract Request
```bash
curl -X POST "https://your-app.railway.app/api/v1/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "searchName": "John Doe",
    "selectedUrls": ["https://example.com"],
    "selectedSocial": [{"url": "https://linkedin.com/in/johndoe", "platform": "linkedin"}]
  }'
```

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for PII extraction | Yes |
| `GOOGLE_CUSTOM_SEARCH_API_KEY` | Google Custom Search API key | Yes |
| `GOOGLE_CUSTOM_SEARCH_ENGINE_ID` | Google Custom Search Engine ID | Yes |
| `APIFY_API_KEY` | APIFY API key for web scraping | Yes |
| `PORT` | Server port (Railway sets this automatically) | No |

## 🚀 Performance Optimizations

- **Concurrent Processing**: Multiple URLs processed simultaneously
- **Connection Pooling**: Efficient HTTP connection management
- **Rate Limiting**: Built-in rate limiting for API calls
- **Caching**: Intelligent caching for repeated requests
- **Async/Await**: Full async support for better performance

## 🔒 Security Features

- **CORS Configuration**: Configurable cross-origin resource sharing
- **Input Validation**: Pydantic models for request validation
- **Error Handling**: Comprehensive error handling and logging
- **Rate Limiting**: Protection against abuse

## 📈 Monitoring & Logging

- **Health Checks**: Multiple health check endpoints
- **Performance Metrics**: Real-time performance statistics
- **Structured Logging**: Comprehensive logging for debugging
- **Error Tracking**: Detailed error reporting

## 🛠️ Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest
```

### Code Quality
```bash
# Install linting tools
pip install black flake8 mypy

# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## 🔄 Migration from Flask

This FastAPI version maintains full compatibility with the original Flask API:

- **Same Endpoints**: All original endpoints preserved
- **Same Logic**: Core business logic unchanged
- **Enhanced Performance**: Better async handling
- **Better Documentation**: Auto-generated API docs
- **Type Safety**: Pydantic models for validation

## 📞 Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the logs in Railway dashboard
3. Check health endpoints for system status

## 📄 License

This project is licensed under the MIT License.
