# PrivacyGuard API - Railway Deployment

A FastAPI-based backend service for PII detection and privacy protection, optimized for Railway deployment.

## 🚀 Quick Start

### Prerequisites
- Railway account
- GitHub repository with this code
- Required API keys (see Environment Variables section)

### Deployment Steps

1. **Fork/Clone this repository**
   ```bash
   git clone <your-repo-url>
   cd PrivacyGuard
   ```

2. **Connect to Railway**
   - Go to [Railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure Environment Variables**
   - In Railway dashboard, go to your project
   - Navigate to "Variables" tab
   - Add the following environment variables:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_CUSTOM_SEARCH_API_KEY=your_google_custom_search_api_key_here
   GOOGLE_CUSTOM_SEARCH_ENGINE_ID=your_google_custom_search_engine_id_here
   APIFY_API_KEY=your_apify_api_key_here
   PORT=8000
   ```

4. **Deploy**
   - Railway will automatically detect the FastAPI application
   - The deployment will start automatically
   - Monitor the build logs for any issues

## 📋 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for PII detection | ✅ |
| `GOOGLE_CUSTOM_SEARCH_API_KEY` | Google Custom Search API key | ✅ |
| `GOOGLE_CUSTOM_SEARCH_ENGINE_ID` | Google Custom Search Engine ID | ✅ |
| `APIFY_API_KEY` | APIFY API key for web scraping | ✅ |
| `PORT` | Server port (Railway sets this automatically) | ❌ |

## 🔧 API Endpoints

### Core Endpoints
- `GET /` - API information and status
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `GET /redoc` - Alternative API documentation

### Search Endpoints
- `GET /search` - Search for person information
- `GET /search/search` - Alternative search endpoint

### Extraction Endpoints
- `POST /extract` - Extract PII from selected URLs
- `POST /extract/extract` - Alternative extraction endpoint

### Health & Performance
- `GET /apify/health` - APIFY service health check
- `GET /apify/test` - Test APIFY configuration
- `GET /performance/stats` - Performance statistics

## 🏗️ Architecture

```
PrivacyGuard API
├── main.py              # FastAPI application entry point
├── core.py              # Core business logic and classes
├── models.py            # Pydantic models for request/response
├── utils.py             # Utility functions
├── apify_scraper.py     # APIFY web scraping integration
├── routes/              # API route modules
│   ├── search.py        # Search functionality
│   ├── extract.py       # PII extraction
│   ├── health.py        # Health checks
│   └── performance.py   # Performance monitoring
├── requirements.txt     # Python dependencies
├── railway.json         # Railway configuration
├── Procfile            # Process configuration
└── Dockerfile          # Docker configuration
```

## 🔍 Features

- **PII Detection**: Advanced PII detection using OpenAI GPT models
- **Web Scraping**: Robust web scraping with APIFY integration and fallback
- **Search Integration**: Google Custom Search API integration
- **Concurrent Processing**: Optimized for high-performance concurrent requests
- **Health Monitoring**: Comprehensive health checks and performance monitoring
- **CORS Support**: Configured for frontend integration

## 🛠️ Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp env.example .env
# Edit .env with your API keys

# Run the application
python main.py
```

### Testing
```bash
# Test the API
curl http://localhost:8000/health

# Test search functionality
curl "http://localhost:8000/search?searchName=John%20Doe&maxResults=10"

# Test PII extraction
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{"searchName": "John Doe", "selectedUrls": ["https://example.com"]}'
```

## 📊 Monitoring

### Health Checks
- **Basic Health**: `GET /health`
- **APIFY Health**: `GET /apify/health`
- **Performance Stats**: `GET /performance/stats`

### Logs
- Railway provides built-in logging
- Check Railway dashboard for application logs
- Monitor error rates and performance metrics

## 🔒 Security

- Environment variables are securely managed by Railway
- CORS is configured for specific frontend domains
- Input validation using Pydantic models
- Rate limiting and timeout configurations

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**
   - Check that all dependencies are in requirements.txt
   - Verify Python version compatibility
   - Check build logs in Railway dashboard

2. **Runtime Errors**
   - Verify all environment variables are set
   - Check application logs for specific error messages
   - Ensure API keys are valid and have proper permissions

3. **CORS Issues**
   - Verify frontend domain is in allowed origins
   - Check that credentials are properly configured

### Support
- Check Railway documentation: https://docs.railway.app
- Review FastAPI documentation: https://fastapi.tiangolo.com
- Check application logs in Railway dashboard

## 📈 Performance

The application is optimized for Railway deployment with:
- Concurrent request processing
- Efficient memory usage
- Proper connection pooling
- Rate limiting and timeout configurations
- Health check endpoints for monitoring

## 🔄 Updates

To update the application:
1. Push changes to your GitHub repository
2. Railway will automatically detect changes
3. A new deployment will be triggered
4. Monitor the deployment logs for any issues

---

**Note**: This application is designed to work with a frontend hosted on Lovable. Make sure to configure the CORS settings if using a different frontend platform.
