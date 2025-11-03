# WhatsApp Integration - Server Deployment Guide

## 📋 Project Overview
This is an AI-powered WhatsApp chatbot system for event management (Kalolsavam). It provides intelligent question-answering capabilities with PDF/Excel document processing, contextual conversations, and real-time event information retrieval.

---

## 🛠️ Technology Stack

### Core Framework & API
- **FastAPI** (v0.115.0) - High-performance async web framework
- **Uvicorn** (v0.32.0) - ASGI server for production deployment
- **Python 3.8+** - Programming language

### AI & Machine Learning
- **OpenAI API** (GPT-3.5-turbo & GPT-4) - Natural language processing and conversational AI
- **LangChain** (langchain-openai, langchain-community) - LLM orchestration framework
- **OpenAI Embeddings** - Text vectorization for semantic search
- **FAISS** (Facebook AI Similarity Search) - Vector database for document retrieval

### Messaging Integration
- **Twilio WhatsApp API** (v9.2.3) - WhatsApp Business API integration

### Database
- **MongoDB** (PyMongo v4.7.1) - NoSQL database for chat history and user conversations
- **DNS Python** (v2.0.0+) - Required for MongoDB Atlas (mongodb+srv://) connections

### Document Processing
- **PyMuPDF** (v1.23.8) - PDF text extraction
- **Pytesseract** (v0.3.10+) - OCR (Optical Character Recognition) for scanned PDFs
- **Pillow** (v10.0.0+) - Image processing library
- **Pandas** (v2.2.3) - Excel/CSV data manipulation
- **OpenPyXL** (v3.1.5) - Excel file handling (.xlsx)

### Utilities
- **python-dotenv** (v1.0.1) - Environment variable management
- **python-multipart** (v0.0.9) - File upload handling
- **requests** (v2.32.5+) - HTTP client library
- **certifi** (v2023.0.0+) - SSL certificate verification

---

## 💾 Server Requirements for 1000 Users

### Compute Resources

#### **Recommended Configuration**
- **CPU**: 4-8 vCPUs (Intel Xeon or AMD EPYC)
- **RAM**: 16-32 GB
- **Storage**: 50-100 GB SSD

#### **Detailed Memory Breakdown**

| Component | Memory Usage | Notes |
|-----------|-------------|-------|
| FastAPI Application | 200-500 MB | Base application footprint |
| FAISS Vector Store | 500 MB - 2 GB | Depends on document corpus size |
| Python Runtime | 100-200 MB | Python interpreter and libraries |
| Per Active User Session | 5-10 MB | Conversation context and state |
| Peak Concurrent Users (200) | 1-2 GB | Assuming 20% concurrency rate |
| System Overhead | 2-4 GB | OS, buffers, caching |
| **Total (Recommended)** | **16-32 GB RAM** | Includes safety margin |

### CPU Requirements
- **Light Load** (1-50 concurrent users): 2 vCPUs
- **Medium Load** (50-200 concurrent users): 4 vCPUs
- **Heavy Load** (200-500 concurrent users): 8 vCPUs
- **Peak Load** (500+ concurrent users): 12+ vCPUs or horizontal scaling

### Network Requirements
- **Bandwidth**: 10-50 Mbps for 1000 users (depends on traffic pattern)
- **Latency**: Low latency connection to:
  - OpenAI API servers
  - MongoDB Atlas (if cloud-hosted)
  - Twilio API servers

### Storage Requirements
- **Application Code**: ~50 MB
- **PDF Documents**: Variable (1-10 GB depending on corpus)
- **FAISS Vector Index**: 500 MB - 5 GB
- **Logs**: 10-50 MB/day (rotate regularly)
- **Total**: 50-100 GB SSD (recommended)

---

## 🗄️ MongoDB Requirements

### Database Configuration

#### **MongoDB Version**
- **Minimum**: MongoDB 4.4+
- **Recommended**: MongoDB 5.0+ or MongoDB Atlas (Cloud)

#### **Database Structure**
```
Database: Steam-Karnival
  └── Collection: whatsapp_chats
      ├── user_mobile_number (String)
      ├── user_timestamp (DateTime)
      ├── user_question (String)
      └── response (String)
```

#### **Storage Estimation (1000 Users)**

| Metric | Calculation | Estimated Size |
|--------|-------------|----------------|
| Average messages/user/day | 10 messages | - |
| Average message size | 500 bytes | - |
| Daily storage | 1000 users × 10 msgs × 500 bytes | ~5 MB/day |
| Monthly storage | 5 MB × 30 days | ~150 MB/month |
| Yearly storage (with indices) | 150 MB × 12 × 1.5 | ~2.5 GB/year |
| **Recommended MongoDB Storage** | - | **10-20 GB** |

#### **MongoDB Connection Modes**
The application supports both:
1. **MongoDB Atlas** (mongodb+srv://) - Cloud hosted, recommended
2. **Self-hosted MongoDB** (mongodb://) - On-premises deployment

#### **MongoDB Atlas Recommended Tier**
For 1000 users:
- **Tier**: M10 or M20 (General Purpose)
- **Storage**: 10-40 GB
- **RAM**: 2-4 GB
- **Estimated Cost**: $57-160/month (AWS/GCP pricing)

#### **Connection Settings**
- **Connection Timeout**: 10 seconds (configurable)
- **TLS/SSL**: Supported (required for Atlas)
- **Connection Pooling**: Enabled (PyMongo default)

### Indexing Requirements
```javascript
// Create index on user_mobile_number for faster queries
db.whatsapp_chats.createIndex({ "user_mobile_number": 1 })

// Create compound index for time-based queries
db.whatsapp_chats.createIndex({ "user_mobile_number": 1, "user_timestamp": -1 })
```

---

## 🔧 Environment Variables Required

Create a `.env` file with the following variables:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx

# Twilio WhatsApp Configuration
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

# MongoDB Configuration
MONGO_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority

# Optional: Application Settings
PORT=8000
HOST=0.0.0.0
```

---

## 📦 Installation & Deployment

### 1. System Dependencies (Ubuntu/Debian)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.8+
sudo apt install python3 python3-pip python3-venv -y

# Install Tesseract OCR (required for PDF OCR)
sudo apt install tesseract-ocr -y

# Install system libraries for image processing
sudo apt install libjpeg-dev libpng-dev -y
```

### 2. Application Setup
```bash
# Clone repository
cd /opt
git clone <your-repository-url> whatsapp-integration
cd whatsapp-integration

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
nano .env  # Edit with your credentials
```

### 3. Initialize Vector Store (if using PDF search)
```bash
# Place your PDF documents in data/pdfs/
# Run the PDF processing script
python pdf_ocr_extractor.py
```

### 4. Run the Application

#### Development Mode
```bash
uvicorn main_demo:app --reload --host 0.0.0.0 --port 8000
```

#### Production Mode
```bash
# Using uvicorn with multiple workers
uvicorn main_demo:app --host 0.0.0.0 --port 8000 --workers 4

# Or using Gunicorn with uvicorn workers (recommended)
pip install gunicorn
gunicorn main_demo:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 🚀 Production Deployment Recommendations

### Process Management
Use **systemd** or **supervisor** to manage the application:

#### Systemd Service File (`/etc/systemd/system/whatsapp-bot.service`)
```ini
[Unit]
Description=WhatsApp AI Chatbot
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/whatsapp-integration
Environment="PATH=/opt/whatsapp-integration/venv/bin"
ExecStart=/opt/whatsapp-integration/venv/bin/gunicorn main_demo:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable whatsapp-bot
sudo systemctl start whatsapp-bot
```

### Reverse Proxy (Nginx)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### SSL/TLS Configuration
```bash
# Install Certbot for Let's Encrypt
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

---

## 📊 Monitoring & Logging

### Application Logs
- FastAPI logs to stdout/stderr
- Configure log rotation with `logrotate`
- Consider centralized logging (ELK stack, Datadog, CloudWatch)

### Health Check Endpoint
The application should expose:
- `GET /health` - Application health status
- Monitor response time and uptime

### Resource Monitoring
- **CPU Usage**: Should stay below 70% under normal load
- **Memory Usage**: Monitor for memory leaks
- **Disk I/O**: Monitor for FAISS index access patterns
- **Network**: Track API call latency (OpenAI, Twilio, MongoDB)

---

## 🔒 Security Considerations

1. **API Keys**: Store in environment variables, never in code
2. **MongoDB**: Use authentication and TLS/SSL connections
3. **Firewall**: Restrict access to port 8000 (only via Nginx)
4. **Rate Limiting**: Implement rate limiting for API endpoints
5. **Input Validation**: Already handled by FastAPI
6. **HTTPS**: Mandatory for production (use Let's Encrypt)

---

## 🧪 Testing the Deployment

### Test API Endpoint
```bash
# Test the /ask endpoint
curl -X POST http://your-domain.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello, what programs are available?", "user_id": "test_user"}'
```

### Test WhatsApp Integration
Send a message to your Twilio WhatsApp number and verify:
1. Message is received by webhook
2. Response is generated
3. Reply is sent back via WhatsApp

---

## 📈 Scaling Strategies

### Horizontal Scaling (Multiple Instances)
- Use load balancer (Nginx, HAProxy, AWS ELB)
- Deploy multiple application instances
- Share MongoDB connection across instances
- Use Redis for session management if needed

### Vertical Scaling
- Increase CPU/RAM as user base grows
- Upgrade MongoDB tier (Atlas M20, M30, etc.)

### Performance Optimization
- Enable Redis caching for frequently accessed data
- Optimize FAISS index (use IVF or HNSW indices for large datasets)
- Implement CDN for static assets
- Use connection pooling for MongoDB

---

## 🆘 Troubleshooting

### Common Issues

1. **MongoDB Connection Timeout**
   - Check firewall rules
   - Verify MongoDB Atlas IP whitelist
   - Test connection string manually

2. **OpenAI API Rate Limits**
   - Implement exponential backoff
   - Consider upgrading OpenAI tier
   - Cache frequent responses

3. **High Memory Usage**
   - Monitor FAISS vector store size
   - Reduce number of uvicorn workers
   - Implement pagination for chat history

4. **Slow Response Times**
   - Check OpenAI API latency
   - Optimize FAISS search parameters (reduce k)
   - Enable caching layer

---

## 📞 Support & Maintenance

### Regular Maintenance Tasks
- **Daily**: Monitor logs and error rates
- **Weekly**: Review system resource usage
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Review and optimize database indices

### Backup Strategy
- **MongoDB**: Enable automated backups (Atlas provides this)
- **FAISS Index**: Backup vector store files weekly
- **Application Code**: Use version control (Git)
- **Environment Config**: Securely backup .env file

---

## 📄 API Documentation

Once deployed, access interactive API documentation at:
- **Swagger UI**: `http://your-domain.com/docs`
- **ReDoc**: `http://your-domain.com/redoc`

---

## 🎯 Summary Checklist

- [ ] Provision server (16-32 GB RAM, 4-8 vCPUs, 50-100 GB SSD)
- [ ] Set up MongoDB Atlas (M10/M20 tier) or self-hosted MongoDB
- [ ] Install system dependencies (Python, Tesseract OCR)
- [ ] Install Python packages from requirements.txt
- [ ] Configure environment variables (.env file)
- [ ] Initialize FAISS vector store from PDFs
- [ ] Set up Nginx reverse proxy with SSL
- [ ] Configure systemd service for auto-restart
- [ ] Set up monitoring and logging
- [ ] Test endpoints and WhatsApp integration
- [ ] Configure automated backups
- [ ] Document access credentials securely

---

**Prepared By**: AI Assistant  
**Last Updated**: November 3, 2025  
**Version**: 1.0

