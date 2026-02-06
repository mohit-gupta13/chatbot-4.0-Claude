# B2B Travel Agency AI Chatbot

Production-ready AI chatbot for travel agencies to query flight and hotel bookings using natural language.

## 🚀 Tech Stack

- **LLM**: Claude Sonnet 4.5 via AWS Bedrock
- **Vector Database**: Chroma (self-hosted, open-source)
- **Embeddings**: Sentence-Transformers (all-mpnet-base-v2)
- **RAG Framework**: LangChain (Python)
- **Backend**: FastAPI

## 📋 Prerequisites

1. **AWS Account** with Bedrock access
   - Request access to Claude Sonnet 4.5 in your region
   - IAM role with `bedrock:InvokeModel` permission

2. **Python 3.9+**

3. **Booking Data**: Excel file with booking records (`Existing Booking.xlsx` or `flight Booking.xlsx`)

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- boto3 (AWS SDK)
- langchain & langchain-aws
- chromadb
- sentence-transformers
- fastapi, uvicorn
- pandas, openpyxl

### 2. Configure AWS Credentials

**Option A: Environment Variables**

Edit `.env` file:
```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
BEDROCK_MODEL_ID=anthropic.claude-sonnet-4-20250514
```

**Option B: AWS Credentials File**

Create `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = your-access-key-id
aws_secret_access_key = your-secret-access-key
region = us-east-1
```

### 3. Verify AWS Bedrock Access

Test your AWS Bedrock connection:
```bash
aws bedrock list-foundation-models --region us-east-1
```

Ensure `anthropic.claude-sonnet-4-20250514` is listed.

### 4. Prepare Booking Data

Place your Excel booking file in the project root:
- `Existing Booking.xlsx` (default)
- Or update `BOOKING_FILE` in `.env`

Expected columns:
- Status (HK, UC, CL, TKT, RQ)
- Lead Pax Name
- Reference No
- Client Name
- City
- Cancellation Deadline

### 5. Run the Application

```bash
python main.py
```

Or:
```bash
uvicorn main:app --reload
```

The server will start at: `http://127.0.0.1:8000`

## 🧪 Testing

### Run Test Script

```bash
python test_agent.py
```

This will test:
- Tool calling functionality
- Language normalization (confirmed → HK, etc.)
- RAG context retrieval
- Multiple filter combinations
- Error handling

### Manual Testing

1. **Check Status**
   ```bash
   curl http://localhost:8000/status
   ```

2. **Test Chat Endpoint**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Show me confirmed bookings"}'
   ```

3. **Open Web Interface** (if available)
   - Navigate to `http://localhost:8000`

## 💬 Sample Queries

Try these natural language queries:

- **"Show me all confirmed bookings"** → Filters status=HK
- **"Find ticketed reservations"** → Filters status=TKT
- **"Bookings for pax srk"** → Filters pax_name containing "srk"
- **"Show ref 12345"** → Filters ref_no=12345
- **"Confirmed bookings for client ABC"** → Multiple filters
- **"What does HK mean?"** → Uses RAG knowledge base

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI       │  /chat endpoint
│   (main.py)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Service    │  LangChain Agent
│ (llm_service.py)│  + Tool Calling
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────┐
│ Vector  │ │ get_bookings │  Tool
│ Service │ │     Tool     │
│ (RAG)   │ └──────┬───────┘
└─────────┘        │
                   ▼
              ┌──────────────┐
              │ Data Service │
              │  (Excel I/O) │
              └──────────────┘
```

## 📁 Project Structure

```
chatbot/
├── main.py                 # FastAPI application
├── config.py              # Configuration & credentials
├── llm_service.py         # AWS Bedrock Claude + LangChain
├── vector_service.py      # Chroma vector DB + embeddings
├── data_service.py        # Excel data access
├── knowledge_base.txt     # Domain knowledge for RAG
├── test_agent.py          # Test script
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration
├── Existing Booking.xlsx  # Booking data (not in repo)
└── chroma_db/            # Vector store (auto-created)
```

## 🔧 Configuration

### Environment Variables (.env)

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region for Bedrock | `us-east-1` |
| `AWS_ACCESS_KEY_ID` | AWS access key | - |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | - |
| `BEDROCK_MODEL_ID` | Claude model ID | `anthropic.claude-sonnet-4-20250514` |
| `CHROMA_DB_DIR` | Vector DB directory | `./chroma_db` |
| `EMBEDDING_MODEL` | Local embedding model | `sentence-transformers/all-mpnet-base-v2` |
| `BOOKING_FILE` | Excel data file | `Existing Booking.xlsx` |

## 📊 Booking Status Codes

| Code | Meaning |
|------|---------|
| HK | Confirmed |
| UC | On Request (Flight) |
| CL | Cancelled |
| TKT | Ticketed |
| RQ | On Request (Hotel) |

The chatbot understands both codes (`HK`) and natural language (`confirmed`).

## 🛡️ Behavior Rules

1. **Never fabricates data** - Always uses `get_bookings` tool
2. **Extracts filters** - Parses natural language queries
3. **Handles vagueness** - Calls tool with available info or asks for clarification
4. **Clear on no results** - Explicitly states when no bookings found
5. **Language normalization** - Maps informal terms to codes

## 🚨 Troubleshooting

### AWS Bedrock Connection Errors

**Error**: `AccessDeniedException`
- Ensure IAM role has `bedrock:InvokeModel` permission
- Verify Bedrock is enabled in your region
- Request access to Claude Sonnet 4.5 model

**Error**: `ResourceNotFoundException`
- Check `BEDROCK_MODEL_ID` is correct
- Verify model is available in your region

### Vector Database Issues

**No knowledge base loaded**
- Ensure `knowledge_base.txt` exists
- Check `CHROMA_DB_DIR` has write permissions

### Data Service Errors

**Excel file not found**
- Verify `BOOKING_FILE` path is correct
- Ensure file exists in project root
- Try fallback: `flight Booking.xlsx`

## 📝 API Endpoints

### POST /chat
Chat with the AI assistant.

**Request**:
```json
{
  "message": "Show me confirmed bookings"
}
```

**Response**:
```json
{
  "response": "I found 15 confirmed bookings...",
  "response_type": "text"
}
```

### GET /status
System health check.

**Response**:
```json
{
  "status": "online",
  "llm_provider": "AWS Bedrock",
  "llm_model": "anthropic.claude-sonnet-4-20250514",
  "vector_db": "Chroma",
  "data_found": true
}
```

### POST /feedback
Submit user feedback.

**Request**:
```json
{
  "question": "Show me bookings",
  "helpful": true
}
```

## 🔐 Security Notes

- **Never commit `.env`** to version control
- **Store AWS credentials securely**
- **Use IAM roles** with minimal permissions
- **Restrict API access** in production

## 📚 Dependencies

Key packages:
- `boto3` - AWS SDK
- `langchain` >= 0.1.0
- `langchain-aws` >= 0.1.0
- `chromadb` >= 0.4.22
- `sentence-transformers` >= 2.3.0
- `fastapi` >= 0.109.0

See `requirements.txt` for full list.

## 🎯 Next Steps

1. **Add frontend UI** - Create `static/index.html` for chat interface
2. **Add logging** - Implement structured logging for production
3. **Add authentication** - Secure API endpoints
4. **Deploy** - Use AWS ECS, Lambda, or EC2
5. **Monitor** - Add CloudWatch metrics

## 📄 License

Proprietary - For internal use only.

## 🤝 Support

For issues or questions, contact the development team.
