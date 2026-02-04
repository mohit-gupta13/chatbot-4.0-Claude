# Travel Agent Back-Office Chatbot (GenAI)

A specialized chatbot designed for travel agency internal staff to query flight and hotel bookings using natural language.

## 🎯 Features

- **Direct Chat**: No separate modes. Simply type your query to search bookings.
- **Flight & Hotel Support**: Unified search across different service types.
- **Status Code Mapping**: Understands industry codes:
  - **HK**: Confirmed
  - **UC**: On Request
  - **CL**: Cancelled
  - **TKT**: Ticketed
  - **RQ**: On Request (Hotel)
- **Column Intelligence**: Searches across Reference No, Lead Pax, Client Name, City, and more.
- **Data Table View**: Displays detailed booking information in an easy-to-read format.

## 🚀 Getting Started

1.  **Configure API Key**:
    *   Create a `.env` file in the root directory.
    *   Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`.

2.  **Data Source**:
    *   Ensure `Existing Booking.xlsx` is present in the root directory.

3.  **Run Application**:
    ```bash
    uv run main.py
    ```
4.  **Access**:
    Open `http://127.0.0.1:8000` in your browser.

## 🏗️ Architecture

- **FastAPI**: Backend web framework.
- **OpenAI (GPT-4)**: Powers the natural language understanding and function calling.
- **Pandas**: Handles fast and efficient Excel data processing.
- **Vanilla JS/CSS**: Clean and fast user interface.

## 📁 Project Structure

- `main.py`: Entry point and API endpoints.
- `config.py`: Configuration and environment settings.
- `data_service.py`: Business logic for data retrieval and filtering.
- `llm_service.py`: AI logic for intent extraction and function calling.
- `static/`: Frontend assets (HTML/CSS).

## 🔒 Security

- **LLM Restriction**: The AI only extracts parameters; it never directly reads or writes to the database/files.
- **Static Schema**: Data operations are performed by pre-defined backend functions.
