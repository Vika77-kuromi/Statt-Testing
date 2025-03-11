# Statt-Testing

## First Thing to Do--Clone the Repository

## Installing Dependencies
Install all the required packages for all the codes to be able to run
```bash
pip install -r requirements.txt
```

## Setting Up OpenAI API Key
This chatbox requires an OpenAI API key to function. Follow these steps:

1. Users need to create a new file named ".env" in the same directory where the chatbot code is located.
   ```bash
   touch .env
   nano .env
   ```
2. Add the following line:
   OPENAI_API_KEY=your-api-key-here
3. Replace "your-api-key-here" with your actual OpenAI API key.
4. Save the file (CTRL+X, then Y, then ENTER).
5. After setting up your API key, run the chatbot with:
   ```bash
   python app.py
   ```


